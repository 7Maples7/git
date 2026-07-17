# -*- coding: utf-8 -*-
"""Radar three-class recognizer worker for C++ bridge via JSON lines.

Protocol:
  stdin  <- {"id": 1, "cmd": "init", "checkpoint_path": "...", "device": "auto"}
  stdout -> {"id": 1, "ok": true, "error_code": "", "error_msg": "", "data": {...}}

The recognize command accepts one DClsEcho binary payload encoded as base64:
  {"id": 2, "cmd": "recognize_echo", "echo_blob_b64": "..."}
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

_WORKER_DIR = Path(__file__).resolve().parent
if str(_WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(_WORKER_DIR))

import numpy as np
import torch

try:
    from .dataset import (
        META_FEATURE_NAMES,
        RadarSample,
        _as_1d_float,
        _echo_abs_series,
        _pc_spectrum_abs_series,
        _pc_spectrum_db_mean,
        _standardize_profile,
        collate_radar_batch,
        decode_dcls_echo_blob,
    )
    from .model import RadarThreeBranchNet
except ImportError:
    from dataset import (  # type: ignore
        META_FEATURE_NAMES,
        RadarSample,
        _as_1d_float,
        _echo_abs_series,
        _pc_spectrum_abs_series,
        _pc_spectrum_db_mean,
        _standardize_profile,
        collate_radar_batch,
        decode_dcls_echo_blob,
    )
    from model import RadarThreeBranchNet  # type: ignore


LEGACY_META_FEATURE_NAMES = (
    "log_range_m",
    "log_pw_us",
    "log_prt_us",
    "log_prt_nbr",
    "log_pluse_band",
    "log_sample_freq",
    "log_prf_hz",
    "duty",
    "log_time_bandwidth",
)


def _configure_stdio_utf8() -> None:
    for stream_name in ("stdin", "stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def _fmt_category(value: Any) -> str:
    parsed = _maybe_int(value)
    if parsed is None:
        return str(value)
    if parsed < 0:
        return str(parsed)
    return f"0x{parsed:X}"


def _maybe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text, 0)
    except ValueError:
        return None


def _safe_log1p(value: float) -> float:
    return float(np.log1p(max(float(value), 0.0)))


def _torch_load_checkpoint(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(str(path), map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("unsupported checkpoint format, expected dict payload")
    return payload


def _resolve_device(device_name: str) -> torch.device:
    name = str(device_name or "auto").strip() or "auto"
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def _default_meta_names_for_dim(meta_dim: int) -> tuple[str, ...]:
    if meta_dim == len(META_FEATURE_NAMES):
        return tuple(META_FEATURE_NAMES)
    if meta_dim == len(LEGACY_META_FEATURE_NAMES):
        return LEGACY_META_FEATURE_NAMES
    raise ValueError(
        "checkpoint missing meta_feature_names and unsupported meta_dim="
        f"{meta_dim}; supported dims are {len(META_FEATURE_NAMES)} and "
        f"{len(LEGACY_META_FEATURE_NAMES)}"
    )


def _coerce_meta_feature_names(value: Any, meta_dim: int) -> tuple[str, ...]:
    if value is None or value == "":
        return _default_meta_names_for_dim(meta_dim)
    if isinstance(value, str):
        names = tuple(x.strip() for x in value.split(",") if x.strip())
    else:
        names = tuple(str(x).strip() for x in value if str(x).strip())
    if len(names) != meta_dim:
        raise ValueError(
            f"meta_feature_names length ({len(names)}) does not match meta_dim ({meta_dim})"
        )
    return names


def _build_radar_sample(decoded: Mapping[str, Any], sample_id: str = "inline") -> RadarSample:
    mtd_echo = _as_1d_float(decoded.get("mtd_echo", ()))
    pc_echo = _as_1d_float(decoded.get("pc_echo", ()))
    prt_nbr = float(decoded.get("prt_nbr", 0.0) or 0.0)
    if prt_nbr <= 0 and mtd_echo.size > 0:
        prt_nbr = float(mtd_echo.size)
    return RadarSample(
        mtd_echo=mtd_echo,
        pc_echo=pc_echo,
        label=None,
        sample_id=sample_id,
        range_m=float(decoded.get("range", decoded.get("range_m", 0.0)) or 0.0),
        pw_us=float(decoded.get("pw_us", 0.0) or 0.0),
        prt_us=float(decoded.get("prt_us", 0.0) or 0.0),
        prt_nbr=prt_nbr,
        pluse_band=float(decoded.get("pluse_band", 0.0) or 0.0),
        sample_freq=float(decoded.get("sample_freq", 0.0) or 0.0),
        mtd_row=int(decoded.get("mtd_row", 0) or 0),
        mtd_col=int(decoded.get("mtd_col", 0) or 0),
        mtd_ch=int(decoded.get("mtd_ch", 0) or 0),
        pc_row=int(decoded.get("pc_row", 0) or 0),
        pc_col=int(decoded.get("pc_col", 0) or 0),
        pc_ch=int(decoded.get("pc_ch", 0) or 0),
    )


def _meta_feature_map(sample: RadarSample, mtd_echo: np.ndarray, pc_echo: np.ndarray) -> dict[str, float]:
    pw_us = max(float(sample.pw_us), 0.0)
    prt_us = max(float(sample.prt_us), 0.0)
    prt_nbr = max(float(sample.prt_nbr), 0.0)
    band_mhz = max(float(sample.pluse_band), 0.0)
    sample_freq_mhz = max(float(sample.sample_freq), 0.0)
    duty = pw_us / max(prt_us, 1e-6)
    prf_hz = 1.0e6 / prt_us if prt_us > 0.0 else 0.0
    time_bandwidth = pw_us * band_mhz
    mtd_energy = float(np.mean(np.square(np.abs(mtd_echo)))) if mtd_echo.size else 0.0
    pc_spec_db_mean = _pc_spectrum_db_mean(sample.pc_echo, sample.pc_row, sample.pc_col, sample.pc_ch)
    return {
        "log_range_m": _safe_log1p(sample.range_m),
        "log_pw_us": _safe_log1p(pw_us),
        "log_prt_us": _safe_log1p(prt_us),
        "log_prt_nbr": _safe_log1p(prt_nbr),
        "log_pluse_band": _safe_log1p(band_mhz),
        "log_pulse_band": _safe_log1p(band_mhz),
        "log_sample_freq": _safe_log1p(sample_freq_mhz),
        "log_prf_hz": _safe_log1p(prf_hz),
        "duty": float(duty),
        "log_time_bandwidth": _safe_log1p(time_bandwidth),
        "log_mtd_energy": _safe_log1p(mtd_energy),
        "pc_spectrum_db_mean": float(pc_spec_db_mean),
        "log_pc_len": _safe_log1p(float(len(pc_echo))),
    }


def _build_batch_item(sample: RadarSample, meta_feature_names: Sequence[str]) -> dict[str, Any]:
    mtd_echo = _echo_abs_series(sample.mtd_echo, sample.mtd_row, sample.mtd_col, sample.mtd_ch)
    pc_echo = _echo_abs_series(sample.pc_echo, sample.pc_row, sample.pc_col, sample.pc_ch)
    pc_spectrum = _pc_spectrum_abs_series(sample.pc_echo, sample.pc_row, sample.pc_col, sample.pc_ch)
    meta_values = _meta_feature_map(sample, mtd_echo, pc_echo)
    unknown = [name for name in meta_feature_names if name not in meta_values]
    if unknown:
        raise ValueError(f"unsupported meta feature(s): {unknown}")
    return {
        "mtd_echo": _standardize_profile(mtd_echo),
        "pc_echo": _standardize_profile(pc_echo),
        "pc_spectrum": _standardize_profile(pc_spectrum),
        "meta": np.asarray([meta_values[name] for name in meta_feature_names], dtype=np.float32),
        "label": -1,
        "range_m": float(sample.range_m),
        "sample_id": sample.sample_id,
    }


def _write_response(
    req_id: Any,
    ok: bool,
    data: Optional[Dict[str, Any]] = None,
    *,
    error_code: str = "",
    error_msg: str = "",
) -> None:
    payload = {
        "id": req_id,
        "ok": bool(ok),
        "error_code": str(error_code),
        "error_msg": str(error_msg),
        "data": data if data is not None else {},
    }
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


class RadarThreeRecognizerRuntime:
    def __init__(self) -> None:
        self.model: Optional[RadarThreeBranchNet] = None
        self.device: Optional[torch.device] = None
        self.id_to_label: dict[int, str] = {}
        self.class_to_target_type: dict[int, int] = {}
        self.meta_feature_names: tuple[str, ...] = tuple(META_FEATURE_NAMES)
        self.strict_model_load = True

    def init_from_request(self, req: Mapping[str, Any]) -> Dict[str, Any]:
        checkpoint_path = Path(str(req.get("checkpoint_path", ""))).expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

        device = _resolve_device(str(req.get("device", "auto")))
        strict_model_load = bool(req.get("strict_model_load", True))
        payload = _torch_load_checkpoint(checkpoint_path)
        model_state = payload.get("model_state_dict", payload)
        if not isinstance(model_state, Mapping):
            raise ValueError("checkpoint missing model_state_dict")

        id_to_label_raw = payload.get("id_to_label") or {}
        id_to_label = {int(k): str(v) for k, v in dict(id_to_label_raw).items()}
        label_to_id_raw = payload.get("label_to_id") or {}
        label_to_id = {str(k): int(v) for k, v in dict(label_to_id_raw).items()}

        if id_to_label:
            num_classes = len(id_to_label)
        elif label_to_id:
            num_classes = max(label_to_id.values()) + 1
            id_to_label = {idx: str(raw) for raw, idx in label_to_id.items()}
        else:
            head_weight = model_state.get("head.4.weight")
            if head_weight is None:
                raise ValueError("checkpoint missing id_to_label/label_to_id and head.4.weight")
            num_classes = int(head_weight.shape[0])
            id_to_label = {idx: str(idx) for idx in range(num_classes)}

        meta_dim = int(payload.get("meta_dim", len(META_FEATURE_NAMES)))
        meta_feature_names = _coerce_meta_feature_names(payload.get("meta_feature_names"), meta_dim)
        config = payload.get("config") or {}
        dropout = float(config.get("dropout", 0.2)) if isinstance(config, Mapping) else 0.2

        model = RadarThreeBranchNet(num_classes=num_classes, meta_dim=meta_dim, dropout=dropout).to(device)
        incompatible = model.load_state_dict(model_state, strict=strict_model_load)
        model.eval()

        class_to_target_type: dict[int, int] = {}
        for raw_label, class_id in label_to_id.items():
            raw_int = _maybe_int(raw_label)
            if raw_int is not None:
                class_to_target_type[int(class_id)] = raw_int
        for class_id in range(num_classes):
            class_to_target_type.setdefault(class_id, class_id)

        self.model = model
        self.device = device
        self.id_to_label = id_to_label
        self.class_to_target_type = class_to_target_type
        self.meta_feature_names = meta_feature_names
        self.strict_model_load = strict_model_load

        data: Dict[str, Any] = {
            "checkpoint_path": str(checkpoint_path),
            "device": str(device),
            "num_classes": int(num_classes),
            "id_to_label": {str(k): v for k, v in sorted(id_to_label.items())},
            "class_to_target_type": {str(k): int(v) for k, v in sorted(class_to_target_type.items())},
            "label_mapping_hex": {
                _fmt_category(v): int(k) for k, v in sorted(class_to_target_type.items())
            },
            "meta_dim": int(meta_dim),
            "meta_feature_names": list(meta_feature_names),
            "strict_model_load": bool(strict_model_load),
        }
        if not strict_model_load:
            data["missing_keys"] = list(getattr(incompatible, "missing_keys", []))
            data["unexpected_keys"] = list(getattr(incompatible, "unexpected_keys", []))
        return data

    @torch.no_grad()
    def recognize_echo(self, echo_blob_b64: str) -> Dict[str, Any]:
        if self.model is None or self.device is None:
            raise RuntimeError("recognizer is not initialized; call init first")

        try:
            echo_blob = base64.b64decode(echo_blob_b64.encode("ascii"), validate=True)
        except Exception as exc:
            raise ValueError(f"invalid base64 echo blob: {exc}") from exc

        decoded = decode_dcls_echo_blob(echo_blob)
        sample = _build_radar_sample(decoded)
        item = _build_batch_item(sample, self.meta_feature_names)
        batch = collate_radar_batch([item])
        for key, value in list(batch.items()):
            if torch.is_tensor(value):
                batch[key] = value.to(self.device)

        logits = self.model(batch)
        probs = torch.softmax(logits[0], dim=0)
        pred_class_id = int(torch.argmax(probs).item())
        pred_target_type = int(self.class_to_target_type.get(pred_class_id, pred_class_id))
        pred_label = str(self.id_to_label.get(pred_class_id, pred_class_id))

        topk = min(3, int(probs.shape[0]))
        score_values, score_indices = torch.topk(probs, k=topk)
        top1_score = float(score_values[0].item())
        top2_score = float(score_values[1].item()) if topk >= 2 else 0.0

        probabilities = {
            str(self.id_to_label.get(i, i)): float(probs[i].detach().cpu().item())
            for i in range(int(probs.shape[0]))
        }
        probabilities_by_class_id = {
            str(i): float(probs[i].detach().cpu().item())
            for i in range(int(probs.shape[0]))
        }
        topk_items = []
        for score, class_idx in zip(score_values.detach().cpu().tolist(), score_indices.detach().cpu().tolist()):
            class_id = int(class_idx)
            target_type = int(self.class_to_target_type.get(class_id, class_id))
            topk_items.append(
                {
                    "class_id": class_id,
                    "class_id_hex": _fmt_category(class_id),
                    "target_type": target_type,
                    "target_type_hex": _fmt_category(target_type),
                    "label": str(self.id_to_label.get(class_id, class_id)),
                    "score": float(score),
                }
            )

        return {
            "pred_target_type": int(pred_target_type),
            "pred_target_type_hex": _fmt_category(pred_target_type),
            "pred_class_id": int(pred_class_id),
            "pred_class_id_hex": _fmt_category(pred_class_id),
            "pred_label": pred_label,
            "score": float(top1_score),
            "top1_score": float(top1_score),
            "top2_score": float(top2_score),
            "margin": float(top1_score - top2_score),
            "top1_target_type": int(topk_items[0]["target_type"]) if topk_items else int(pred_target_type),
            "top1_target_type_hex": str(topk_items[0]["target_type_hex"]) if topk_items else _fmt_category(pred_target_type),
            "probabilities": probabilities,
            "probabilities_by_class_id": probabilities_by_class_id,
            "topk": topk_items,
            "binary_uav_mode": False,
            "uav_threshold_used": None,
        }


def _main() -> int:
    runtime = RadarThreeRecognizerRuntime()

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue

        req_id: Any = None
        try:
            req = json.loads(line)
            if not isinstance(req, dict):
                raise ValueError("request must be a JSON object")
            req_id = req.get("id")
            cmd = str(req.get("cmd", "")).strip()
            if not cmd:
                raise ValueError("missing cmd")

            if cmd == "ping":
                _write_response(req_id, True, {"status": "alive"})
                continue
            if cmd == "init":
                _write_response(req_id, True, runtime.init_from_request(req))
                continue
            if cmd == "recognize_echo":
                _write_response(req_id, True, runtime.recognize_echo(str(req.get("echo_blob_b64", ""))))
                continue
            if cmd == "shutdown":
                _write_response(req_id, True, {"status": "bye"})
                break

            raise ValueError(f"unsupported cmd: {cmd}")
        except Exception as exc:
            _write_response(req_id, False, error_code="WORKER_ERROR", error_msg=str(exc))

    return 0


if __name__ == "__main__":
    _configure_stdio_utf8()
    raise SystemExit(_main())
