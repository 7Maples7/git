# -*- coding: utf-8 -*-
"""Dataset utilities for DClsEcho radar target classification."""

from __future__ import annotations

import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


DATA_HEAD = 0xFF88FF88
DATA_TAILE = 0xFF77FF77
DEFAULT_TABLE_NAME = "mSignalProcEcho"
DEFAULT_CLASS_NAMES = ("uav", "bird", "clutter")
PC_LEN = 129
META_FEATURE_NAMES = (
    "log_range_m",
    "log_pw_us",
    "log_prt_us",
    "log_prt_nbr",
    "duty",
    "log_time_bandwidth",
    "log_mtd_energy",
    "pc_spectrum_db_mean",
    "log_pc_len",
)
META_DIM = len(META_FEATURE_NAMES)
DEFAULT_META_FEATURE_NAMES = META_FEATURE_NAMES

_DB_SUFFIXES = {".db", ".sqlite", ".sqlite3"}
_PREFIX_STRUCT = struct.Struct("<q6f3ii")
_PC_META_STRUCT = struct.Struct("<3ii")
_TAIL_STRUCT = struct.Struct("<q")


LABEL_ALIASES = {
    "uav": "uav",
    "drone": "uav",
    "bird": "bird",
    "clutter": "clutter",
    "noise": "clutter",
    "background": "clutter",
}


@dataclass
class RadarSample:
    """One DClsEcho sample prepared for the three-class network."""

    mtd_echo: np.ndarray
    pc_echo: np.ndarray
    label: Any | None = None
    sample_id: str = ""
    range_m: float = 0.0
    pw_us: float = 0.0
    prt_us: float = 0.0
    prt_nbr: float = 0.0
    pluse_band: float = 0.0
    sample_freq: float = 0.0
    mtd_row: int = 0
    mtd_col: int = 0
    mtd_ch: int = 0
    pc_row: int = 0
    pc_col: int = 0
    pc_ch: int = 0


class DClsEchoParseError(RuntimeError):
    """Raised when a DClsEcho binary payload cannot be decoded."""


def canonical_label(label: Any) -> Any:
    if isinstance(label, bytes):
        label = label.decode("utf-8")
    if isinstance(label, str):
        key = label.strip().lower()
        return LABEL_ALIASES.get(key, key)
    if isinstance(label, np.generic):
        return label.item()
    return label


def build_label_mapping(labels: Iterable[Any], class_names: Sequence[str] = DEFAULT_CLASS_NAMES) -> dict[Any, int]:
    canonical = [canonical_label(x) for x in labels]
    if all(isinstance(x, (int, np.integer)) for x in canonical):
        return {raw: idx for idx, raw in enumerate(sorted({int(x) for x in canonical}))}

    mapping: dict[Any, int] = {}
    for idx, name in enumerate(class_names):
        mapping[canonical_label(name)] = idx
    for label in canonical:
        if label not in mapping:
            mapping[label] = len(mapping)
    return mapping


def resolve_meta_feature_indices(selection: str | Sequence[str] | Sequence[int] | None = None) -> tuple[list[int], list[str]]:
    """Resolve selected meta feature names/indices to stable feature indices.

    Empty selection means all currently defined meta features. The GUI passes the
    selected feature names explicitly so checkpoints can recreate the same input.
    """
    if selection is None or selection == "":
        indices = list(range(len(META_FEATURE_NAMES)))
        return indices, [META_FEATURE_NAMES[i] for i in indices]

    if isinstance(selection, str):
        items: list[Any] = [x.strip() for x in selection.split(",") if x.strip()]
    else:
        items = list(selection)

    name_to_index = {name: i for i, name in enumerate(META_FEATURE_NAMES)}
    indices: list[int] = []
    for item in items:
        if isinstance(item, (int, np.integer)):
            idx = int(item)
        else:
            text = str(item).strip()
            idx = int(text) if text.isdigit() else name_to_index[text]
        if idx < 0 or idx >= len(META_FEATURE_NAMES):
            raise ValueError(f"meta feature index out of range: {idx}")
        if idx not in indices:
            indices.append(idx)
    if not indices:
        raise ValueError("at least one meta feature must be selected")
    return indices, [META_FEATURE_NAMES[i] for i in indices]


def _quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    sql = "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1"
    return conn.execute(sql, (str(table_name),)).fetchone() is not None


def _pick_column(cols: Sequence[str], candidates: Sequence[str], suffix: str | None = None) -> str | None:
    col_by_lower = {c.lower(): c for c in cols}
    for candidate in candidates:
        found = col_by_lower.get(candidate.lower())
        if found is not None:
            return found
    if suffix:
        suffix_l = suffix.lower()
        for col in cols:
            if col.lower().endswith(suffix_l):
                return col
    return None


def _detect_columns(conn: sqlite3.Connection, table_name: str) -> dict[str, str | None]:
    rows = conn.execute(f"PRAGMA table_info({_quote_ident(table_name)})").fetchall()
    if not rows:
        raise ValueError(f"Table not found or empty schema: {table_name}")

    cols = [str(row[1]) for row in rows]
    data_col = _pick_column(cols, ("_data", "data"), "_data")
    if data_col is None:
        raise ValueError(f"Cannot detect DClsEcho data column in '{table_name}', columns={cols}")
    return {
        "id": _pick_column(cols, ("_id", "id"), "_id"),
        "time": _pick_column(cols, ("_timeWrite", "timeWrite", "time_write"), "timewrite"),
        "type": _pick_column(cols, ("_type", "type", "label", "target"), "_type"),
        "data": data_col,
    }


def decode_dcls_echo_blob(payload: bytes | bytearray | memoryview) -> dict[str, Any]:
    """Decode bytes written by DClsEcho::data()."""
    if payload is None:
        raise DClsEchoParseError("Payload is None.")
    raw = bytes(payload)
    if len(raw) < _PREFIX_STRUCT.size + _PC_META_STRUCT.size + _TAIL_STRUCT.size:
        raise DClsEchoParseError(f"Payload too small: {len(raw)} bytes.")

    offset = 0
    (
        header,
        range_m,
        pw_us,
        prt_us,
        prt_nbr,
        pluse_band,
        sample_freq,
        mtd_row,
        mtd_col,
        mtd_ch,
        mtd_echo_size,
    ) = _PREFIX_STRUCT.unpack_from(raw, offset)
    offset += _PREFIX_STRUCT.size

    if header != DATA_HEAD:
        raise DClsEchoParseError(f"Invalid head: {header:#x}")
    if mtd_row < 0 or mtd_col < 0 or mtd_ch < 0 or mtd_echo_size < 0:
        raise DClsEchoParseError(
            f"Invalid mtd metadata: row={mtd_row}, col={mtd_col}, ch={mtd_ch}, size={mtd_echo_size}"
        )

    mtd_bytes = int(mtd_echo_size) * 4
    if offset + mtd_bytes > len(raw):
        raise DClsEchoParseError(f"Invalid mtd echo size: {mtd_echo_size}, remain={len(raw) - offset}")
    mtd_echo = struct.unpack_from(f"<{int(mtd_echo_size)}f", raw, offset) if mtd_echo_size else tuple()
    offset += mtd_bytes

    if offset + _PC_META_STRUCT.size > len(raw):
        raise DClsEchoParseError("Not enough bytes for pc metadata.")
    pc_row, pc_col, pc_ch, pc_echo_size = _PC_META_STRUCT.unpack_from(raw, offset)
    offset += _PC_META_STRUCT.size

    if pc_row < 0 or pc_col < 0 or pc_ch < 0 or pc_echo_size < 0:
        raise DClsEchoParseError(
            f"Invalid pc metadata: row={pc_row}, col={pc_col}, ch={pc_ch}, size={pc_echo_size}"
        )

    pc_bytes = int(pc_echo_size) * 4
    if offset + pc_bytes + _TAIL_STRUCT.size > len(raw):
        raise DClsEchoParseError(f"Invalid pc echo size: {pc_echo_size}, remain={len(raw) - offset}")
    pc_echo = struct.unpack_from(f"<{int(pc_echo_size)}f", raw, offset) if pc_echo_size else tuple()
    offset += pc_bytes

    if offset + _TAIL_STRUCT.size != len(raw):
        raise DClsEchoParseError(f"Unexpected trailing bytes: total={len(raw)}, offset={offset}")
    (tail,) = _TAIL_STRUCT.unpack_from(raw, offset)
    if tail != DATA_TAILE:
        raise DClsEchoParseError(f"Invalid tail: {tail:#x}")

    return {
        "range": float(range_m),
        "pw_us": float(pw_us),
        "prt_us": float(prt_us),
        "prt_nbr": float(prt_nbr),
        "pluse_band": float(pluse_band),
        "sample_freq": float(sample_freq),
        "mtd_row": int(mtd_row),
        "mtd_col": int(mtd_col),
        "mtd_ch": int(mtd_ch),
        "mtd_echo": tuple(float(x) for x in mtd_echo),
        "pc_row": int(pc_row),
        "pc_col": int(pc_col),
        "pc_ch": int(pc_ch),
        "pc_echo": tuple(float(x) for x in pc_echo),
    }


def _as_1d_float(x: Any) -> np.ndarray:
    arr = np.asarray(x)
    if np.iscomplexobj(arr):
        arr = np.abs(arr)
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _fix_len(x: Any, length: int | None) -> np.ndarray:
    arr = _as_1d_float(x)
    if length is None or length <= 0:
        return arr
    out = np.zeros(int(length), dtype=np.float32)
    n = min(int(arr.size), int(length))
    if n > 0:
        out[:n] = arr[:n]
    return out


def _fix_pc_len(x: Any, length: int = PC_LEN) -> np.ndarray:
    return _fix_len(x, length)


def _standardize_profile(x: np.ndarray) -> np.ndarray:
    mag = np.log1p(np.abs(np.asarray(x, dtype=np.float32)))
    mean = float(mag.mean()) if mag.size else 0.0
    std = float(mag.std()) if mag.size else 1.0
    return ((mag - mean) / max(std, 1e-5)).astype(np.float32)


def _reshape_echo_channels(raw: Any, row: int, col: int, ch: int) -> np.ndarray:
    arr = _as_1d_float(raw)
    row = max(0, int(row))
    col = max(0, int(col))
    ch = max(1, int(ch))
    expected = row * col * ch
    if arr.size == 0:
        return np.zeros((0, ch), dtype=np.float32)
    if expected > 0 and arr.size == expected:
        return arr.reshape(row * col, ch)

    sample_count = int(arr.size) // ch
    if sample_count > 0:
        used = sample_count * ch
        return arr[:used].reshape(sample_count, ch)
    return arr.reshape(-1, 1)


def _echo_abs_series(raw: Any, row: int, col: int, ch: int) -> np.ndarray:
    data = _reshape_echo_channels(raw, row, col, ch)
    if data.size == 0:
        return np.asarray([], dtype=np.float32)
    if data.shape[1] >= 2:
        return np.sqrt(np.square(data[:, 0]) + np.square(data[:, 1])).astype(np.float32)
    return np.abs(data[:, 0]).astype(np.float32)


def _pc_spectrum_abs_series(raw: Any, row: int, col: int, ch: int) -> np.ndarray:
    data = _reshape_echo_channels(raw, row, col, ch)
    if data.size == 0:
        return np.asarray([], dtype=np.float32)
    if data.shape[1] >= 2:
        signal = data[:, 0].astype(np.float64, copy=False) + 1j * data[:, 1].astype(np.float64, copy=False)
    else:
        signal = data[:, 0].astype(np.float64, copy=False)
    spectrum = np.fft.fftshift(np.fft.fft(signal))
    return np.abs(spectrum).astype(np.float32)


def _pc_spectrum_db_mean(raw: Any, row: int, col: int, ch: int) -> float:
    data = _reshape_echo_channels(raw, row, col, ch)
    if data.size == 0 or data.shape[1] < 2:
        return 0.0

    i_data = data[:, 0].astype(np.float64, copy=False)
    q_data = data[:, 1].astype(np.float64, copy=False)
    pc_complex = i_data + 1j * q_data
    pc_spec = np.fft.fftshift(np.fft.fft(pc_complex))
    pc_spec_db = 20.0 * np.log10(np.maximum(np.abs(pc_spec), 1.0e-12))
    return float(np.mean(pc_spec_db)) if pc_spec_db.size else 0.0


def _safe_log1p(value: float) -> float:
    return float(np.log1p(max(float(value), 0.0)))


def _meta_features(sample: RadarSample, mtd_echo: np.ndarray, pc_echo: np.ndarray) -> np.ndarray:
    pw_us = max(float(sample.pw_us), 0.0)
    prt_us = max(float(sample.prt_us), 0.0)
    prt_nbr = max(float(sample.prt_nbr), 0.0)
    band_mhz = max(float(sample.pluse_band), 0.0)
    duty = pw_us / max(prt_us, 1e-6)
    time_bandwidth = pw_us * band_mhz
    mtd_energy = float(np.mean(np.square(np.abs(mtd_echo)))) if mtd_echo.size else 0.0
    pc_spectrum_db_mean = _pc_spectrum_db_mean(sample.pc_echo, sample.pc_row, sample.pc_col, sample.pc_ch)
    return np.asarray(
        [
            _safe_log1p(sample.range_m),
            _safe_log1p(pw_us),
            _safe_log1p(prt_us),
            _safe_log1p(prt_nbr),
            float(duty),
            _safe_log1p(time_bandwidth),
            _safe_log1p(mtd_energy),
            pc_spectrum_db_mean,
            _safe_log1p(float(len(pc_echo))),
        ],
        dtype=np.float32,
    )


def _record_from_decoded(decoded: Mapping[str, Any], label: Any | None, sample_id: str) -> RadarSample:
    prt_nbr = float(decoded.get("prt_nbr", 0.0) or 0.0)
    mtd_echo = _as_1d_float(decoded.get("mtd_echo", ()))
    if prt_nbr <= 0 and mtd_echo.size > 0:
        prt_nbr = float(mtd_echo.size)
    return RadarSample(
        mtd_echo=mtd_echo,
        pc_echo=_as_1d_float(decoded.get("pc_echo", ())),
        label=canonical_label(label) if label is not None else None,
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


def _records_from_db_file(path: Path, require_label: bool, table_name: str = DEFAULT_TABLE_NAME) -> list[RadarSample]:
    conn = sqlite3.connect(str(path))
    try:
        if not _table_exists(conn, table_name):
            return []
        cols = _detect_columns(conn, table_name)
        if require_label and cols["type"] is None:
            raise ValueError(f"Cannot detect label/type column in '{table_name}' for {path}")

        select_cols = ["rowid"]
        aliases = ["rowid"]
        for alias in ("id", "time", "type", "data"):
            col = cols[alias]
            if col is not None:
                select_cols.append(_quote_ident(col))
                aliases.append(alias)
        sql = f"SELECT {', '.join(select_cols)} FROM {_quote_ident(table_name)} ORDER BY rowid"
        rows = conn.execute(sql).fetchall()
    finally:
        conn.close()

    records: list[RadarSample] = []
    for row in rows:
        values = dict(zip(aliases, row))
        blob = values.get("data")
        if blob is None:
            continue
        try:
            decoded = decode_dcls_echo_blob(blob)
        except DClsEchoParseError:
            continue
        raw_label = values.get("type") if require_label else None
        rowid = values.get("rowid", len(records))
        rec_id = values.get("id")
        if rec_id is None or int(rec_id) == int(rowid):
            sample_key = str(rowid)
        else:
            sample_key = f"rowid={rowid},id={rec_id}"
        records.append(_record_from_decoded(decoded, raw_label, f"{path}#{sample_key}"))
    return records


def _pick_key(data: Mapping[str, Any], candidates: Sequence[str], required: bool = True) -> str | None:
    for key in candidates:
        if key in data:
            return key
    if required:
        raise KeyError(f"missing required key, candidates={candidates}")
    return None


def _maybe_len(x: Any) -> int | None:
    arr = np.asarray(x, dtype=object if isinstance(x, np.ndarray) and x.dtype == object else None)
    if arr.ndim == 0:
        return None
    return int(len(arr))


def _npz_record_count(data: Mapping[str, Any]) -> int:
    for key in ("mtd_echo", "doppler", "pc_echo", "mtd"):
        if key not in data:
            continue
        arr = np.asarray(data[key])
        if arr.ndim >= 2:
            return int(arr.shape[0])
        if arr.dtype == object and arr.ndim == 1 and arr.size > 0:
            first = arr[0]
            if isinstance(first, (list, tuple, np.ndarray)):
                return int(arr.shape[0])
    for key in ("label", "labels", "y", "target", "type"):
        if key in data:
            n = _maybe_len(data[key])
            if n and n > 1:
                return n
    return 1


def _value_at(value: Any, index: int, sample_count: int) -> Any:
    arr = np.asarray(value, dtype=object if isinstance(value, np.ndarray) and value.dtype == object else None)
    if sample_count <= 1:
        return value
    if arr.ndim == 0:
        return arr.item()
    return arr[index]


def _scalar_at(
    data: Mapping[str, Any],
    candidates: Sequence[str],
    index: int,
    sample_count: int,
    default: float | None = 0.0,
) -> float | None:
    key = _pick_key(data, candidates, required=False)
    if key is None:
        return default
    value = _value_at(data[key], index, sample_count)
    arr = np.asarray(value)
    if arr.size == 0:
        return default
    return float(arr.reshape(-1)[0])


def _label_at(data: Mapping[str, Any], index: int, sample_count: int, require_label: bool) -> Any | None:
    key = _pick_key(data, ("label", "labels", "y", "target", "type"), required=require_label)
    if key is None:
        return None
    value = _value_at(data[key], index, sample_count)
    arr = np.asarray(value)
    if arr.size == 0:
        return None
    return arr.reshape(-1)[0].item() if isinstance(arr.reshape(-1)[0], np.generic) else arr.reshape(-1)[0]


def _waveform_values(data: Mapping[str, Any], index: int, sample_count: int) -> tuple[float | None, float | None, float | None]:
    key = _pick_key(data, ("waveform", "meta", "waveform_params"), required=False)
    if key is None:
        return None, None, None
    arr = np.asarray(_value_at(data[key], index, sample_count), dtype=np.float32).reshape(-1)
    if arr.size < 3:
        return None, None, None
    return float(arr[0]), float(arr[1]), float(arr[2])


def _record_from_npz_mapping(data: Mapping[str, Any], path: Path, index: int, sample_count: int, require_label: bool) -> RadarSample:
    mtd_key = _pick_key(
        data,
        ("mtd_echo", "doppler", "doppler_pulse", "pc_doppler", "pulse_doppler", "doppler_data"),
    )
    pc_key = _pick_key(
        data,
        ("pc_echo", "pc", "pulse_compression", "pulse_compressed", "mtd", "mtd_range", "range_mtd", "mtd_data"),
    )
    mtd_echo = _as_1d_float(_value_at(data[mtd_key], index, sample_count))
    pc_echo = _as_1d_float(_value_at(data[pc_key], index, sample_count))

    wf_pw_us, wf_prt_nbr, wf_prf_hz = _waveform_values(data, index, sample_count)
    pw_us = _scalar_at(data, ("pw_us", "pulse_width_us", "pulse_width", "pw"), index, sample_count, wf_pw_us)
    prt_nbr = _scalar_at(data, ("prt_nbr", "pulse_count", "pulse_num", "pulse_nbr", "n_pulses"), index, sample_count, wf_prt_nbr)
    prt_us = _scalar_at(data, ("prt_us", "prt", "pulse_repetition_period_us"), index, sample_count, None)
    prf_hz = _scalar_at(data, ("prf_hz", "prf", "pulse_repetition_frequency"), index, sample_count, wf_prf_hz)
    if prt_us is None and prf_hz is not None and prf_hz > 0:
        prt_us = 1.0e6 / float(prf_hz)
    if prt_nbr is None or prt_nbr <= 0:
        prt_nbr = float(mtd_echo.size)

    decoded = {
        "range": _scalar_at(data, ("range", "range_m", "distance"), index, sample_count, 0.0),
        "pw_us": pw_us or 0.0,
        "prt_us": prt_us or 0.0,
        "prt_nbr": prt_nbr or float(mtd_echo.size),
        "pluse_band": _scalar_at(data, ("pluse_band", "pulse_band", "bandwidth_mhz", "band_mhz"), index, sample_count, 0.0),
        "sample_freq": _scalar_at(data, ("sample_freq", "sample_frequency_mhz", "fs_mhz"), index, sample_count, 0.0),
        "mtd_row": _scalar_at(data, ("mtd_row",), index, sample_count, 1.0),
        "mtd_col": _scalar_at(data, ("mtd_col",), index, sample_count, float(mtd_echo.size)),
        "mtd_ch": _scalar_at(data, ("mtd_ch",), index, sample_count, 0.0),
        "mtd_echo": mtd_echo,
        "pc_row": _scalar_at(data, ("pc_row",), index, sample_count, 1.0),
        "pc_col": _scalar_at(data, ("pc_col",), index, sample_count, float(pc_echo.size)),
        "pc_ch": _scalar_at(data, ("pc_ch",), index, sample_count, 0.0),
        "pc_echo": pc_echo,
    }
    label = _label_at(data, index, sample_count, require_label)
    return _record_from_decoded(decoded, label, f"{path}#{index}" if sample_count > 1 else str(path))


def _records_from_npz_file(path: Path, require_label: bool) -> list[RadarSample]:
    with np.load(path, allow_pickle=True) as npz:
        data = {key: npz[key] for key in npz.files}
    sample_count = _npz_record_count(data)
    return [_record_from_npz_mapping(data, path, i, sample_count, require_label) for i in range(sample_count)]


def _db_contains_echo_table(path: Path, table_name: str = DEFAULT_TABLE_NAME) -> bool:
    try:
        conn = sqlite3.connect(str(path))
    except sqlite3.Error:
        return False
    try:
        return _table_exists(conn, table_name)
    except sqlite3.Error:
        return False
    finally:
        conn.close()


def load_records(path: str | Path, require_label: bool = True) -> list[RadarSample]:
    """Load samples from .npz files, SQLite DClsEcho DBs, or directories containing them."""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(p)

    if p.is_dir():
        files = sorted(
            f for f in p.rglob("*") if f.is_file() and (f.suffix.lower() == ".npz" or f.suffix.lower() in _DB_SUFFIXES)
        )
        files = [
            f for f in files
            if f.suffix.lower() == ".npz" or _db_contains_echo_table(f)
        ]
        if not files:
            raise ValueError(f"no .npz files or SQLite files with table '{DEFAULT_TABLE_NAME}' found under {p}")
        records: list[RadarSample] = []
        for file_path in files:
            records.extend(load_records(file_path, require_label=require_label))
        if not records:
            raise ValueError(f"no valid radar samples decoded under {p}")
        return records

    suffix = p.suffix.lower()
    if suffix == ".npz":
        return _records_from_npz_file(p, require_label=require_label)
    if suffix in _DB_SUFFIXES:
        records = _records_from_db_file(p, require_label=require_label)
        if not records:
            raise ValueError(f"no valid DClsEcho samples decoded from {p}")
        return records
    raise ValueError(f"unsupported data file: {p}")


class RadarSampleDataset(Dataset):
    def __init__(
        self,
        records: Sequence[RadarSample],
        label_to_id: Mapping[Any, int] | None = None,
        meta_features: str | Sequence[str] | Sequence[int] | None = None,
    ) -> None:
        self.records = list(records)
        if not self.records:
            raise ValueError("RadarSampleDataset is empty")
        if label_to_id is None:
            labels = [r.label for r in self.records if r.label is not None]
            label_to_id = build_label_mapping(labels)
        self.label_to_id = dict(label_to_id)
        self.meta_indices, self.meta_feature_names = resolve_meta_feature_indices(meta_features)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        rec = self.records[index]
        mtd_echo = _echo_abs_series(rec.mtd_echo, rec.mtd_row, rec.mtd_col, rec.mtd_ch)
        pc_echo = _echo_abs_series(rec.pc_echo, rec.pc_row, rec.pc_col, rec.pc_ch)
        pc_spectrum = _pc_spectrum_abs_series(rec.pc_echo, rec.pc_row, rec.pc_col, rec.pc_ch)
        label_id = -1 if rec.label is None else int(self.label_to_id[canonical_label(rec.label)])
        return {
            "mtd_echo": _standardize_profile(mtd_echo),
            "mtd_echo_raw": _as_1d_float(mtd_echo),
            "pc_echo": _standardize_profile(pc_echo),
            "pc_echo_raw": _as_1d_float(pc_echo),
            "pc_spectrum": _standardize_profile(pc_spectrum),
            "pc_spectrum_raw": _as_1d_float(pc_spectrum),
            "meta": _meta_features(rec, mtd_echo, pc_echo)[self.meta_indices],
            "label": label_id,
            "sample_id": rec.sample_id,
        }


def collate_radar_batch(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    max_mtd_len = max(2, max(int(x["mtd_echo"].shape[0]) for x in batch))
    max_pc_len = max(2, max(int(x["pc_echo"].shape[0]) for x in batch))
    max_pc_spectrum_len = max(2, max(int(x["pc_spectrum"].shape[0]) for x in batch))
    mtd_echo = torch.zeros(len(batch), max_mtd_len, dtype=torch.float32)
    mtd_mask = torch.zeros(len(batch), max_mtd_len, dtype=torch.bool)
    pc_echo = torch.zeros(len(batch), max_pc_len, dtype=torch.float32)
    pc_mask = torch.zeros(len(batch), max_pc_len, dtype=torch.bool)
    pc_spectrum = torch.zeros(len(batch), max_pc_spectrum_len, dtype=torch.float32)
    pc_spectrum_mask = torch.zeros(len(batch), max_pc_spectrum_len, dtype=torch.bool)
    for i, item in enumerate(batch):
        x = torch.as_tensor(item["mtd_echo"], dtype=torch.float32).flatten()
        n = min(max_mtd_len, int(x.numel()))
        if n > 0:
            mtd_echo[i, :n] = x[:n]
            mtd_mask[i, :n] = True
        y = torch.as_tensor(item["pc_echo"], dtype=torch.float32).flatten()
        m = min(max_pc_len, int(y.numel()))
        if m > 0:
            pc_echo[i, :m] = y[:m]
            pc_mask[i, :m] = True
        z = torch.as_tensor(item["pc_spectrum"], dtype=torch.float32).flatten()
        k = min(max_pc_spectrum_len, int(z.numel()))
        if k > 0:
            pc_spectrum[i, :k] = z[:k]
            pc_spectrum_mask[i, :k] = True

    return {
        "mtd_echo": mtd_echo,
        "mtd_mask": mtd_mask,
        "pc_echo": pc_echo,
        "pc_mask": pc_mask,
        "pc_spectrum": pc_spectrum,
        "pc_spectrum_mask": pc_spectrum_mask,
        "meta": torch.stack([torch.as_tensor(x["meta"], dtype=torch.float32) for x in batch], dim=0),
        "label": torch.as_tensor([int(x["label"]) for x in batch], dtype=torch.long),
        "sample_id": [str(x["sample_id"]) for x in batch],
    }
