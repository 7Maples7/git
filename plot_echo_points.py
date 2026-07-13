# -*- coding: utf-8 -*-
"""Plot per-point DClsEcho pulse-compression and Doppler/profile data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np

try:
    from .dataset import RadarSample, canonical_label, load_records
except ImportError:
    from dataset import RadarSample, canonical_label, load_records  # type: ignore


def _safe_name(text: str, max_len: int = 140) -> str:
    text = re.sub(r"[\\/:*?\"<>|\s]+", "_", str(text)).strip("_")
    if len(text) > max_len:
        text = text[:max_len].rstrip("_")
    return text or "sample"


def _split_sample_id(sample_id: str) -> str:
    if "#" not in sample_id:
        return sample_id
    return sample_id.rsplit("#", 1)[1]


def _reshape_echo(raw: Any, row: int, col: int, ch: int) -> tuple[np.ndarray, str]:
    """Return [samples, channels] and a short status string."""
    arr = np.asarray(raw, dtype=np.float32).reshape(-1)
    row = max(0, int(row))
    col = max(0, int(col))
    ch = max(1, int(ch))
    expected = row * col * ch
    if arr.size == 0:
        return np.zeros((0, ch), dtype=np.float32), "empty"

    if expected > 0 and arr.size == expected:
        return arr.reshape(row * col, ch), "ok"

    sample_count = int(arr.size) // ch
    if sample_count > 0:
        used = sample_count * ch
        return arr[:used].reshape(sample_count, ch), f"size_mismatch_used_{used}_of_{arr.size}"

    return arr.reshape(-1, 1), f"size_mismatch_raw_{arr.size}"


def _series_abs(data: np.ndarray) -> np.ndarray:
    if data.size == 0:
        return np.asarray([], dtype=np.float32)
    if data.shape[1] >= 2:
        return np.sqrt(np.square(data[:, 0]) + np.square(data[:, 1]))
    return np.abs(data[:, 0])


def _plot_series(ax, title: str, data: np.ndarray, expected_points: int, status: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("sample index")
    ax.set_ylabel("amplitude")
    ax.grid(True, alpha=0.25)
    if data.size == 0:
        if expected_points > 0:
            ax.set_xlim(0, max(1, expected_points - 1))
        ax.text(
            0.5,
            0.5,
            f"no payload data\nexpected points: {expected_points}\nstatus: {status}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    x = np.arange(data.shape[0])
    abs_y = _series_abs(data)
    ax.plot(x, abs_y, label="ABS", linewidth=1.6)
    ax.plot(x, data[:, 0], label="CH1", linewidth=0.8, alpha=0.75)
    if data.shape[1] >= 2:
        ax.plot(x, data[:, 1], label="CH2", linewidth=0.8, alpha=0.75)
    ax.legend(loc="upper right", fontsize=8)
    ax.text(
        0.01,
        0.98,
        f"points={data.shape[0]}, channels={data.shape[1]}, status={status}",
        ha="left",
        va="top",
        transform=ax.transAxes,
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )


def _waveform_text(rec: RadarSample) -> str:
    prf_hz = 1.0e6 / rec.prt_us if rec.prt_us > 0 else 0.0
    duty = rec.pw_us / rec.prt_us if rec.prt_us > 0 else 0.0
    return (
        f"label={canonical_label(rec.label)} | range={rec.range_m:.3f} m | "
        f"pw={rec.pw_us:.3f} us | prt={rec.prt_us:.3f} us | "
        f"prt_nbr={rec.prt_nbr:.0f} | prf={prf_hz:.3f} Hz | duty={duty:.6f} | "
        f"band={rec.pluse_band:.3f} MHz | fs={rec.sample_freq:.3f} MHz\n"
        f"mtd_dim=[{rec.mtd_row},{rec.mtd_col},{rec.mtd_ch}], mtd_len={len(rec.mtd_echo)} | "
        f"pc_dim=[{rec.pc_row},{rec.pc_col},{rec.pc_ch}], pc_len={len(rec.pc_echo)} | "
        f"{_split_sample_id(rec.sample_id)}"
    )


def plot_record(rec: RadarSample, out_path: Path, dpi: int = 130) -> dict[str, Any]:
    mtd_data, mtd_status = _reshape_echo(rec.mtd_echo, rec.mtd_row, rec.mtd_col, rec.mtd_ch)
    pc_data, pc_status = _reshape_echo(rec.pc_echo, rec.pc_row, rec.pc_col, rec.pc_ch)

    fig, axes = plt.subplots(2, 1, figsize=(12.5, 7.2), constrained_layout=True)
    fig.suptitle(_waveform_text(rec), fontsize=10)

    _plot_series(
        axes[0],
        f"Pulse compression profile from mtd_dim col={rec.mtd_col}",
        mtd_data,
        int(rec.mtd_col),
        mtd_status,
    )
    _plot_series(
        axes[1],
        f"pc_dim profile col={rec.pc_col}",
        pc_data,
        int(rec.pc_col),
        pc_status,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

    return {
        "sample_id": rec.sample_id,
        "label": str(canonical_label(rec.label)),
        "range_m": float(rec.range_m),
        "pw_us": float(rec.pw_us),
        "prt_us": float(rec.prt_us),
        "prt_nbr": float(rec.prt_nbr),
        "pluse_band": float(rec.pluse_band),
        "sample_freq": float(rec.sample_freq),
        "mtd_dim": [int(rec.mtd_row), int(rec.mtd_col), int(rec.mtd_ch)],
        "mtd_len": int(len(rec.mtd_echo)),
        "mtd_plot_points": int(mtd_data.shape[0]),
        "mtd_status": mtd_status,
        "pc_dim": [int(rec.pc_row), int(rec.pc_col), int(rec.pc_ch)],
        "pc_len": int(len(rec.pc_echo)),
        "pc_plot_points": int(pc_data.shape[0]),
        "pc_status": pc_status,
        "plot_path": str(out_path),
    }


def _select_records(records: list[RadarSample], start: int, limit: int) -> list[RadarSample]:
    start = max(0, int(start))
    if limit == 0:
        return records[start:]
    return records[start : start + max(0, int(limit))]


def _write_index(rows: list[dict[str, Any]], out_dir: Path) -> None:
    json_path = out_dir / "index.json"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = out_dir / "index.csv"
    fieldnames = [
        "sample_id",
        "label",
        "range_m",
        "pw_us",
        "prt_us",
        "prt_nbr",
        "pluse_band",
        "sample_freq",
        "mtd_dim",
        "mtd_len",
        "mtd_plot_points",
        "mtd_status",
        "pc_dim",
        "pc_len",
        "pc_plot_points",
        "pc_status",
        "plot_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["mtd_dim"] = json.dumps(csv_row["mtd_dim"], ensure_ascii=False)
            csv_row["pc_dim"] = json.dumps(csv_row["pc_dim"], ensure_ascii=False)
            writer.writerow(csv_row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot DClsEcho point echo data parsed by dataset.py.")
    parser.add_argument("data_path", help="dataset root, split directory, .db/.sqlite, or .npz path")
    parser.add_argument("--out-dir", default="radar_three_cls/runs/echo_plots")
    parser.add_argument("--start", type=int, default=0, help="start index after loading records")
    parser.add_argument("--limit", type=int, default=30, help="number of points to plot; 0 means all")
    parser.add_argument("--dpi", type=int, default=130)
    parser.add_argument("--no-label", action="store_true", help="parse data without requiring labels")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(args.data_path, require_label=not args.no_label)
    selected = _select_records(records, args.start, args.limit)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    total = len(selected)
    for i, rec in enumerate(selected, start=1):
        label = _safe_name(str(canonical_label(rec.label)))
        sample_key = _safe_name(_split_sample_id(rec.sample_id))
        filename = f"{args.start + i - 1:06d}_{label}_{sample_key}.png"
        out_path = out_dir / filename
        rows.append(plot_record(rec, out_path, dpi=args.dpi))
        if i == 1 or i == total or i % 50 == 0:
            print(f"[{i}/{total}] {out_path}", flush=True)

    _write_index(rows, out_dir)
    print(f"records_loaded={len(records)} plotted={len(rows)}")
    print(f"output_dir={out_dir}")
    print(f"index_json={out_dir / 'index.json'}")
    print(f"index_csv={out_dir / 'index.csv'}")


if __name__ == "__main__":
    main()
