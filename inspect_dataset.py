# -*- coding: utf-8 -*-
"""Inspect DClsEcho datasets before training.

This script does not train a model. It checks whether SQLite/.npz data can be
parsed by radar_three_cls.dataset and whether the resulting model inputs match
the expected shapes:

- mtd_echo: Doppler dimension at the target range cell, length ~= prt_nbr
- pc_echo: slow-time IQ data described by pc_dim; model input length is the
  point count after channel reduction
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .dataset import (
        DEFAULT_TABLE_NAME,
        META_DIM,
        RadarSample,
        RadarSampleDataset,
        build_label_mapping,
        canonical_label,
        collate_radar_batch,
        decode_dcls_echo_blob,
        load_records,
        _detect_columns,
        _quote_ident,
        _table_exists,
    )
except ImportError:
    from dataset import (  # type: ignore
        DEFAULT_TABLE_NAME,
        META_DIM,
        RadarSample,
        RadarSampleDataset,
        build_label_mapping,
        canonical_label,
        collate_radar_batch,
        decode_dcls_echo_blob,
        load_records,
        _detect_columns,
        _quote_ident,
        _table_exists,
    )


DB_SUFFIXES = {".db", ".sqlite", ".sqlite3"}


def _as_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _as_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_jsonable(v) for v in value]
    return value


def _top_counts(values: list[Any], limit: int = 12) -> dict[str, int]:
    return {str(k): int(v) for k, v in Counter(values).most_common(limit)}


def _numeric_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    values_sorted = sorted(float(v) for v in values)
    n = len(values_sorted)

    def pct(p: float) -> float:
        idx = min(n - 1, max(0, int(round((n - 1) * p))))
        return values_sorted[idx]

    return {
        "count": n,
        "min": values_sorted[0],
        "p50": pct(0.50),
        "p95": pct(0.95),
        "max": values_sorted[-1],
    }


def _channel_count(ch: int) -> int:
    return max(1, int(ch))


def _payload_expected_len(row: int, col: int, ch: int) -> int:
    row = max(0, int(row))
    col = max(0, int(col))
    if row <= 0 or col <= 0:
        return 0
    return row * col * _channel_count(ch)


def _payload_point_count(raw_len: int, row: int, col: int, ch: int) -> int:
    row = max(0, int(row))
    col = max(0, int(col))
    if row > 0 and col > 0:
        return row * col
    channels = _channel_count(ch)
    return int(raw_len) // channels if int(raw_len) % channels == 0 else int(raw_len)


def _find_data_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in DB_SUFFIXES or path.suffix.lower() == ".npz" else []
    return sorted(
        p
        for p in path.rglob("*")
        if p.is_file() and (p.suffix.lower() in DB_SUFFIXES or p.suffix.lower() == ".npz")
    )


def inspect_db_file(path: Path, table_name: str) -> dict[str, Any]:
    info: dict[str, Any] = {
        "path": str(path),
        "table": table_name,
        "exists": False,
        "columns": {},
        "row_count": 0,
        "decoded_count": 0,
        "failed_count": 0,
        "error_counts": {},
        "first_errors": [],
    }

    conn = sqlite3.connect(str(path))
    try:
        if not _table_exists(conn, table_name):
            return info
        info["exists"] = True
        columns = _detect_columns(conn, table_name)
        info["columns"] = {k: v for k, v in columns.items()}
        data_col = columns["data"]
        sql = f"SELECT rowid, {_quote_ident(data_col)} FROM {_quote_ident(table_name)}"
        errors: Counter[str] = Counter()
        first_errors: list[dict[str, Any]] = []
        for rowid, payload in conn.execute(sql):
            info["row_count"] += 1
            try:
                decode_dcls_echo_blob(payload)
                info["decoded_count"] += 1
            except Exception as exc:  # noqa: BLE001 - diagnostic script
                info["failed_count"] += 1
                msg = str(exc)
                errors[msg] += 1
                if len(first_errors) < 5:
                    first_errors.append({"rowid": int(rowid), "error": msg})
        info["error_counts"] = {str(k): int(v) for k, v in errors.most_common()}
        info["first_errors"] = first_errors
        return info
    finally:
        conn.close()


def inspect_raw_files(path: Path, table_name: str) -> list[dict[str, Any]]:
    infos: list[dict[str, Any]] = []
    for file_path in _find_data_files(path):
        if file_path.suffix.lower() in DB_SUFFIXES:
            infos.append(inspect_db_file(file_path, table_name))
        else:
            infos.append({"path": str(file_path), "kind": "npz"})
    return infos


def summarize_records(
    records: list[RadarSample],
    label_to_id: dict[Any, int],
    batch_size: int,
) -> dict[str, Any]:
    labels = [canonical_label(r.label) for r in records]
    mtd_lens = [int(len(r.mtd_echo)) for r in records]
    pc_lens = [int(len(r.pc_echo)) for r in records]
    mtd_points = [_payload_point_count(int(len(r.mtd_echo)), int(r.mtd_row), int(r.mtd_col), int(r.mtd_ch)) for r in records]
    pc_points = [_payload_point_count(int(len(r.pc_echo)), int(r.pc_row), int(r.pc_col), int(r.pc_ch)) for r in records]
    prt_expected = [int(round(float(r.prt_nbr))) if float(r.prt_nbr) > 0 else 0 for r in records]
    ranges = [float(r.range_m) for r in records]
    pw_us = [float(r.pw_us) for r in records]
    prt_us = [float(r.prt_us) for r in records]
    band = [float(r.pluse_band) for r in records]
    fs = [float(r.sample_freq) for r in records]
    mtd_dims = [(int(r.mtd_row), int(r.mtd_col), int(r.mtd_ch)) for r in records]
    pc_dims = [(int(r.pc_row), int(r.pc_col), int(r.pc_ch)) for r in records]

    mtd_prt_mismatch = [
        {
            "sample_id": r.sample_id,
            "prt_nbr": float(r.prt_nbr),
            "mtd_len": int(len(r.mtd_echo)),
            "mtd_points": _payload_point_count(int(len(r.mtd_echo)), int(r.mtd_row), int(r.mtd_col), int(r.mtd_ch)),
        }
        for r in records
        if int(round(float(r.prt_nbr))) > 0
        and _payload_point_count(int(len(r.mtd_echo)), int(r.mtd_row), int(r.mtd_col), int(r.mtd_ch))
        != int(round(float(r.prt_nbr)))
    ]
    mtd_dim_mismatch = [
        {
            "sample_id": r.sample_id,
            "mtd_dim": [int(r.mtd_row), int(r.mtd_col), int(r.mtd_ch)],
            "mtd_len": int(len(r.mtd_echo)),
            "expected_len": _payload_expected_len(int(r.mtd_row), int(r.mtd_col), int(r.mtd_ch)),
        }
        for r in records
        if _payload_expected_len(int(r.mtd_row), int(r.mtd_col), int(r.mtd_ch)) > 0
        and _payload_expected_len(int(r.mtd_row), int(r.mtd_col), int(r.mtd_ch)) != int(len(r.mtd_echo))
    ]
    pc_dim_mismatch = [
        {
            "sample_id": r.sample_id,
            "pc_dim": [int(r.pc_row), int(r.pc_col), int(r.pc_ch)],
            "pc_len": int(len(r.pc_echo)),
            "expected_len": _payload_expected_len(int(r.pc_row), int(r.pc_col), int(r.pc_ch)),
        }
        for r in records
        if _payload_expected_len(int(r.pc_row), int(r.pc_col), int(r.pc_ch)) > 0
        and _payload_expected_len(int(r.pc_row), int(r.pc_col), int(r.pc_ch)) != int(len(r.pc_echo))
    ]

    dataset = RadarSampleDataset(records, label_to_id=label_to_id)
    sample_count = min(max(1, int(batch_size)), len(dataset))
    batch = collate_radar_batch([dataset[i] for i in range(sample_count)])
    batch_shapes = {
        key: list(value.shape)
        for key, value in batch.items()
        if hasattr(value, "shape")
    }

    examples = []
    for r in records[: min(5, len(records))]:
        examples.append(
            {
                "sample_id": r.sample_id,
                "label": str(canonical_label(r.label)),
                "range_m": float(r.range_m),
                "pw_us": float(r.pw_us),
                "prt_us": float(r.prt_us),
                "prt_nbr": float(r.prt_nbr),
                "pluse_band": float(r.pluse_band),
                "sample_freq": float(r.sample_freq),
                "mtd_dim": [int(r.mtd_row), int(r.mtd_col), int(r.mtd_ch)],
                "mtd_len": int(len(r.mtd_echo)),
                "pc_dim": [int(r.pc_row), int(r.pc_col), int(r.pc_ch)],
                "pc_len": int(len(r.pc_echo)),
                "mtd_head": [float(x) for x in r.mtd_echo[:5]],
                "pc_head": [float(x) for x in r.pc_echo[:5]],
            }
        )

    semantic_format_ok = (
        len(mtd_prt_mismatch) == 0
        and len(mtd_dim_mismatch) == 0
        and len(pc_dim_mismatch) == 0
    )

    return {
        "record_count": len(records),
        "semantic_format_ok": semantic_format_ok,
        "semantic_expectation": {
            "mtd_echo": "point_count(mtd_echo) == round(prt_nbr)",
            "pc_echo": "pc_dim describes the stored slow-time IQ payload",
            "dims": "row * col * max(ch, 1) == stored vector length when row/col are positive",
        },
        "label_counts": _top_counts(labels),
        "label_to_id": {str(k): int(v) for k, v in label_to_id.items()},
        "mtd_len": _numeric_summary([float(x) for x in mtd_lens]),
        "pc_len": _numeric_summary([float(x) for x in pc_lens]),
        "mtd_points": _numeric_summary([float(x) for x in mtd_points]),
        "pc_points": _numeric_summary([float(x) for x in pc_points]),
        "prt_nbr": _numeric_summary([float(x) for x in prt_expected if x > 0]),
        "range_m": _numeric_summary(ranges),
        "pw_us": _numeric_summary(pw_us),
        "prt_us": _numeric_summary(prt_us),
        "pluse_band": _numeric_summary(band),
        "sample_freq": _numeric_summary(fs),
        "mtd_dims_top": _top_counts(mtd_dims),
        "pc_dims_top": _top_counts(pc_dims),
        "mtd_len_not_equal_prt_nbr_count": len(mtd_prt_mismatch),
        "mtd_len_not_equal_prt_nbr_examples": mtd_prt_mismatch[:5],
        "mtd_row_col_ch_not_len_count": len(mtd_dim_mismatch),
        "mtd_row_col_ch_not_len_examples": mtd_dim_mismatch[:5],
        "pc_row_col_ch_not_len_count": len(pc_dim_mismatch),
        "pc_row_col_ch_not_len_examples": pc_dim_mismatch[:5],
        "expected_model_input": {
            "mtd_echo": ["batch", "max_prt_nbr_in_batch"],
            "mtd_mask": ["batch", "max_prt_nbr_in_batch"],
            "pc_echo": ["batch", "max_pc_points_in_batch"],
            "pc_mask": ["batch", "max_pc_points_in_batch"],
            "pc_spectrum": ["batch", "max_pc_points_in_batch"],
            "pc_spectrum_mask": ["batch", "max_pc_points_in_batch"],
            "meta": ["batch", META_DIM],
        },
        "batch_shapes": batch_shapes,
        "examples": examples,
    }


def discover_splits(root: Path, train_path: str, test_path: str) -> dict[str, Path]:
    if train_path or test_path:
        splits = {}
        if train_path:
            splits["train"] = Path(train_path).expanduser().resolve()
        if test_path:
            splits["test"] = Path(test_path).expanduser().resolve()
        return splits

    train_dir = root / "train"
    test_dir = root / "test"
    if train_dir.exists() or test_dir.exists():
        splits = {}
        if train_dir.exists():
            splits["train"] = train_dir
        if test_dir.exists():
            splits["test"] = test_dir
        return splits
    return {"all": root}


def inspect_dataset(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.data_root).expanduser().resolve()
    splits = discover_splits(root, args.train_path, args.test_path)

    loaded: dict[str, list[RadarSample]] = {}
    split_errors: dict[str, str] = {}
    for split, path in splits.items():
        try:
            loaded[split] = load_records(path, require_label=not args.no_label)
        except Exception as exc:  # noqa: BLE001 - diagnostic script
            loaded[split] = []
            split_errors[split] = str(exc)

    all_records = [r for records in loaded.values() for r in records]
    label_to_id = build_label_mapping([r.label for r in all_records if r.label is not None])

    result: dict[str, Any] = {
        "data_root": str(root),
        "table_name": args.table_name,
        "splits": {},
        "split_errors": split_errors,
    }

    for split, path in splits.items():
        split_info: dict[str, Any] = {
            "path": str(path),
            "files": inspect_raw_files(path, args.table_name),
        }
        records = loaded.get(split, [])
        if records:
            split_info["records"] = summarize_records(records, label_to_id, args.batch_size)
        else:
            split_info["records"] = {"record_count": 0}
        result["splits"][split] = split_info
    return result


def print_report(report: dict[str, Any]) -> None:
    print(f"Data root: {report['data_root']}")
    print(f"Table: {report['table_name']}")
    if report.get("split_errors"):
        print("Split errors:")
        for split, error in report["split_errors"].items():
            print(f"  - {split}: {error}")

    for split, info in report["splits"].items():
        print("")
        print(f"=== {split} ===")
        print(f"Path: {info['path']}")
        for file_info in info["files"]:
            if file_info.get("kind") == "npz":
                print(f"File: {file_info['path']} (.npz)")
                continue
            print(f"DB: {file_info['path']}")
            print(f"  table_exists={file_info['exists']} columns={file_info.get('columns', {})}")
            print(
                "  rows={row_count} decoded={decoded_count} failed={failed_count}".format(
                    **file_info
                )
            )
            if file_info.get("error_counts"):
                print(f"  errors={file_info['error_counts']}")
                print(f"  first_errors={file_info['first_errors']}")

        records = info["records"]
        print(f"Records loaded by dataset.py: {records.get('record_count', 0)}")
        if records.get("record_count", 0) <= 0:
            continue
        print(f"Semantic format OK: {records['semantic_format_ok']}")
        print(f"Labels: {records['label_counts']}")
        print(f"Label to id: {records['label_to_id']}")
        print(f"mtd_echo len: {records['mtd_len']}")
        print(f"pc_echo len: {records['pc_len']}")
        print(f"mtd_echo points: {records['mtd_points']}")
        print(f"pc_echo points: {records['pc_points']}")
        print(f"prt_nbr: {records['prt_nbr']}")
        print(f"range_m: {records['range_m']}")
        print(f"pw_us: {records['pw_us']}")
        print(f"prt_us: {records['prt_us']}")
        print(f"pluse_band: {records['pluse_band']}")
        print(f"sample_freq: {records['sample_freq']}")
        print(f"Top mtd dims: {records['mtd_dims_top']}")
        print(f"Top pc dims: {records['pc_dims_top']}")
        print(
            "Checks: "
            f"mtd_len!=prt_nbr {records['mtd_len_not_equal_prt_nbr_count']}, "
            f"mtd_row*col*ch!=len {records['mtd_row_col_ch_not_len_count']}, "
            f"pc_row*col*ch!=len {records['pc_row_col_ch_not_len_count']}"
        )
        print(f"Batch shapes: {records['batch_shapes']}")
        print("Examples:")
        for example in records["examples"]:
            print(json.dumps(example, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect DClsEcho dataset parsing before training.")
    parser.add_argument("data_root", default="", help="dataset root, split dir, .db/.sqlite, or .npz path")
    parser.add_argument("--train-path", default="", help="optional explicit train path")
    parser.add_argument("--test-path", default="", help="optional explicit test path")
    parser.add_argument("--table-name", default=DEFAULT_TABLE_NAME)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--no-label", action="store_true", help="parse data without requiring labels")
    parser.add_argument("--json-output", default="", help="optional path to save the full report as JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = inspect_dataset(args)
    print_report(report)
    if args.json_output:
        out = Path(args.json_output).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(_as_jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8")
        print("")
        print(f"JSON report saved: {out}")


if __name__ == "__main__":
    main()
