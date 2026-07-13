# -*- coding: utf-8 -*-
"""Convenience inference entrypoint for the model in runs/gui_train.

Examples:
    python -m radar_three_cls.use_gui_train_model --input path/to/echo.db --device auto
    python -m radar_three_cls.use_gui_train_model --input path/to/data_dir --output-json infer.json --output-csv infer.csv
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import threading
import webbrowser
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

try:
    from .dataset import RadarSample, RadarSampleDataset, collate_radar_batch, load_records
    from .infer import load_checkpoint
    from .train import move_batch, resolve_device
except ImportError:
    from dataset import RadarSample, RadarSampleDataset, collate_radar_batch, load_records  # type: ignore
    from infer import load_checkpoint  # type: ignore
    from train import move_batch, resolve_device  # type: ignore


DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "runs" / "gui_train" / "best.pth"


def _top_k(probabilities: dict[str, float], k: int) -> list[dict[str, Any]]:
    items = sorted(probabilities.items(), key=lambda item: float(item[1]), reverse=True)
    return [{"label": label, "probability": float(prob)} for label, prob in items[: max(1, int(k))]]


def _augment_results(results: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    augmented: list[dict[str, Any]] = []
    for item in results:
        row = dict(item)
        row["top_k"] = _top_k(dict(row.get("probabilities", {})), top_k)
        augmented.append(row)
    return augmented


def _sample_rowid(sample_id: str) -> int | None:
    tail = str(sample_id).rsplit("#", 1)[-1]
    if tail.startswith("rowid="):
        tail = tail.split(",", 1)[0].split("=", 1)[1]
    try:
        return int(tail)
    except ValueError:
        return None


def _select_records(
    records: list[RadarSample],
    rowids: set[int] | None,
    offset: int,
    limit: int,
) -> list[RadarSample]:
    selected = records
    if rowids:
        selected = [rec for rec in selected if _sample_rowid(rec.sample_id) in rowids]
    if offset > 0:
        selected = selected[offset:]
    if limit > 0:
        selected = selected[:limit]
    return selected


@torch.no_grad()
def run_inference(
    checkpoint: str | Path,
    input_path: str | Path,
    device_name: str,
    batch_size: int,
    rowids: set[int] | None = None,
    offset: int = 0,
    limit: int = 0,
) -> list[dict[str, Any]]:
    device = resolve_device(device_name)
    model, id_to_label, meta_feature_names = load_checkpoint(checkpoint, device)
    records = load_records(input_path, require_label=False)
    records = _select_records(records, rowids=rowids, offset=offset, limit=limit)
    for rec in records:
        rec.label = None
    if not records:
        return []

    dataset = RadarSampleDataset(records, label_to_id={}, meta_features=meta_feature_names)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_radar_batch)
    outputs: list[dict[str, Any]] = []
    for batch in loader:
        sample_ids = batch["sample_id"]
        batch = move_batch(batch, device)
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
        scores, preds = torch.max(probs, dim=1)
        for i, sample_id in enumerate(sample_ids):
            prob_map = {
                id_to_label[j]: float(probs[i, j].detach().cpu().item())
                for j in range(len(id_to_label))
            }
            pred_id = int(preds[i].detach().cpu().item())
            outputs.append(
                {
                    "sample_id": sample_id,
                    "rowid": _sample_rowid(str(sample_id)),
                    "pred_id": pred_id,
                    "pred_label": id_to_label[pred_id],
                    "score": float(scores[i].detach().cpu().item()),
                    "probabilities": prob_map,
                }
            )
    return outputs


def _write_json(path: str | Path, results: list[dict[str, Any]]) -> None:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: str | Path, results: list[dict[str, Any]]) -> None:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels: list[str] = []
    for item in results:
        for label in dict(item.get("probabilities", {})):
            if label not in labels:
                labels.append(label)

    fieldnames = ["sample_id", "pred_id", "pred_label", "score"] + [f"prob_{label}" for label in labels]
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            probabilities = dict(item.get("probabilities", {}))
            row = {
                "sample_id": item.get("sample_id", ""),
                "pred_id": item.get("pred_id", ""),
                "pred_label": item.get("pred_label", ""),
                "score": item.get("score", 0.0),
            }
            for label in labels:
                row[f"prob_{label}"] = probabilities.get(label, 0.0)
            writer.writerow(row)


def _probability_bars(probabilities: dict[str, float]) -> str:
    rows: list[str] = []
    for label, prob in sorted(probabilities.items(), key=lambda item: float(item[1]), reverse=True):
        pct = max(0.0, min(100.0, float(prob) * 100.0))
        rows.append(
            "<div class='prob-row'>"
            f"<span class='prob-label'>{html.escape(label)}</span>"
            "<span class='prob-bar-wrap'>"
            f"<span class='prob-bar' style='width:{pct:.2f}%'></span>"
            "</span>"
            f"<span class='prob-value'>{pct:.2f}%</span>"
            "</div>"
        )
    return "".join(rows)


def _write_html(path: str | Path, results: list[dict[str, Any]], checkpoint: Path, input_path: Path) -> None:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter(str(item.get("pred_label", "")) for item in results)
    max_count = max(counts.values(), default=1)

    count_rows = []
    for label, count in counts.most_common():
        width = 100.0 * count / max_count
        count_rows.append(
            "<div class='count-row'>"
            f"<span class='count-label'>{html.escape(label)}</span>"
            "<span class='count-bar-wrap'>"
            f"<span class='count-bar' style='width:{width:.2f}%'></span>"
            "</span>"
            f"<span class='count-value'>{count}</span>"
            "</div>"
        )

    table_rows = []
    for index, item in enumerate(results, start=1):
        sample_id = str(item.get("sample_id", ""))
        rowid = "" if item.get("rowid") is None else str(item.get("rowid"))
        pred_label = str(item.get("pred_label", ""))
        score = float(item.get("score", 0.0))
        probabilities = dict(item.get("probabilities", {}))
        table_rows.append(
            "<tr>"
            f"<td>{index}</td>"
            f"<td>{html.escape(rowid)}</td>"
            f"<td class='sample'>{html.escape(sample_id)}</td>"
            f"<td>{html.escape(pred_label)}</td>"
            f"<td>{score:.6f}</td>"
            f"<td>{_probability_bars(probabilities)}</td>"
            "</tr>"
        )

    text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Radar Inference Report</title>
  <style>
    body {{ margin: 0; font-family: Arial, "Microsoft YaHei", sans-serif; color: #1f2933; background: #f5f7fa; }}
    header {{ padding: 22px 28px; background: #1f6f8b; color: white; }}
    h1 {{ margin: 0 0 8px; font-size: 24px; }}
    .sub {{ opacity: .92; font-size: 13px; line-height: 1.6; }}
    main {{ padding: 22px 28px 40px; }}
    section {{ background: white; border: 1px solid #d8e0e7; border-radius: 6px; padding: 18px; margin-bottom: 18px; }}
    h2 {{ margin: 0 0 14px; font-size: 18px; }}
    .count-row, .prob-row {{ display: grid; grid-template-columns: minmax(220px, 360px) 1fr 70px; gap: 10px; align-items: center; margin: 7px 0; }}
    .count-bar-wrap, .prob-bar-wrap {{ height: 12px; border-radius: 4px; background: #eef2f5; overflow: hidden; }}
    .count-bar {{ display: block; height: 100%; background: #1f6f8b; }}
    .prob-bar {{ display: block; height: 100%; background: #53a66f; }}
    .count-value, .prob-value {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .toolbar {{ display: flex; gap: 10px; margin-bottom: 12px; }}
    input {{ width: 360px; max-width: 100%; padding: 8px 10px; border: 1px solid #cbd5df; border-radius: 4px; }}
    table {{ border-collapse: collapse; width: 100%; background: white; }}
    th, td {{ border-bottom: 1px solid #e5ebf0; padding: 8px 9px; vertical-align: top; font-size: 13px; }}
    th {{ position: sticky; top: 0; background: #eef3f7; text-align: left; z-index: 1; }}
    .sample {{ max-width: 520px; word-break: break-all; color: #486273; }}
    .prob-label {{ overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  </style>
</head>
<body>
  <header>
    <h1>雷达回波识别推理报告</h1>
    <div class="sub">
      模型: {html.escape(str(checkpoint))}<br>
      输入: {html.escape(str(input_path))}<br>
      样本数: {len(results)}
    </div>
  </header>
  <main>
    <section>
      <h2>预测类别统计</h2>
      {''.join(count_rows) if count_rows else '<p>无结果</p>'}
    </section>
    <section>
      <h2>逐点迹结果</h2>
      <div class="toolbar">
        <input id="q" placeholder="搜索 rowid / sample_id / 预测类别" oninput="filterRows()">
      </div>
      <table id="resultTable">
        <thead>
          <tr><th>#</th><th>rowid</th><th>sample_id</th><th>预测类别</th><th>置信度</th><th>类别概率</th></tr>
        </thead>
        <tbody>
          {''.join(table_rows)}
        </tbody>
      </table>
    </section>
  </main>
  <script>
    function filterRows() {{
      const q = document.getElementById('q').value.toLowerCase();
      for (const tr of document.querySelectorAll('#resultTable tbody tr')) {{
        tr.style.display = tr.innerText.toLowerCase().includes(q) ? '' : 'none';
      }}
    }}
  </script>
</body>
</html>
"""
    output_path.write_text(text, encoding="utf-8")


def _print_summary(results: list[dict[str, Any]], print_limit: int) -> None:
    print(f"Total samples: {len(results)}")
    counts = Counter(str(item.get("pred_label", "")) for item in results)
    print("Prediction counts:")
    for label, count in counts.most_common():
        print(f"  {label}: {count}")

    limit = len(results) if print_limit <= 0 else min(len(results), int(print_limit))
    print(f"\nFirst {limit} result(s):")
    for idx, item in enumerate(results[:limit], start=1):
        print(
            f"[{idx}] sample_id={item.get('sample_id')} | "
            f"pred={item.get('pred_label')} | score={float(item.get('score', 0.0)):.6f}"
        )
        for top in item.get("top_k", []):
            print(f"    {top['label']}: {float(top['probability']):.6f}")


def _probability_text(probabilities: dict[str, float]) -> str:
    items = sorted(probabilities.items(), key=lambda item: float(item[1]), reverse=True)
    return " | ".join(f"{label}: {float(prob):.6f}" for label, prob in items)


def _parse_rowids(text: str) -> set[int] | None:
    values = []
    for part in text.replace("，", ",").replace(" ", ",").split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    return set(values) if values else None


def launch_gui() -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    class InferenceApp(tk.Tk):
        def __init__(self) -> None:
            super().__init__()
            self.title("雷达回波识别模型推理")
            self.geometry("1280x760")
            self.minsize(980, 620)
            self.results: list[dict[str, Any]] = []

            self.checkpoint_var = tk.StringVar(value=str(DEFAULT_CHECKPOINT))
            self.input_var = tk.StringVar(value="")
            self.device_var = tk.StringVar(value="auto")
            self.batch_size_var = tk.StringVar(value="32")
            self.rowid_var = tk.StringVar(value="")
            self.offset_var = tk.StringVar(value="0")
            self.limit_var = tk.StringVar(value="0")
            self.status_var = tk.StringVar(value="请选择输入数据后开始识别。")
            self.summary_var = tk.StringVar(value="未运行")

            self._build_ui()

        def _build_ui(self) -> None:
            root = ttk.Frame(self, padding=12)
            root.pack(fill=tk.BOTH, expand=True)
            root.columnconfigure(0, weight=1)
            root.rowconfigure(3, weight=1)

            path_frame = ttk.LabelFrame(root, text="输入")
            path_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
            path_frame.columnconfigure(1, weight=1)
            ttk.Label(path_frame, text="模型:").grid(row=0, column=0, padx=8, pady=7, sticky="w")
            ttk.Entry(path_frame, textvariable=self.checkpoint_var).grid(row=0, column=1, padx=6, pady=7, sticky="ew")
            ttk.Button(path_frame, text="选择模型", command=self.choose_checkpoint).grid(row=0, column=2, padx=6, pady=7)
            ttk.Label(path_frame, text="数据:").grid(row=1, column=0, padx=8, pady=7, sticky="w")
            ttk.Entry(path_frame, textvariable=self.input_var).grid(row=1, column=1, padx=6, pady=7, sticky="ew")
            ttk.Button(path_frame, text="选择文件", command=self.choose_input_file).grid(row=1, column=2, padx=6, pady=7)
            ttk.Button(path_frame, text="选择目录", command=self.choose_input_dir).grid(row=1, column=3, padx=6, pady=7)

            opt_frame = ttk.LabelFrame(root, text="选项")
            opt_frame.grid(row=1, column=0, sticky="ew", pady=(0, 8))
            for col in range(12):
                opt_frame.columnconfigure(col, weight=0)
            ttk.Label(opt_frame, text="设备").grid(row=0, column=0, padx=8, pady=7)
            ttk.Combobox(opt_frame, textvariable=self.device_var, values=("auto", "cpu", "cuda"), width=8, state="readonly").grid(row=0, column=1, padx=4, pady=7)
            ttk.Label(opt_frame, text="batch").grid(row=0, column=2, padx=8, pady=7)
            ttk.Entry(opt_frame, textvariable=self.batch_size_var, width=8).grid(row=0, column=3, padx=4, pady=7)
            ttk.Label(opt_frame, text="rowid").grid(row=0, column=4, padx=8, pady=7)
            ttk.Entry(opt_frame, textvariable=self.rowid_var, width=18).grid(row=0, column=5, padx=4, pady=7)
            ttk.Label(opt_frame, text="offset").grid(row=0, column=6, padx=8, pady=7)
            ttk.Entry(opt_frame, textvariable=self.offset_var, width=8).grid(row=0, column=7, padx=4, pady=7)
            ttk.Label(opt_frame, text="limit").grid(row=0, column=8, padx=8, pady=7)
            ttk.Entry(opt_frame, textvariable=self.limit_var, width=8).grid(row=0, column=9, padx=4, pady=7)
            ttk.Button(opt_frame, text="开始识别", command=self.start_inference).grid(row=0, column=10, padx=12, pady=7)
            ttk.Button(opt_frame, text="清空结果", command=self.clear_results).grid(row=0, column=11, padx=4, pady=7)

            info_frame = ttk.Frame(root)
            info_frame.grid(row=2, column=0, sticky="ew", pady=(0, 8))
            info_frame.columnconfigure(0, weight=1)
            ttk.Label(info_frame, textvariable=self.summary_var).grid(row=0, column=0, sticky="w")
            ttk.Label(info_frame, textvariable=self.status_var).grid(row=1, column=0, sticky="w", pady=(4, 0))
            ttk.Button(info_frame, text="导出 JSON", command=self.export_json).grid(row=0, column=1, padx=4)
            ttk.Button(info_frame, text="导出 CSV", command=self.export_csv).grid(row=0, column=2, padx=4)

            result_pane = ttk.PanedWindow(root, orient=tk.VERTICAL)
            result_pane.grid(row=3, column=0, sticky="nsew")

            table_frame = ttk.Frame(result_pane)
            table_frame.rowconfigure(0, weight=1)
            table_frame.columnconfigure(0, weight=1)
            columns = ("idx", "rowid", "pred_label", "score", "sample_id", "probabilities")
            self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=18)
            self.tree.heading("idx", text="#")
            self.tree.heading("rowid", text="rowid")
            self.tree.heading("pred_label", text="预测类别")
            self.tree.heading("score", text="置信度")
            self.tree.heading("sample_id", text="sample_id")
            self.tree.heading("probabilities", text="类别概率")
            self.tree.column("idx", width=52, anchor=tk.CENTER, stretch=False)
            self.tree.column("rowid", width=82, anchor=tk.CENTER, stretch=False)
            self.tree.column("pred_label", width=260, stretch=False)
            self.tree.column("score", width=90, anchor=tk.CENTER, stretch=False)
            self.tree.column("sample_id", width=360, stretch=True)
            self.tree.column("probabilities", width=460, stretch=True)
            self.tree.grid(row=0, column=0, sticky="nsew")
            ybar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
            ybar.grid(row=0, column=1, sticky="ns")
            xbar = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
            xbar.grid(row=1, column=0, sticky="ew")
            self.tree.configure(yscrollcommand=ybar.set, xscrollcommand=xbar.set)
            self.tree.bind("<<TreeviewSelect>>", self.show_selected_detail)
            result_pane.add(table_frame, weight=4)

            detail_frame = ttk.LabelFrame(result_pane, text="选中点迹详情")
            detail_frame.rowconfigure(0, weight=1)
            detail_frame.columnconfigure(0, weight=1)
            self.detail_text = tk.Text(detail_frame, height=8, wrap=tk.WORD)
            self.detail_text.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
            detail_scroll = ttk.Scrollbar(detail_frame, orient=tk.VERTICAL, command=self.detail_text.yview)
            detail_scroll.grid(row=0, column=1, sticky="ns")
            self.detail_text.configure(yscrollcommand=detail_scroll.set)
            result_pane.add(detail_frame, weight=1)

        def choose_checkpoint(self) -> None:
            path = filedialog.askopenfilename(
                title="选择模型权重",
                filetypes=(("PyTorch checkpoint", "*.pth *.pt"), ("All files", "*.*")),
            )
            if path:
                self.checkpoint_var.set(path)

        def choose_input_file(self) -> None:
            path = filedialog.askopenfilename(
                title="选择输入数据",
                filetypes=(
                    ("Radar data", "*.db *.sqlite *.sqlite3 *.npz"),
                    ("All files", "*.*"),
                ),
            )
            if path:
                self.input_var.set(path)

        def choose_input_dir(self) -> None:
            path = filedialog.askdirectory(title="选择包含 DB/NPZ 的目录")
            if path:
                self.input_var.set(path)

        def _read_options(self) -> tuple[Path, Path, str, int, set[int] | None, int, int]:
            checkpoint = Path(self.checkpoint_var.get()).expanduser().resolve()
            input_path = Path(self.input_var.get()).expanduser().resolve()
            if not checkpoint.exists():
                raise FileNotFoundError(f"模型文件不存在: {checkpoint}")
            if not input_path.exists():
                raise FileNotFoundError(f"输入数据不存在: {input_path}")
            batch_size = max(1, int(self.batch_size_var.get()))
            rowids = _parse_rowids(self.rowid_var.get())
            offset = max(0, int(self.offset_var.get() or 0))
            limit = max(0, int(self.limit_var.get() or 0))
            return checkpoint, input_path, self.device_var.get(), batch_size, rowids, offset, limit

        def start_inference(self) -> None:
            try:
                options = self._read_options()
            except Exception as exc:
                messagebox.showerror("参数错误", str(exc))
                return
            self.status_var.set("正在识别，请稍候...")
            self.summary_var.set("运行中")
            self.clear_results()

            def worker() -> None:
                try:
                    checkpoint, input_path, device, batch_size, rowids, offset, limit = options
                    results = run_inference(
                        checkpoint,
                        input_path,
                        device_name=device,
                        batch_size=batch_size,
                        rowids=rowids,
                        offset=offset,
                        limit=limit,
                    )
                    results = _augment_results(results, 3)
                except Exception as exc:
                    self.after(0, lambda: self._finish_error(exc))
                    return
                self.after(0, lambda: self._finish_success(results))

            threading.Thread(target=worker, daemon=True).start()

        def _finish_success(self, results: list[dict[str, Any]]) -> None:
            self.results = results
            self.populate_results()
            counts = Counter(str(item.get("pred_label", "")) for item in results)
            count_text = "；".join(f"{label}: {count}" for label, count in counts.most_common())
            self.summary_var.set(f"识别完成，共 {len(results)} 条。{count_text}")
            self.status_var.set("完成")

        def _finish_error(self, exc: Exception) -> None:
            self.status_var.set("识别失败")
            self.summary_var.set("失败")
            messagebox.showerror("识别失败", str(exc))

        def populate_results(self) -> None:
            for iid in self.tree.get_children():
                self.tree.delete(iid)
            for index, item in enumerate(self.results, start=1):
                probs = dict(item.get("probabilities", {}))
                self.tree.insert(
                    "",
                    tk.END,
                    iid=str(index - 1),
                    values=(
                        index,
                        "" if item.get("rowid") is None else item.get("rowid"),
                        item.get("pred_label", ""),
                        f"{float(item.get('score', 0.0)):.6f}",
                        item.get("sample_id", ""),
                        _probability_text(probs),
                    ),
                )

        def show_selected_detail(self, _event: tk.Event | None = None) -> None:
            selection = self.tree.selection()
            if not selection:
                return
            index = int(selection[0])
            if index < 0 or index >= len(self.results):
                return
            item = self.results[index]
            lines = [
                f"sample_id: {item.get('sample_id', '')}",
                f"rowid: {item.get('rowid', '')}",
                f"pred_id: {item.get('pred_id', '')}",
                f"pred_label: {item.get('pred_label', '')}",
                f"score: {float(item.get('score', 0.0)):.6f}",
                "",
                "probabilities:",
            ]
            for label, prob in sorted(dict(item.get("probabilities", {})).items(), key=lambda x: float(x[1]), reverse=True):
                lines.append(f"  {label}: {float(prob):.6f}")
            self.detail_text.delete("1.0", tk.END)
            self.detail_text.insert("1.0", "\n".join(lines))

        def clear_results(self) -> None:
            self.results = []
            for iid in self.tree.get_children():
                self.tree.delete(iid)
            self.detail_text.delete("1.0", tk.END)

        def export_json(self) -> None:
            if not self.results:
                messagebox.showwarning("无结果", "请先完成识别。")
                return
            path = filedialog.asksaveasfilename(
                title="保存 JSON",
                defaultextension=".json",
                filetypes=(("JSON", "*.json"), ("All files", "*.*")),
            )
            if path:
                _write_json(path, self.results)
                self.status_var.set(f"JSON 已保存: {path}")

        def export_csv(self) -> None:
            if not self.results:
                messagebox.showwarning("无结果", "请先完成识别。")
                return
            path = filedialog.asksaveasfilename(
                title="保存 CSV",
                defaultextension=".csv",
                filetypes=(("CSV", "*.csv"), ("All files", "*.*")),
            )
            if path:
                _write_csv(path, self.results)
                self.status_var.set(f"CSV 已保存: {path}")

    app = InferenceApp()
    app.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use runs/gui_train/best.pth to infer radar target classes.")
    parser.add_argument("--gui", action="store_true", help="open the desktop GUI")
    parser.add_argument("--input", default="", help=".db/.sqlite/.npz file, or a directory containing them")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="model checkpoint path")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--rowid", type=int, action="append", default=[], help="only infer the specified DB rowid; can repeat")
    parser.add_argument("--offset", type=int, default=0, help="skip the first N loaded records before inference")
    parser.add_argument("--limit", type=int, default=0, help="infer at most N loaded records; <=0 means all")
    parser.add_argument("--top-k", type=int, default=3, help="number of probabilities to print per sample")
    parser.add_argument("--print-limit", type=int, default=30, help="max samples to print; <=0 prints all")
    parser.add_argument("--output-json", default="", help="optional JSON result path")
    parser.add_argument("--output-csv", default="", help="optional CSV result path")
    parser.add_argument("--output-html", default="", help="optional visual HTML report path")
    parser.add_argument("--open-html", action="store_true", help="open the HTML report after writing it")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.gui or not args.input:
        launch_gui()
        return

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    input_path = Path(args.input).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if not input_path.exists():
        raise FileNotFoundError(f"input not found: {input_path}")

    print(f"Checkpoint: {checkpoint}")
    print(f"Input: {input_path}")
    rowids = set(int(x) for x in args.rowid) if args.rowid else None
    results = run_inference(
        checkpoint,
        input_path,
        device_name=args.device,
        batch_size=args.batch_size,
        rowids=rowids,
        offset=max(0, int(args.offset)),
        limit=max(0, int(args.limit)),
    )
    results = _augment_results(results, args.top_k)
    _print_summary(results, args.print_limit)

    if args.output_json:
        _write_json(args.output_json, results)
        print(f"\nJSON saved: {Path(args.output_json).expanduser().resolve()}")
    if args.output_csv:
        _write_csv(args.output_csv, results)
        print(f"CSV saved: {Path(args.output_csv).expanduser().resolve()}")
    if args.output_html:
        html_path = Path(args.output_html).expanduser().resolve()
        _write_html(html_path, results, checkpoint, input_path)
        print(f"HTML saved: {html_path}")
        if args.open_html:
            webbrowser.open(html_path.as_uri())


if __name__ == "__main__":
    main()
