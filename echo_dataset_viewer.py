# -*- coding: utf-8 -*-
"""Tkinter workbench for viewing DClsEcho datasets and starting training."""

from __future__ import annotations

import argparse
import contextlib
import threading
import tkinter as tk
from collections import Counter
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

import matplotlib

matplotlib.use("TkAgg")
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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np

try:
    from .dataset import (
        DEFAULT_META_FEATURE_NAMES,
        META_FEATURE_NAMES,
        RadarSample,
        canonical_label,
        load_records,
    )
    from .model import RadarThreeBranchNet
    from .target_types import format_target_type
    from .train import TrainConfig, train as train_model
except ImportError:
    from dataset import (  # type: ignore
        DEFAULT_META_FEATURE_NAMES,
        META_FEATURE_NAMES,
        RadarSample,
        canonical_label,
        load_records,
    )
    from model import RadarThreeBranchNet  # type: ignore
    from target_types import format_target_type  # type: ignore
    from train import TrainConfig, train as train_model  # type: ignore


META_DESCRIPTIONS = {
    "log_range_m": "距离 log1p(range_m)",
    "log_pw_us": "脉宽 log1p(pw_us)",
    "log_prt_us": "PRT log1p(prt_us)",
    "log_prt_nbr": "脉冲数 log1p(prt_nbr)",
    "duty": "占空比 pw/prt",
    "log_time_bandwidth": "时宽带宽积 log1p(pw*band)",
    "log_mtd_energy": "MTD 能量 log1p(mean(x^2))",
    "pc_spectrum_db_mean": "PC 频谱 dB 均值 mean(20log10(|FFT(I+jQ)|))",
    "log_pc_len": "PC payload 长度 log1p(len)",
}

APP_BG = "#eef3f7"
PANEL_BG = "#ffffff"
PANEL_ALT = "#f7fafc"
TEXT_BG = "#fbfdff"
TEXT_FG = "#243447"
MUTED_FG = "#667085"
ACCENT = "#1f6f8b"
ACCENT_DARK = "#155b73"
ACCENT_SOFT = "#d9edf5"
BORDER = "#cbd8e3"
SUCCESS = "#237a57"
TREE_PREVIEW_LIMIT = 5000


def _split_sample_id(sample_id: str) -> str:
    if "#" not in sample_id:
        return sample_id
    return sample_id.rsplit("#", 1)[1]


def _display_label(label: Any) -> str:
    return format_target_type(canonical_label(label), show_code=True)


def _format_distance_edge(value: float) -> str:
    if abs(value - round(value)) < 1.0e-6:
        return str(int(round(value)))
    return f"{value:.1f}"


def _format_range_bin(start: float, width: float) -> str:
    end = start + width
    return f"{_format_distance_edge(start)}-{_format_distance_edge(end)}"


def _range_distribution(
    records: list[RadarSample],
    bin_width_m: float,
) -> tuple[list[str], list[tuple[float, str, int, dict[str, int]]], int, tuple[float, float] | None]:
    valid: list[tuple[float, str]] = []
    for rec in records:
        distance = float(rec.range_m)
        if not np.isfinite(distance):
            continue
        valid.append((distance, _display_label(rec.label)))

    if not valid:
        return [], [], 0, None

    label_counts = Counter(label for _, label in valid)
    labels = [label for label, _count in label_counts.most_common()]
    bins: dict[int, Counter[str]] = {}
    for distance, label in valid:
        bin_index = int(np.floor(distance / bin_width_m))
        bins.setdefault(bin_index, Counter())[label] += 1

    rows: list[tuple[float, str, int, dict[str, int]]] = []
    for bin_index in sorted(bins):
        start = float(bin_index) * bin_width_m
        counts = dict(bins[bin_index])
        total = int(sum(counts.values()))
        rows.append((start, _format_range_bin(start, bin_width_m), total, counts))

    distances = [distance for distance, _label in valid]
    return labels, rows, len(valid), (float(min(distances)), float(max(distances)))


def _reshape_echo(raw: Any, row: int, col: int, ch: int) -> tuple[np.ndarray, str]:
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


def _fft_spectrum(data: np.ndarray, prt_us: float) -> tuple[np.ndarray, np.ndarray, str, str]:
    if data.size == 0:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32), "empty", "frequency (Hz)"

    if data.shape[1] < 2:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32), "need two channels", "frequency (Hz)"

    signal = data[:, 0].astype(np.float32) + 1j * data[:, 1].astype(np.float32)
    spectrum = np.fft.fftshift(np.fft.fft(signal))
    if prt_us > 0:
        sample_interval_s = float(prt_us) * 1.0e-6
        bins = np.fft.fftshift(np.fft.fftfreq(signal.size, d=sample_interval_s))
        mode = f"complex IQ FFT + fftshift, PRF={1.0 / sample_interval_s:.3f} Hz"
        x_label = "frequency (Hz)"
    else:
        bins = np.fft.fftshift(np.fft.fftfreq(signal.size) * signal.size)
        mode = "complex IQ FFT + fftshift, missing prt_us"
        x_label = "frequency bin"
    magnitude = (20.0 * np.log10(np.maximum(np.abs(spectrum), 1.0e-12))).astype(np.float32)
    return bins.astype(np.float32), magnitude, mode, x_label


def _format_dim(row: int, col: int, ch: int) -> str:
    return f"[{int(row)}, {int(col)}, {int(ch)}]"


def _record_info(rec: RadarSample, index: int, total: int) -> str:
    prf_hz = 1.0e6 / rec.prt_us if rec.prt_us > 0 else 0.0
    duty = rec.pw_us / rec.prt_us if rec.prt_us > 0 else 0.0
    return "\n".join(
        [
            f"点迹序号: {index + 1} / {total}",
            f"sample_id: {rec.sample_id}",
            f"label/type: {_display_label(rec.label)}",
            f"range: {rec.range_m:.3f} m",
            f"pw: {rec.pw_us:.3f} us",
            f"prt: {rec.prt_us:.3f} us",
            f"prt_nbr: {rec.prt_nbr:.0f}",
            f"prf: {prf_hz:.3f} Hz",
            f"duty: {duty:.6f}",
            f"pluse_band: {rec.pluse_band:.3f} MHz",
            f"sample_freq: {rec.sample_freq:.3f} MHz",
            f"mtd_dim: {_format_dim(rec.mtd_row, rec.mtd_col, rec.mtd_ch)}",
            f"mtd_echo length: {len(rec.mtd_echo)}",
            f"pc_dim: {_format_dim(rec.pc_row, rec.pc_col, rec.pc_ch)}",
            f"pc_echo length: {len(rec.pc_echo)}",
            "",
            "显示说明:",
            "上图: mtd_dim 对应 payload，通常为 [1, 129, 2] 的双通道数据。",
            "下图: pc_dim 对应 pc_echo payload；若当前数据库未录取 pc_echo，则显示空图标注。",
        ]
    )


class TkLogWriter:
    def __init__(self, app: "EchoDatasetViewer") -> None:
        self.app = app

    def write(self, text: str) -> int:
        if text:
            try:
                self.app.after(0, self.app._append_train_log, text)
            except tk.TclError:
                pass
        return len(text)

    def flush(self) -> None:
        return None


class EchoDatasetViewer(tk.Tk):
    def __init__(self, initial_path: str = "") -> None:
        super().__init__()
        self.title("DClsEcho 数据集查看与训练工作台")
        self.geometry("1680x980")
        self.minsize(1360, 820)
        self.configure(bg=APP_BG)

        self.records: list[RadarSample] = []
        self.current_index = -1
        self.load_thread: threading.Thread | None = None
        self.train_thread: threading.Thread | None = None

        default_out = Path(__file__).resolve().parent / "runs" / "gui_train"
        self.path_var = tk.StringVar(value=initial_path)
        self.status_var = tk.StringVar(value="请选择数据集目录、split 目录或 .db/.sqlite/.npz 文件")
        self.index_var = tk.StringVar(value="1")
        self.summary_var = tk.StringVar(value="未加载")
        self.range_bin_width_var = tk.StringVar(value="100")
        self.range_stats_var = tk.StringVar(value="未加载")
        self.no_label_var = tk.BooleanVar(value=False)

        self.train_data_var = tk.StringVar(value="")
        self.val_data_var = tk.StringVar(value="")
        self.test_data_var = tk.StringVar(value="")
        self.out_dir_var = tk.StringVar(value=str(default_out))
        self.epochs_var = tk.IntVar(value=30)
        self.batch_size_var = tk.IntVar(value=32)
        self.lr_var = tk.StringVar(value="0.001")
        self.weight_decay_var = tk.StringVar(value="0.0001")
        self.dropout_var = tk.StringVar(value="0.2")
        self.train_distance_bin_width_var = tk.StringVar(value="100")
        self.device_var = tk.StringVar(value="auto")
        self.class_weight_var = tk.BooleanVar(value=True)
        self.meta_vars = {
            name: tk.BooleanVar(value=name in DEFAULT_META_FEATURE_NAMES)
            for name in META_FEATURE_NAMES
        }

        self._setup_theme()
        self._build_widgets()
        self._update_model_preview()
        if initial_path:
            self.after(200, self.load_dataset)

    def _setup_theme(self) -> None:
        self.style = ttk.Style(self)
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass

        default_font = ("Microsoft YaHei UI", 9)
        heading_font = ("Microsoft YaHei UI", 10, "bold")
        title_font = ("Microsoft YaHei UI", 18, "bold")
        subtitle_font = ("Microsoft YaHei UI", 10)

        self.option_add("*Font", default_font)
        self.option_add("*tearOff", False)
        self.option_add("*takeFocus", "0")

        self.style.configure(".", font=default_font, background=APP_BG, foreground=TEXT_FG)
        self.style.configure("App.TFrame", background=APP_BG)
        self.style.configure("Panel.TFrame", background=PANEL_BG)
        self.style.configure("Header.TFrame", background=ACCENT)
        self.style.configure("HeaderTitle.TLabel", background=ACCENT, foreground="#ffffff", font=title_font)
        self.style.configure("HeaderSub.TLabel", background=ACCENT, foreground="#d6eef6", font=subtitle_font)
        self.style.configure("TLabel", background=PANEL_BG, foreground=TEXT_FG)
        self.style.configure("Panel.TLabel", background=PANEL_BG, foreground=TEXT_FG)
        self.style.configure("Muted.TLabel", background=PANEL_BG, foreground=MUTED_FG)
        self.style.configure("Status.TLabel", background="#dfeaf1", foreground="#314355")
        self.style.configure("TCheckbutton", background=PANEL_BG, foreground=TEXT_FG)
        self.style.map("TCheckbutton", background=[("active", PANEL_BG)], foreground=[("disabled", "#98a2b3")])
        self._setup_checkmark_indicator()
        self.style.configure(
            "TLabelframe",
            background=PANEL_BG,
            bordercolor=BORDER,
            relief="solid",
        )
        self.style.configure(
            "TLabelframe.Label",
            background=PANEL_BG,
            foreground=ACCENT_DARK,
            font=heading_font,
        )
        self.style.configure(
            "TNotebook",
            background=APP_BG,
            borderwidth=0,
            tabmargins=(2, 4, 2, 0),
        )
        self.style.configure(
            "TNotebook.Tab",
            padding=(18, 9),
            background="#d7e4ec",
            foreground="#405466",
            font=heading_font,
        )
        self.style.map(
            "TNotebook.Tab",
            background=[("selected", PANEL_BG), ("active", "#e7f1f6")],
            foreground=[("selected", ACCENT_DARK), ("active", ACCENT_DARK)],
        )
        self.style.configure(
            "TButton",
            padding=(10, 5),
            background="#e7eef4",
            foreground=TEXT_FG,
            bordercolor=BORDER,
            focusthickness=0,
            focuscolor="#e7eef4",
        )
        self.style.map(
            "TButton",
            background=[("active", "#dce9f1"), ("disabled", "#eef2f5")],
            foreground=[("disabled", "#98a2b3")],
        )
        self.style.configure(
            "Accent.TButton",
            background=ACCENT,
            foreground="#ffffff",
            bordercolor=ACCENT_DARK,
            font=heading_font,
        )
        self.style.map(
            "Accent.TButton",
            background=[("active", ACCENT_DARK), ("disabled", "#9bbac7")],
            foreground=[("disabled", "#edf6fa")],
        )
        self.style.configure(
            "Success.TButton",
            background=SUCCESS,
            foreground="#ffffff",
            bordercolor="#1d6548",
            font=heading_font,
        )
        self.style.map("Success.TButton", background=[("active", "#1d6548")])
        self.style.configure(
            "TEntry",
            fieldbackground=TEXT_BG,
            foreground=TEXT_FG,
            bordercolor=BORDER,
            lightcolor=ACCENT_SOFT,
            darkcolor=BORDER,
            padding=(4, 2),
        )
        self.style.configure(
            "TCombobox",
            fieldbackground=TEXT_BG,
            foreground=TEXT_FG,
            bordercolor=BORDER,
            arrowcolor=ACCENT_DARK,
        )
        self.style.configure(
            "Treeview",
            background=TEXT_BG,
            fieldbackground=TEXT_BG,
            foreground=TEXT_FG,
            rowheight=28,
            bordercolor=BORDER,
            borderwidth=1,
        )
        self.style.configure(
            "Treeview.Heading",
            background="#dfeaf1",
            foreground=ACCENT_DARK,
            font=heading_font,
            relief="flat",
        )
        self.style.map(
            "Treeview",
            background=[("selected", ACCENT)],
            foreground=[("selected", "#ffffff")],
        )
        self._remove_button_focus_layout()
        self._remove_notebook_focus_layout()

    def _setup_checkmark_indicator(self) -> None:
        self._check_images = {
            "off": self._make_checkbox_image(False, False),
            "on": self._make_checkbox_image(True, False),
            "off_disabled": self._make_checkbox_image(False, True),
            "on_disabled": self._make_checkbox_image(True, True),
        }
        element = "Checkmark.indicator"
        try:
            self.style.element_create(
                element,
                "image",
                self._check_images["off"],
                ("selected", "disabled", self._check_images["on_disabled"]),
                ("disabled", self._check_images["off_disabled"]),
                ("selected", self._check_images["on"]),
                border=0,
                sticky="",
            )
        except tk.TclError:
            pass
        self.style.layout(
            "Checkmark.TCheckbutton",
            [
                (
                    "Checkbutton.padding",
                    {
                        "sticky": "nswe",
                        "children": [
                            (element, {"side": "left", "sticky": ""}),
                            ("Checkbutton.label", {"side": "left", "sticky": "w"}),
                        ],
                    },
                )
            ],
        )
        self.style.configure("Checkmark.TCheckbutton", background=PANEL_BG, foreground=TEXT_FG)
        self.style.map(
            "Checkmark.TCheckbutton",
            background=[("active", PANEL_BG), ("disabled", PANEL_BG)],
            foreground=[("disabled", "#98a2b3")],
        )

    @staticmethod
    def _make_checkbox_image(checked: bool, disabled: bool) -> tk.PhotoImage:
        size = 18
        image = tk.PhotoImage(width=size, height=size)
        bg = PANEL_BG
        fill = "#ffffff" if not disabled else "#eef2f5"
        border = BORDER if not disabled else "#d7dee6"
        mark = ACCENT if not disabled else "#9aa8b5"
        image.put(bg, to=(0, 0, size, size))
        image.put(fill, to=(2, 2, size - 2, size - 2))
        image.put(border, to=(2, 2, size - 2, 3))
        image.put(border, to=(2, size - 3, size - 2, size - 2))
        image.put(border, to=(2, 2, 3, size - 2))
        image.put(border, to=(size - 3, 2, size - 2, size - 2))
        if checked:
            points = [(5, 9), (6, 10), (7, 11), (8, 12), (9, 11), (10, 10), (11, 9), (12, 8), (13, 7), (14, 6)]
            for x, y in points:
                image.put(mark, to=(x, y, x + 1, y + 1))
                image.put(mark, to=(x, y + 1, x + 1, y + 2))
                image.put(mark, to=(x + 1, y, x + 2, y + 1))
        return image

    def _remove_button_focus_layout(self) -> None:
        layout = [
            (
                "Button.border",
                {
                    "sticky": "nswe",
                    "border": "1",
                    "children": [
                        (
                            "Button.padding",
                            {
                                "sticky": "nswe",
                                "children": [("Button.label", {"sticky": "nswe"})],
                            },
                        )
                    ],
                },
            )
        ]
        for style_name in ("TButton", "Accent.TButton", "Success.TButton"):
            try:
                self.style.layout(style_name, layout)
            except tk.TclError:
                pass

    def _remove_notebook_focus_layout(self) -> None:
        layout = [
            (
                "Notebook.tab",
                {
                    "sticky": "nswe",
                    "children": [
                        (
                            "Notebook.padding",
                            {
                                "side": "top",
                                "sticky": "nswe",
                                "children": [("Notebook.label", {"side": "top", "sticky": ""})],
                            },
                        )
                    ],
                },
            )
        ]
        try:
            self.style.layout("TNotebook.Tab", layout)
        except tk.TclError:
            pass

    def _build_widgets(self) -> None:
        main = ttk.Frame(self, style="App.TFrame", padding=(12, 10, 12, 0))
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        header = ttk.Frame(main, style="Header.TFrame", padding=(18, 14, 18, 12))
        header.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        ttk.Label(header, text="DClsEcho 数据集查看与训练工作台", style="HeaderTitle.TLabel").pack(anchor=tk.W)
        ttk.Label(
            header,
            text="点迹波形检查、目标类型映射、meta 特征选择和三分支模型训练",
            style="HeaderSub.TLabel",
        ).pack(anchor=tk.W, pady=(4, 0))

        notebook = ttk.Notebook(main)
        notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        viewer_tab = ttk.Frame(notebook, style="App.TFrame", padding=(0, 10, 0, 0))
        train_tab = ttk.Frame(notebook, style="App.TFrame", padding=(0, 10, 0, 0))
        notebook.add(viewer_tab, text="数据查看")
        notebook.add(train_tab, text="模型训练")

        self._build_viewer_tab(viewer_tab)
        self._build_train_tab(train_tab)

        status = ttk.Label(self, textvariable=self.status_var, anchor=tk.W, padding=(12, 7), style="Status.TLabel")
        status.pack(side=tk.BOTTOM, fill=tk.X)
        self._disable_focus_rectangles(self)

    def _disable_focus_rectangles(self, widget: tk.Widget) -> None:
        try:
            widget.configure(takefocus=False)
        except tk.TclError:
            pass
        for child in widget.winfo_children():
            self._disable_focus_rectangles(child)

    def _build_viewer_tab(self, parent: ttk.Frame) -> None:
        top = ttk.Frame(parent, style="Panel.TFrame", padding=(12, 10))
        top.pack(side=tk.TOP, fill=tk.X, padx=10)

        ttk.Label(top, text="数据路径").pack(side=tk.LEFT)
        path_entry = ttk.Entry(top, textvariable=self.path_var)
        path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 6))
        ttk.Button(top, text="选择目录", command=self.choose_directory).pack(side=tk.LEFT, padx=3)
        ttk.Button(top, text="选择文件", command=self.choose_file).pack(side=tk.LEFT, padx=3)
        ttk.Checkbutton(
            top,
            text="无标签解析",
            variable=self.no_label_var,
            style="Checkmark.TCheckbutton",
            takefocus=False,
        ).pack(side=tk.LEFT, padx=8)
        self.load_button = ttk.Button(top, text="加载数据集", command=self.load_dataset, style="Accent.TButton")
        self.load_button.pack(side=tk.LEFT, padx=3)

        paned = ttk.Panedwindow(parent, orient=tk.HORIZONTAL)
        paned.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(10, 8))

        left = ttk.Frame(paned, width=400, style="App.TFrame")
        right = ttk.Frame(paned, style="App.TFrame")
        paned.add(left, weight=1)
        paned.add(right, weight=6)

        summary_frame = ttk.LabelFrame(left, text="数据集概要", padding=8)
        summary_frame.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(summary_frame, textvariable=self.summary_var, justify=tk.LEFT).pack(anchor=tk.W)

        range_frame = ttk.LabelFrame(left, text="距离分布统计", padding=8)
        range_frame.pack(side=tk.TOP, fill=tk.BOTH, pady=(8, 0))
        range_controls = ttk.Frame(range_frame, style="Panel.TFrame")
        range_controls.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(range_controls, text="距离段(m)").pack(side=tk.LEFT)
        ttk.Entry(range_controls, textvariable=self.range_bin_width_var, width=8).pack(side=tk.LEFT, padx=(6, 6))
        ttk.Button(range_controls, text="统计", command=self.refresh_range_distribution).pack(side=tk.LEFT)
        ttk.Label(range_frame, textvariable=self.range_stats_var, style="Muted.TLabel", justify=tk.LEFT).pack(
            side=tk.TOP,
            fill=tk.X,
            pady=(6, 4),
        )
        range_table = ttk.Frame(range_frame, style="Panel.TFrame")
        range_table.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.range_tree = ttk.Treeview(range_table, show="headings", height=7)
        range_scroll_y = ttk.Scrollbar(range_table, orient=tk.VERTICAL, command=self.range_tree.yview)
        range_scroll_x = ttk.Scrollbar(range_table, orient=tk.HORIZONTAL, command=self.range_tree.xview)
        self.range_tree.configure(yscrollcommand=range_scroll_y.set, xscrollcommand=range_scroll_x.set)
        self.range_tree.grid(row=0, column=0, sticky="nsew")
        range_scroll_y.grid(row=0, column=1, sticky="ns")
        range_scroll_x.grid(row=1, column=0, sticky="ew")
        range_table.rowconfigure(0, weight=1)
        range_table.columnconfigure(0, weight=1)
        self._clear_range_distribution()

        nav = ttk.LabelFrame(left, text="点迹选择", padding=8)
        nav.pack(side=tk.TOP, fill=tk.X, pady=(8, 0))
        ttk.Button(nav, text="上一条", command=lambda: self.move_index(-1)).pack(side=tk.LEFT)
        ttk.Entry(nav, textvariable=self.index_var, width=8).pack(side=tk.LEFT, padx=6)
        ttk.Button(nav, text="跳转", command=self.jump_to_index).pack(side=tk.LEFT)
        ttk.Button(nav, text="下一条", command=lambda: self.move_index(1)).pack(side=tk.LEFT, padx=(6, 0))

        list_frame = ttk.LabelFrame(left, text="点迹列表", padding=6)
        list_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(8, 0))
        columns = ("idx", "label", "range", "prt", "mtd", "pc")
        self.tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=18)
        headings = {
            "idx": "序号",
            "label": "目标类型",
            "range": "距离(m)",
            "prt": "脉冲数",
            "mtd": "mtd_dim",
            "pc": "pc_dim",
        }
        widths = {"idx": 58, "label": 155, "range": 82, "prt": 62, "mtd": 94, "pc": 94}
        for col in columns:
            self.tree.heading(col, text=headings[col])
            self.tree.column(col, width=widths[col], anchor=tk.CENTER, stretch=False)
        tree_scroll_y = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.tree.yview)
        tree_scroll_x = ttk.Scrollbar(list_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=tree_scroll_y.set, xscrollcommand=tree_scroll_x.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        tree_scroll_y.grid(row=0, column=1, sticky="ns")
        tree_scroll_x.grid(row=1, column=0, sticky="ew")
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)
        self.tree.tag_configure("even", background=TEXT_BG)
        self.tree.tag_configure("odd", background="#f2f7fb")
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)

        info_frame = ttk.LabelFrame(right, text="当前点迹信息", padding=8)
        info_frame.pack(side=tk.TOP, fill=tk.X)
        self.info_text = tk.Text(info_frame, height=7, wrap=tk.WORD)
        self._style_text_widget(self.info_text)
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        info_scroll = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=self.info_text.yview)
        info_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.info_text.configure(yscrollcommand=info_scroll.set)

        plot_frame = ttk.Frame(right, style="Panel.TFrame", padding=8)
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(8, 0))
        self.figure = Figure(figsize=(13.6, 8.4), dpi=100)
        self.figure.patch.set_facecolor(PANEL_BG)
        self.ax_mtd = self.figure.add_subplot(3, 1, 1)
        self.ax_pc = self.figure.add_subplot(3, 1, 2)
        self.ax_fft = self.figure.add_subplot(3, 1, 3)
        self.figure.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        self._clear_plot()

    def _build_train_tab(self, parent: ttk.Frame) -> None:
        paned = ttk.Panedwindow(parent, orient=tk.HORIZONTAL)
        paned.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=8)

        left = ttk.Frame(paned, width=470, style="App.TFrame")
        right = ttk.Frame(paned, style="App.TFrame")
        paned.add(left, weight=1)
        paned.add(right, weight=3)

        data_frame = ttk.LabelFrame(left, text="训练数据", padding=8)
        data_frame.pack(side=tk.TOP, fill=tk.X)
        self._path_row(data_frame, "训练集", self.train_data_var, 0)
        self._path_row(data_frame, "验证集", self.val_data_var, 1)
        self._path_row(data_frame, "测试集", self.test_data_var, 2)
        self._path_row(data_frame, "输出目录", self.out_dir_var, 3, file_button=False)
        ttk.Button(data_frame, text="从查看路径自动填入 train/test", command=self.autofill_train_test_paths).grid(
            row=4,
            column=1,
            columnspan=3,
            sticky="ew",
            pady=(6, 0),
        )
        data_frame.columnconfigure(1, weight=1)

        param_frame = ttk.LabelFrame(left, text="训练参数", padding=8)
        param_frame.pack(side=tk.TOP, fill=tk.X, pady=(8, 0))
        self._entry_row(param_frame, "epochs", self.epochs_var, 0)
        self._entry_row(param_frame, "batch_size", self.batch_size_var, 1)
        self._entry_row(param_frame, "lr", self.lr_var, 2)
        self._entry_row(param_frame, "weight_decay", self.weight_decay_var, 3)
        self._entry_row(param_frame, "dropout", self.dropout_var, 4)
        self._entry_row(param_frame, "评估距离段(m)", self.train_distance_bin_width_var, 5)
        ttk.Label(param_frame, text="device").grid(row=6, column=0, sticky="w", pady=3)
        ttk.Combobox(
            param_frame,
            textvariable=self.device_var,
            values=("auto", "cpu", "cuda"),
            state="readonly",
            width=12,
        ).grid(row=6, column=1, sticky="ew", pady=3)
        ttk.Checkbutton(
            param_frame,
            text="使用类别权重",
            variable=self.class_weight_var,
            style="Checkmark.TCheckbutton",
            takefocus=False,
        ).grid(
            row=7,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(4, 0),
        )
        param_frame.columnconfigure(1, weight=1)

        action_frame = ttk.Frame(left, style="App.TFrame")
        action_frame.pack(side=tk.TOP, fill=tk.X, pady=(8, 0))
        self.train_button = ttk.Button(action_frame, text="开始训练", command=self.start_training, style="Success.TButton")
        self.train_button.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(action_frame, text="清空日志", command=self._clear_train_log).pack(side=tk.LEFT, padx=(8, 0))

        meta_frame = ttk.LabelFrame(left, text="meta 特征选择", padding=8)
        meta_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(8, 0))
        meta_frame.rowconfigure(0, weight=1)
        meta_frame.columnconfigure(0, weight=1)
        meta_canvas = tk.Canvas(meta_frame, background=PANEL_BG, highlightthickness=0, borderwidth=0)
        meta_scroll = ttk.Scrollbar(meta_frame, orient=tk.VERTICAL, command=meta_canvas.yview)
        meta_canvas.configure(yscrollcommand=meta_scroll.set)
        meta_canvas.grid(row=0, column=0, sticky="nsew")
        meta_scroll.grid(row=0, column=1, sticky="ns")
        meta_inner = ttk.Frame(meta_canvas, style="Panel.TFrame")
        meta_window = meta_canvas.create_window((0, 0), window=meta_inner, anchor=tk.NW)

        def update_meta_scrollregion(_event: tk.Event) -> None:
            meta_canvas.configure(scrollregion=meta_canvas.bbox("all"))

        def fit_meta_inner_width(event: tk.Event) -> None:
            meta_canvas.itemconfigure(meta_window, width=event.width)

        def wheel_meta(event: tk.Event) -> None:
            meta_canvas.yview_scroll(-int(event.delta / 120), "units")

        meta_inner.bind("<Configure>", update_meta_scrollregion)
        meta_canvas.bind("<Configure>", fit_meta_inner_width)
        meta_canvas.bind("<MouseWheel>", wheel_meta)
        meta_inner.bind("<MouseWheel>", wheel_meta)
        for row, name in enumerate(META_FEATURE_NAMES):
            ttk.Checkbutton(
                meta_inner,
                text=f"{row + 1}. {META_DESCRIPTIONS.get(name, name)}",
                variable=self.meta_vars[name],
                command=self._update_model_preview,
                style="Checkmark.TCheckbutton",
                takefocus=False,
            ).grid(row=row, column=0, sticky="w", pady=1)
        button_row = len(META_FEATURE_NAMES)
        ttk.Button(meta_inner, text="默认特征", command=self.select_core_meta).grid(
            row=button_row,
            column=0,
            sticky="ew",
            pady=(8, 2),
        )
        ttk.Button(meta_inner, text="全部8维", command=self.select_all_meta).grid(
            row=button_row + 1,
            column=0,
            sticky="ew",
            pady=2,
        )
        meta_inner.columnconfigure(0, weight=1)

        model_frame = ttk.LabelFrame(right, text="模型结构预览", padding=8)
        model_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        model_frame.rowconfigure(0, weight=1)
        model_frame.columnconfigure(0, weight=2)
        model_frame.columnconfigure(1, weight=1)

        self.model_figure = Figure(figsize=(8.4, 4.4), dpi=100)
        self.model_figure.patch.set_facecolor(PANEL_BG)
        self.model_ax = self.model_figure.add_subplot(1, 1, 1)
        self.model_canvas = FigureCanvasTkAgg(self.model_figure, master=model_frame)
        self.model_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self.model_text = tk.Text(model_frame, width=38, height=12, wrap=tk.WORD)
        self._style_text_widget(self.model_text)
        self.model_text.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        self.model_text.configure(state=tk.DISABLED)

        log_frame = ttk.LabelFrame(right, text="训练日志", padding=8)
        log_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(8, 0))
        self.train_log = tk.Text(log_frame, height=14, wrap=tk.WORD)
        self._style_text_widget(self.train_log, mono=True)
        self.train_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.train_log.yview)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.train_log.configure(yscrollcommand=log_scroll.set)

    def _path_row(
        self,
        parent: ttk.LabelFrame,
        label: str,
        var: tk.StringVar,
        row: int,
        file_button: bool = True,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky="ew", padx=(6, 4), pady=3)
        ttk.Button(parent, text="目录", command=lambda v=var: self.choose_path_for_var(v, True)).grid(
            row=row,
            column=2,
            sticky="ew",
            padx=2,
            pady=3,
        )
        if file_button:
            ttk.Button(parent, text="文件", command=lambda v=var: self.choose_path_for_var(v, False)).grid(
                row=row,
                column=3,
                sticky="ew",
                padx=2,
                pady=3,
            )

    @staticmethod
    def _entry_row(parent: ttk.LabelFrame, label: str, var: tk.Variable, row: int) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky="ew", padx=(8, 0), pady=3)

    @staticmethod
    def _style_text_widget(widget: tk.Text, mono: bool = False) -> None:
        font = ("Consolas", 9) if mono else ("Microsoft YaHei UI", 9)
        widget.configure(
            background=TEXT_BG,
            foreground=TEXT_FG,
            insertbackground=ACCENT_DARK,
            selectbackground=ACCENT,
            selectforeground="#ffffff",
            relief=tk.FLAT,
            borderwidth=0,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=ACCENT,
            padx=8,
            pady=6,
            font=font,
        )

    @staticmethod
    def _style_axis(ax) -> None:
        ax.set_facecolor(TEXT_BG)
        ax.tick_params(colors=MUTED_FG, labelsize=8)
        ax.xaxis.label.set_color(MUTED_FG)
        ax.yaxis.label.set_color(MUTED_FG)
        ax.title.set_color(TEXT_FG)
        for spine in ax.spines.values():
            spine.set_color(BORDER)

    def choose_directory(self) -> None:
        path = filedialog.askdirectory(title="选择数据集目录")
        if path:
            self.path_var.set(path)

    def choose_file(self) -> None:
        path = filedialog.askopenfilename(
            title="选择数据文件",
            filetypes=[
                ("Dataset files", "*.db *.sqlite *.sqlite3 *.npz"),
                ("SQLite DB", "*.db *.sqlite *.sqlite3"),
                ("NPZ", "*.npz"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.path_var.set(path)

    def choose_path_for_var(self, var: tk.StringVar, directory: bool) -> None:
        if directory:
            path = filedialog.askdirectory(title="选择目录")
        else:
            path = filedialog.askopenfilename(
                title="选择数据文件",
                filetypes=[
                    ("Dataset files", "*.db *.sqlite *.sqlite3 *.npz"),
                    ("SQLite DB", "*.db *.sqlite *.sqlite3"),
                    ("NPZ", "*.npz"),
                    ("All files", "*.*"),
                ],
            )
        if path:
            var.set(path)

    def autofill_train_test_paths(self) -> None:
        raw = self.path_var.get().strip()
        if not raw:
            raw = filedialog.askdirectory(title="选择包含 train/test 的数据集根目录")
        if not raw:
            return
        root = Path(raw).expanduser()
        train_dir = root / "train"
        test_dir = root / "test"
        if train_dir.exists():
            self.train_data_var.set(str(train_dir))
        else:
            self.train_data_var.set(str(root))
        if test_dir.exists():
            self.test_data_var.set(str(test_dir))
        self.status_var.set("已根据查看路径填入训练集/测试集路径")

    def load_dataset(self) -> None:
        path = self.path_var.get().strip()
        if not path:
            messagebox.showwarning("缺少路径", "请先选择数据路径。")
            return
        if self.load_thread is not None and self.load_thread.is_alive():
            return

        self.load_button.configure(state=tk.DISABLED)
        self.status_var.set("正在解析数据集，请稍候...")
        self.summary_var.set("加载中...")
        self.records = []
        self.current_index = -1
        self._clear_tree()
        self._clear_range_distribution("加载中...")
        self._clear_plot()
        self._set_info("")

        require_label = not self.no_label_var.get()

        def worker() -> None:
            try:
                records = load_records(path, require_label=require_label)
            except Exception as exc:  # noqa: BLE001 - UI diagnostic
                self.after(0, lambda: self._finish_load_error(exc))
                return
            self.after(0, lambda: self._finish_load_success(path, records))

        self.load_thread = threading.Thread(target=worker, daemon=True)
        self.load_thread.start()

    def _finish_load_error(self, exc: Exception) -> None:
        self.load_button.configure(state=tk.NORMAL)
        self.status_var.set("加载失败")
        self.summary_var.set("加载失败")
        self._clear_range_distribution("加载失败")
        messagebox.showerror("加载失败", str(exc))

    def _finish_load_success(self, path: str, records: list[RadarSample]) -> None:
        self.load_button.configure(state=tk.NORMAL)
        self.records = records
        self.current_index = 0 if records else -1
        self._populate_summary(path)
        self.refresh_range_distribution(show_errors=False)
        self._populate_tree()
        if records:
            self.show_record(0)
        self.status_var.set(f"加载完成: {len(records)} 条点迹")

    def _populate_summary(self, path: str) -> None:
        label_counts = Counter(_display_label(r.label) for r in self.records)
        mtd_dims = Counter(_format_dim(r.mtd_row, r.mtd_col, r.mtd_ch) for r in self.records)
        pc_dims = Counter(_format_dim(r.pc_row, r.pc_col, r.pc_ch) for r in self.records)
        lines = [
            f"路径: {path}",
            f"点迹数: {len(self.records)}",
            f"列表预览: 前 {min(len(self.records), TREE_PREVIEW_LIMIT)} 条，可输入序号跳转任意点迹",
            f"目标类型统计: {dict(label_counts.most_common())}",
            f"mtd_dim: {dict(mtd_dims.most_common(5))}",
            f"pc_dim: {dict(pc_dims.most_common(5))}",
        ]
        self.summary_var.set("\n".join(lines))

    def _parse_range_bin_width(self) -> float:
        try:
            bin_width = float(self.range_bin_width_var.get())
        except ValueError as exc:
            raise ValueError("距离段宽度必须是数字。") from exc
        if not np.isfinite(bin_width) or bin_width <= 0:
            raise ValueError("距离段宽度必须大于 0。")
        return float(bin_width)

    def _clear_range_distribution(self, message: str = "未加载") -> None:
        if not hasattr(self, "range_tree"):
            return
        self.range_stats_var.set(message)
        self.range_tree.configure(columns=("range_bin", "total"))
        self.range_tree.heading("range_bin", text="距离段(m)")
        self.range_tree.heading("total", text="总数")
        self.range_tree.column("range_bin", width=110, minwidth=92, anchor=tk.CENTER, stretch=False)
        self.range_tree.column("total", width=58, minwidth=52, anchor=tk.CENTER, stretch=False)
        for iid in self.range_tree.get_children():
            self.range_tree.delete(iid)

    def refresh_range_distribution(self, show_errors: bool = True) -> None:
        if not self.records:
            self._clear_range_distribution("未加载数据")
            return
        try:
            bin_width = self._parse_range_bin_width()
        except ValueError as exc:
            self.range_stats_var.set(str(exc))
            if show_errors:
                messagebox.showwarning("距离段设置错误", str(exc))
            return

        labels, rows, valid_count, span = _range_distribution(self.records, bin_width)
        if span is None:
            self._clear_range_distribution("没有可统计的有效距离")
            return

        label_columns = [f"label_{idx}" for idx in range(len(labels))]
        columns = ("range_bin", "total", *label_columns)
        self.range_tree.configure(columns=columns)
        self.range_tree.heading("range_bin", text="距离段(m)")
        self.range_tree.heading("total", text="总数")
        self.range_tree.column("range_bin", width=110, minwidth=92, anchor=tk.CENTER, stretch=False)
        self.range_tree.column("total", width=58, minwidth=52, anchor=tk.CENTER, stretch=False)
        for col, label in zip(label_columns, labels):
            self.range_tree.heading(col, text=label)
            self.range_tree.column(col, width=128, minwidth=96, anchor=tk.CENTER, stretch=False)

        for iid in self.range_tree.get_children():
            self.range_tree.delete(iid)
        for row_idx, (_start, bin_label, total, counts) in enumerate(rows):
            values = [bin_label, total, *(counts.get(label, 0) for label in labels)]
            self.range_tree.insert("", tk.END, iid=f"range_{row_idx}", values=values)

        skipped = len(self.records) - valid_count
        suffix = f"；跳过无效距离 {skipped} 条" if skipped else ""
        min_range, max_range = span
        self.range_stats_var.set(
            f"范围 {_format_distance_edge(min_range)}-{_format_distance_edge(max_range)} m；"
            f"段宽 {_format_distance_edge(bin_width)} m；有效 {valid_count} 条{suffix}"
        )

    def _populate_tree(self) -> None:
        self._clear_tree()
        for idx, rec in enumerate(self.records[:TREE_PREVIEW_LIMIT]):
            self.tree.insert(
                "",
                tk.END,
                iid=str(idx),
                values=(
                    idx + 1,
                    _display_label(rec.label),
                    f"{rec.range_m:.1f}",
                    f"{rec.prt_nbr:.0f}",
                    _format_dim(rec.mtd_row, rec.mtd_col, rec.mtd_ch),
                    _format_dim(rec.pc_row, rec.pc_col, rec.pc_ch),
                ),
                tags=("odd" if idx % 2 else "even",),
            )

    def _clear_tree(self) -> None:
        for iid in self.tree.get_children():
            self.tree.delete(iid)

    def _clear_plot(self) -> None:
        self.ax_mtd.clear()
        self.ax_pc.clear()
        self.ax_fft.clear()
        self.ax_mtd.set_title("mtd_dim [1,129,2] 数据")
        self.ax_pc.set_title("pc_dim 数据")
        self.ax_fft.set_title("pc_echo FFT 频谱")
        for ax in (self.ax_mtd, self.ax_pc, self.ax_fft):
            self._style_axis(ax)
            ax.set_xlabel("sample index")
            ax.set_ylabel("amplitude")
            ax.grid(True, alpha=0.25, color="#9fb7c8")
        self.ax_fft.set_xlabel("frequency (Hz)")
        self.ax_fft.set_ylabel("magnitude (dB)")
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _set_info(self, text: str) -> None:
        self.info_text.configure(state=tk.NORMAL)
        self.info_text.delete("1.0", tk.END)
        self.info_text.insert("1.0", text)
        self.info_text.configure(state=tk.DISABLED)

    def on_tree_select(self, _event: tk.Event) -> None:
        selection = self.tree.selection()
        if not selection:
            return
        try:
            index = int(selection[0])
        except ValueError:
            return
        if index != self.current_index:
            self.show_record(index, select_tree=False)

    def jump_to_index(self) -> None:
        if not self.records:
            return
        try:
            index = int(self.index_var.get()) - 1
        except ValueError:
            messagebox.showwarning("序号错误", "请输入有效的点迹序号。")
            return
        self.show_record(index)

    def move_index(self, delta: int) -> None:
        if not self.records:
            return
        base = self.current_index if self.current_index >= 0 else 0
        self.show_record(base + int(delta))

    def show_record(self, index: int, select_tree: bool = True) -> None:
        if not self.records:
            return
        index = max(0, min(int(index), len(self.records) - 1))
        self.current_index = index
        self.index_var.set(str(index + 1))
        rec = self.records[index]
        self._set_info(_record_info(rec, index, len(self.records)))
        self._draw_record(rec)
        self.status_var.set(f"当前点迹: {index + 1} / {len(self.records)}")

        if select_tree:
            iid = str(index)
            if self.tree.exists(iid):
                self.tree.selection_set(iid)
                self.tree.focus(iid)
                self.tree.see(iid)

    def _draw_record(self, rec: RadarSample) -> None:
        mtd_data, mtd_status = _reshape_echo(rec.mtd_echo, rec.mtd_row, rec.mtd_col, rec.mtd_ch)
        pc_data, pc_status = _reshape_echo(rec.pc_echo, rec.pc_row, rec.pc_col, rec.pc_ch)

        self.ax_mtd.clear()
        self.ax_pc.clear()
        self.ax_fft.clear()
        self._draw_series(
            self.ax_mtd,
            f"mtd_echo 脉压/距离维数据: mtd_dim={_format_dim(rec.mtd_row, rec.mtd_col, rec.mtd_ch)}",
            mtd_data,
            int(rec.mtd_col),
            mtd_status,
        )
        self._draw_series(
            self.ax_pc,
            f"pc_echo 多普勒/脉冲维数据: pc_dim={_format_dim(rec.pc_row, rec.pc_col, rec.pc_ch)}",
            pc_data,
            int(rec.pc_col),
            pc_status,
        )
        self._draw_fft_series(
            self.ax_fft,
            f"pc_echo FFT 频谱: pc_dim={_format_dim(rec.pc_row, rec.pc_col, rec.pc_ch)}",
            pc_data,
            pc_status,
            rec.prt_us,
        )
        self.figure.tight_layout()
        self.canvas.draw_idle()

    @staticmethod
    def _draw_series(ax, title: str, data: np.ndarray, expected_points: int, status: str) -> None:
        ax.set_title(title)
        EchoDatasetViewer._style_axis(ax)
        ax.set_xlabel("sample index")
        ax.set_ylabel("amplitude")
        ax.grid(True, alpha=0.25, color="#9fb7c8")
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
        ax.plot(x, _series_abs(data), label="ABS", linewidth=1.7, color=ACCENT)
        ax.plot(x, data[:, 0], label="CH1", linewidth=0.9, alpha=0.78, color="#d4872d")
        if data.shape[1] >= 2:
            ax.plot(x, data[:, 1], label="CH2", linewidth=0.9, alpha=0.78, color="#5f7dc8")
        ax.legend(loc="upper right", fontsize=8, frameon=True, facecolor=PANEL_BG, edgecolor=BORDER)
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

    @staticmethod
    def _draw_fft_series(ax, title: str, data: np.ndarray, status: str, prt_us: float) -> None:
        ax.set_title(title)
        EchoDatasetViewer._style_axis(ax)
        ax.set_ylabel("magnitude (dB)")
        ax.grid(True, alpha=0.25, color="#9fb7c8")
        bins, magnitude, mode, x_label = _fft_spectrum(data, prt_us)
        ax.set_xlabel(x_label)
        if bins.size == 0:
            ax.text(
                0.5,
                0.5,
                f"no complex IQ FFT data\nmode: {mode}\nstatus: {status}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return

        ax.plot(bins, magnitude, label="20log10(|FFT|)", linewidth=1.4, color="#7a5bc7")
        ax.axvline(0.0, color=BORDER, linewidth=0.9, linestyle="--")
        ax.legend(loc="upper right", fontsize=8, frameon=True, facecolor=PANEL_BG, edgecolor=BORDER)
        ax.text(
            0.01,
            0.98,
            f"points={data.shape[0]}, channels={data.shape[1]}, mode={mode}, status={status}",
            ha="left",
            va="top",
            transform=ax.transAxes,
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )

    def select_core_meta(self) -> None:
        for name, var in self.meta_vars.items():
            var.set(name in DEFAULT_META_FEATURE_NAMES)
        self._update_model_preview()

    def select_all_meta(self) -> None:
        for var in self.meta_vars.values():
            var.set(True)
        self._update_model_preview()

    def _selected_meta_features(self) -> list[str]:
        return [name for name in META_FEATURE_NAMES if self.meta_vars[name].get()]

    def _update_model_preview(self) -> None:
        if not hasattr(self, "model_ax"):
            return
        selected = self._selected_meta_features()
        meta_dim = len(selected)
        class_count = 3
        total_params = 0
        try:
            model = RadarThreeBranchNet(num_classes=class_count, meta_dim=max(meta_dim, 1))
            total_params = sum(int(p.numel()) for p in model.parameters())
        except Exception:
            total_params = 0

        self._draw_model_diagram(meta_dim)
        text = [
            "输入/输出约定",
            "",
            "mtd_echo: [batch, prt_nbr]",
            "mtd_mask: [batch, prt_nbr]",
            "pc_echo: [batch, prt_nbr]",
            "pc_mask: [batch, prt_nbr]",
            "pc_spectrum: [batch, prt_nbr]",
            "pc_spectrum_mask: [batch, prt_nbr]",
            f"meta: [batch, {meta_dim}]",
            f"logits: [batch, {class_count}]",
            "",
            f"已选择 meta 特征: {meta_dim}",
            *(f"- {name}" for name in selected),
        ]
        if total_params:
            text.extend(["", f"预览参数量: {total_params:,}"])
        self.model_text.configure(state=tk.NORMAL)
        self.model_text.delete("1.0", tk.END)
        self.model_text.insert("1.0", "\n".join(text))
        self.model_text.configure(state=tk.DISABLED)

    def _draw_model_diagram(self, meta_dim: int) -> None:
        ax = self.model_ax
        ax.clear()
        ax.set_facecolor(PANEL_BG)
        ax.axis("off")
        boxes = [
            ("mtd_echo\n长度 prt_nbr", 0.05, 0.76, 0.18, 0.14, "#eef4f8", "#8aa8b8"),
            ("MTD 分支\n1D CNN + Pool", 0.31, 0.76, 0.20, 0.14, "#dcecf6", ACCENT),
            ("pc_echo\n慢时间", 0.05, 0.53, 0.18, 0.14, "#eef4f8", "#8aa8b8"),
            ("PC 时域分支\n1D CNN + Pool", 0.31, 0.53, 0.20, 0.14, "#f6ead8", "#c78335"),
            ("pc_spectrum\nFFT+shift+abs", 0.05, 0.30, 0.18, 0.14, "#eef4f8", "#8aa8b8"),
            ("PC 频谱分支\n1D CNN + Pool", 0.31, 0.30, 0.20, 0.14, "#f2e7fb", "#7a5bc7"),
            (f"meta\n{meta_dim} 维", 0.05, 0.07, 0.18, 0.14, "#eef4f8", "#8aa8b8"),
            ("Meta 分支\nMLP", 0.31, 0.07, 0.20, 0.14, "#e2f1e8", SUCCESS),
            ("特征拼接", 0.60, 0.42, 0.16, 0.16, "#e2eef4", ACCENT_DARK),
            ("分类头\nMLP", 0.81, 0.42, 0.14, 0.16, "#e5f3ee", SUCCESS),
        ]
        for label, x, y, w, h, face, edge in boxes:
            ax.add_patch(
                matplotlib.patches.FancyBboxPatch(
                    (x, y),
                    w,
                    h,
                    boxstyle="round,pad=0.02,rounding_size=0.015",
                    linewidth=1.0,
                    edgecolor=edge,
                    facecolor=face,
                )
            )
            ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9)

        arrows = [
            ((0.23, 0.83), (0.31, 0.83)),
            ((0.23, 0.60), (0.31, 0.60)),
            ((0.23, 0.37), (0.31, 0.37)),
            ((0.23, 0.14), (0.31, 0.14)),
            ((0.51, 0.83), (0.60, 0.52)),
            ((0.51, 0.60), (0.60, 0.48)),
            ((0.51, 0.37), (0.60, 0.44)),
            ((0.51, 0.14), (0.60, 0.40)),
            ((0.76, 0.50), (0.81, 0.50)),
        ]
        for start, end in arrows:
            ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": ACCENT_DARK, "lw": 1.2})
        ax.text(0.88, 0.32, "3 类 logits", ha="center", va="center", fontsize=9, color=TEXT_FG)
        self.model_figure.tight_layout()
        self.model_canvas.draw_idle()

    def start_training(self) -> None:
        if self.train_thread is not None and self.train_thread.is_alive():
            return
        train_path = self.train_data_var.get().strip()
        test_path = self.test_data_var.get().strip()
        val_path = self.val_data_var.get().strip()
        out_dir = self.out_dir_var.get().strip()
        if not train_path:
            self.autofill_train_test_paths()
            train_path = self.train_data_var.get().strip()
            test_path = self.test_data_var.get().strip()
        if not train_path:
            messagebox.showwarning("缺少训练集", "请先选择训练集路径。")
            return
        if not out_dir:
            messagebox.showwarning("缺少输出目录", "请先选择权重保存目录。")
            return
        selected_meta = self._selected_meta_features()
        if not selected_meta:
            messagebox.showwarning("缺少 meta 特征", "请至少选择一个 meta 特征。")
            return
        try:
            cfg = TrainConfig(
                train_data=train_path,
                val_data=val_path,
                test_data=test_path,
                out_dir=out_dir,
                epochs=int(self.epochs_var.get()),
                batch_size=int(self.batch_size_var.get()),
                lr=float(self.lr_var.get()),
                weight_decay=float(self.weight_decay_var.get()),
                dropout=float(self.dropout_var.get()),
                class_weight=bool(self.class_weight_var.get()),
                distance_bin_width_m=float(self.train_distance_bin_width_var.get()),
                device=self.device_var.get(),
                meta_features=",".join(selected_meta),
            )
        except Exception as exc:  # noqa: BLE001 - UI validation
            messagebox.showerror("训练参数错误", str(exc))
            return

        self._clear_train_log()
        self._append_train_log("开始训练...\n")
        self._append_train_log(f"训练集: {cfg.train_data}\n")
        if cfg.val_data:
            self._append_train_log(f"验证集: {cfg.val_data}\n")
        if cfg.test_data:
            self._append_train_log(f"测试集: {cfg.test_data}\n")
        self._append_train_log(f"输出目录: {cfg.out_dir}\n")
        self._append_train_log(f"meta_features: {selected_meta}\n")
        self._append_train_log(f"distance_bin_width_m: {cfg.distance_bin_width_m:g}\n\n")
        self.train_button.configure(state=tk.DISABLED)
        self.status_var.set("训练中...")

        def worker() -> None:
            writer = TkLogWriter(self)
            try:
                with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                    summary = train_model(cfg)
            except Exception as exc:  # noqa: BLE001 - UI diagnostic
                self.after(0, lambda: self._finish_training_error(exc))
                return
            self.after(0, lambda: self._finish_training_success(summary, cfg))

        self.train_thread = threading.Thread(target=worker, daemon=True)
        self.train_thread.start()

    def _finish_training_error(self, exc: Exception) -> None:
        self.train_button.configure(state=tk.NORMAL)
        self.status_var.set("训练失败")
        self._append_train_log(f"\n训练失败: {exc}\n")
        messagebox.showerror("训练失败", str(exc))

    def _finish_training_success(self, summary: dict, cfg: TrainConfig) -> None:
        self.train_button.configure(state=tk.NORMAL)
        self.status_var.set("训练完成")
        best_path = Path(cfg.out_dir).expanduser().resolve() / "best.pth"
        last_path = Path(cfg.out_dir).expanduser().resolve() / "last.pth"
        best_acc = summary.get("best_val_accuracy", None)
        self._append_train_log("\n训练完成。\n")
        if best_acc is not None:
            self._append_train_log(f"best_val_accuracy: {float(best_acc):.4f}\n")
        self._append_train_log(f"best.pth: {best_path}\n")
        self._append_train_log(f"last.pth: {last_path}\n")

    def _append_train_log(self, text: str) -> None:
        self.train_log.configure(state=tk.NORMAL)
        self.train_log.insert(tk.END, text)
        self.train_log.see(tk.END)
        self.train_log.configure(state=tk.NORMAL)

    def _clear_train_log(self) -> None:
        self.train_log.configure(state=tk.NORMAL)
        self.train_log.delete("1.0", tk.END)
        self.train_log.configure(state=tk.NORMAL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open an interactive DClsEcho dataset workbench.")
    parser.add_argument("data_path", nargs="?", default="", help="optional dataset path to load on startup")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = EchoDatasetViewer(initial_path=args.data_path)
    app.mainloop()


if __name__ == "__main__":
    main()
