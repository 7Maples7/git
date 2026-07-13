# -*- coding: utf-8 -*-
"""Target type code names used by the recorded radar echo databases."""

from __future__ import annotations

from typing import Any

import numpy as np


TARGET_TYPE_NAMES: dict[int, tuple[str, str]] = {
    0x0000: ("InvalidTarget", "无效目标"),
    0x0105: ("FixedWingUav", "固定翼无人机"),
    0x0106: ("MultiRotorUav", "多旋翼无人机"),
    0x0501: ("SmallBird", "小型鸟"),
    0x0601: ("GroundClutter", "地杂波"),
}


def target_type_name(value: Any) -> str:
    """Return a concise human-readable name for a raw target type code."""
    if value is None:
        return "无标签"
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        value = value.item()
    try:
        code = int(value)
    except (TypeError, ValueError):
        return str(value)
    english, chinese = TARGET_TYPE_NAMES.get(code, (f"UnknownTarget({code})", "未知目标"))
    return f"{chinese} / {english}"


def format_target_type(value: Any, show_code: bool = True) -> str:
    """Format a target type for UI/report display."""
    name = target_type_name(value)
    if not show_code or value is None:
        return name
    try:
        code = int(value.item() if isinstance(value, np.generic) else value)
    except (TypeError, ValueError):
        return name
    return f"{name} [{code} / 0x{code:04X}]"
