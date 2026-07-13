# -*- coding: utf-8 -*-
"""Create a small synthetic .npz dataset for smoke testing the training code."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def make_sample(label: int, rng: np.random.Generator):
    pulse_count = int(rng.integers(32, 96))
    pw_us = float(rng.uniform(5, 30))
    prf = float(rng.uniform(800, 2500))
    t = np.arange(pulse_count, dtype=np.float32) / max(prf, 1.0)

    if label == 0:  # uav: tonal rotor-like spectral lines
        f1 = rng.uniform(80, 180)
        f2 = f1 * rng.uniform(1.8, 2.4)
        mtd_echo = 1.2 * np.sin(2 * np.pi * f1 * t) + 0.6 * np.sin(2 * np.pi * f2 * t)
        pc_center = 64 + rng.normal(0, 2)
        pc_width = rng.uniform(4, 8)
    elif label == 1:  # bird: lower frequency flapping and broader range profile
        f1 = rng.uniform(8, 25)
        mtd_echo = 1.0 * np.sin(2 * np.pi * f1 * t) * (1.0 + 0.5 * np.sin(2 * np.pi * f1 * 0.3 * t))
        pc_center = 64 + rng.normal(0, 5)
        pc_width = rng.uniform(9, 16)
    else:  # clutter: weak irregular profile
        mtd_echo = rng.normal(0, 0.5, size=pulse_count)
        pc_center = rng.uniform(20, 108)
        pc_width = rng.uniform(18, 35)

    mtd_echo = mtd_echo + rng.normal(0, 0.25, size=pulse_count)
    x = np.arange(129, dtype=np.float32)
    pc_echo = np.exp(-0.5 * ((x - pc_center) / pc_width) ** 2)
    pc_echo = pc_echo + rng.normal(0, 0.08, size=129)
    prt_us = 1.0e6 / max(prf, 1.0)
    return mtd_echo.astype(np.float32), pc_echo.astype(np.float32), pw_us, prt_us, pulse_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="radar_three_cls/example_data.npz")
    parser.add_argument("--samples-per-class", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    mtd_echoes, pc_echoes, pw_us, prt_us, prt_nbr, labels = [], [], [], [], [], []
    for label in range(3):
        for _ in range(args.samples_per_class):
            mtd_echo, pc_echo, pw, prt, pulses = make_sample(label, rng)
            mtd_echoes.append(mtd_echo)
            pc_echoes.append(pc_echo)
            pw_us.append(pw)
            prt_us.append(prt)
            prt_nbr.append(pulses)
            labels.append(label)
    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        mtd_echo=np.asarray(mtd_echoes, dtype=object),
        pc_echo=np.asarray(pc_echoes, dtype=np.float32),
        pw_us=np.asarray(pw_us, dtype=np.float32),
        prt_us=np.asarray(prt_us, dtype=np.float32),
        prt_nbr=np.asarray(prt_nbr, dtype=np.float32),
        pluse_band=np.full(len(labels), 2.0, dtype=np.float32),
        sample_freq=np.full(len(labels), 10.0, dtype=np.float32),
        label=np.asarray(labels, dtype=np.int64),
    )
    print(out)


if __name__ == "__main__":
    main()
