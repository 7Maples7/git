# -*- coding: utf-8 -*-
"""Three-branch PyTorch model for DClsEcho radar target classification."""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from .dataset import META_DIM
except ImportError:
    from dataset import META_DIM  # type: ignore


class Conv1dEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 32) -> None:
        super().__init__()

        def gn(channels: int) -> nn.GroupNorm:
            groups = 8 if channels % 8 == 0 else 1
            return nn.GroupNorm(groups, channels)

        self.net = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=5, padding=2),
            gn(base_channels),
            nn.SiLU(inplace=True),
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=5, stride=2, padding=2),
            gn(base_channels * 2),
            nn.SiLU(inplace=True),
            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            gn(base_channels * 4),
            nn.SiLU(inplace=True),
        )
        self.out_dim = base_channels * 4 * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.net(x)
        avg = feat.mean(dim=-1)
        mx = feat.amax(dim=-1)
        return torch.cat([avg, mx], dim=1)


class RadarThreeBranchNet(nn.Module):
    """Fuse MTD Doppler, PC slow-time, PC spectrum, and meta features."""

    def __init__(self, num_classes: int = 3, meta_dim: int = META_DIM, dropout: float = 0.2) -> None:
        super().__init__()
        self.mtd_echo_branch = Conv1dEncoder(1, 32)
        self.pc_echo_branch = Conv1dEncoder(1, 32)
        self.pc_spectrum_branch = Conv1dEncoder(1, 32)
        self.meta_branch = nn.Sequential(
            nn.Linear(meta_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.SiLU(inplace=True),
        )
        fused_dim = (
            self.mtd_echo_branch.out_dim
            + self.pc_echo_branch.out_dim
            + self.pc_spectrum_branch.out_dim
            + 64
        )
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        mtd_echo = batch["mtd_echo"]
        if "mtd_mask" in batch:
            mtd_echo = mtd_echo * batch["mtd_mask"].to(mtd_echo.dtype)
        pc_echo = batch["pc_echo"]
        if "pc_mask" in batch:
            pc_echo = pc_echo * batch["pc_mask"].to(pc_echo.dtype)
        pc_spectrum = batch["pc_spectrum"]
        if "pc_spectrum_mask" in batch:
            pc_spectrum = pc_spectrum * batch["pc_spectrum_mask"].to(pc_spectrum.dtype)

        raw_z = self.mtd_echo_branch(mtd_echo.unsqueeze(1))
        pc_z = self.pc_echo_branch(pc_echo.unsqueeze(1))
        pc_fft_z = self.pc_spectrum_branch(pc_spectrum.unsqueeze(1))
        meta_z = self.meta_branch(batch["meta"])
        return self.head(torch.cat([raw_z, pc_z, pc_fft_z, meta_z], dim=1))
