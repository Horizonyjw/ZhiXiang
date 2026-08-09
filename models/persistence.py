"""
Persistence 基线：将输入序列最后一帧复制为未来各个预测时刻。

公共接口：
    x: [B, Tin, C, H, W] -> y: [B, Tout, C, H, W]
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Persistence(nn.Module):
    """无参数 Persistence 模型。"""

    def __init__(self, tin: int, tout: int, in_channels: int = 1) -> None:
        super().__init__()
        if tin < 1 or tout < 1:
            raise ValueError("tin / tout 必须 >= 1")
        if in_channels < 1:
            raise ValueError("in_channels 必须 >= 1")

        self.tin = tin
        self.tout = tout
        self.in_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"期望输入为 5D [B,Tin,C,H,W]，实际 ndim={x.ndim}")
        b, tin, c, _h, _w = x.shape
        if tin != self.tin:
            raise ValueError(f"Tin 不匹配：期望 {self.tin}，实际 {tin}")
        if c != self.in_channels:
            raise ValueError(f"C 不匹配：期望 {self.in_channels}，实际 {c}")

        last = x[:, -1:, :, :, :]  # [B, 1, C, H, W]
        return last.expand(b, self.tout, c, x.size(-2), x.size(-1)).contiguous()
