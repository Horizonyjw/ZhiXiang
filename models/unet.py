"""
经典 2D U-Net 基线（雷达回波短临预测）。

设计要点：
- 公共接口保持 5D：[B, Tin, C, H, W] -> [B, Tout, C, H, W]
- 模型内部将时间维拼到通道维，再送入 2D U-Net
- 输出为线性回归头，不做 softmax/sigmoid
- 结构参考 Ronneberger et al. U-Net；实现为项目内自包含代码

空间尺寸约束：H、W 需能被 2^depth 整除（默认 depth=4，即需整除 16）。
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv => BN => ReLU) * 2"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # 处理奇数尺寸导致的 1 像素偏差
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        if dy != 0 or dx != 0:
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class ClassicUNet2D(nn.Module):
    """标准 2D U-Net：输入/输出均为 [B, Channels, H, W]。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 32,
        depth: int = 4,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth 必须 >= 1")

        self.depth = depth
        chs: List[int] = [base_channels * (2**i) for i in range(depth)]

        self.inc = DoubleConv(in_channels, chs[0])
        self.downs = nn.ModuleList(
            [Down(chs[i], chs[i + 1]) for i in range(depth - 1)]
        )
        self.ups = nn.ModuleList(
            [Up(chs[i], chs[i - 1]) for i in range(depth - 1, 0, -1)]
        )
        self.outc = nn.Conv2d(chs[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: List[torch.Tensor] = []
        x = self.inc(x)
        skips.append(x)
        for down in self.downs:
            x = down(x)
            skips.append(x)

        # skips: [level0, level1, ..., bottleneck]；上采样从瓶颈前一层开始用
        x = skips[-1]
        skip_idx = len(skips) - 2
        for up in self.ups:
            x = up(x, skips[skip_idx])
            skip_idx -= 1
        return self.outc(x)


class RadarUNet(nn.Module):
    """
    雷达短临预测 U-Net 包装器。

    外部接口：
        x: [B, Tin, C, H, W] -> y: [B, Tout, C, H, W]
    """

    def __init__(
        self,
        tin: int,
        tout: int,
        in_channels: int = 1,
        base_channels: int = 32,
        depth: int = 4,
    ) -> None:
        super().__init__()
        if tin < 1 or tout < 1:
            raise ValueError("tin / tout 必须 >= 1")
        if in_channels < 1:
            raise ValueError("in_channels 必须 >= 1")

        self.tin = tin
        self.tout = tout
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.depth = depth

        self.backbone = ClassicUNet2D(
            in_channels=tin * in_channels,
            out_channels=tout * in_channels,
            base_channels=base_channels,
            depth=depth,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"期望输入为 5D [B,Tin,C,H,W]，实际 ndim={x.ndim}")
        b, tin, c, h, w = x.shape
        if tin != self.tin:
            raise ValueError(f"Tin 不匹配：期望 {self.tin}，实际 {tin}")
        if c != self.in_channels:
            raise ValueError(f"C 不匹配：期望 {self.in_channels}，实际 {c}")

        # [B, Tin, C, H, W] -> [B, Tin*C, H, W]
        x2d = x.reshape(b, tin * c, h, w)
        y2d = self.backbone(x2d)
        # [B, Tout*C, H, W] -> [B, Tout, C, H, W]
        return y2d.reshape(b, self.tout, c, h, w)
