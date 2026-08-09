"""
轻量 2D FNO 适配（雷达回波短临预测接口测试用）。

说明：
- 仓库内未找到外部 PDE/FNO/autoresearch 工程入口，故提供项目内自包含实现；
- 外部保持统一 5D 接口：[B, Tin, C, H, W] -> [B, Tout, C, H, W]；
- 内部将时间维拼到通道维后走 2D 频谱卷积；
- 本文件用于随机张量前向与接口验证，不代表已完成正式训练。

结构参考 Li et al. Fourier Neural Operator；实现为项目内简化版。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SpectralConv2d(nn.Module):
    """2D 傅里叶层：低频模态可学习复数权重。"""

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int) -> None:
        super().__init__()
        if modes1 < 1 or modes2 < 1:
            raise ValueError("modes1 / modes2 必须 >= 1")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input_ft: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # input_ft: [B, Cin, H, W_half]
        # weights:  [Cin, Cout, modes1, modes2]
        return torch.einsum("bixy,ioxy->boxy", input_ft, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _c, h, w = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")

        out_ft = torch.zeros(
            b,
            self.out_channels,
            h,
            w // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        m1 = min(self.modes1, h)
        m2 = min(self.modes2, w // 2 + 1)

        out_ft[:, :, :m1, :m2] = self.compl_mul2d(
            x_ft[:, :, :m1, :m2],
            self.weights1[:, :, :m1, :m2],
        )
        out_ft[:, :, -m1:, :m2] = self.compl_mul2d(
            x_ft[:, :, -m1:, :m2],
            self.weights2[:, :, :m1, :m2],
        )
        return torch.fft.irfft2(out_ft, s=(h, w), norm="ortho")


class FNOBlock2d(nn.Module):
    def __init__(self, width: int, modes1: int, modes2: int) -> None:
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes1, modes2)
        self.w = nn.Conv2d(width, width, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(self.spectral(x) + self.w(x))


class FNO2d(nn.Module):
    """标准通道式 FNO2d：输入/输出均为 [B, Channels, H, W]。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        width: int = 32,
        modes1: int = 12,
        modes2: int = 12,
        n_layers: int = 4,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers 必须 >= 1")

        self.lift = nn.Conv2d(in_channels, width, kernel_size=1)
        self.blocks = nn.ModuleList(
            [FNOBlock2d(width, modes1, modes2) for _ in range(n_layers)]
        )
        self.proj = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(x)
        for block in self.blocks:
            x = block(x)
        return self.proj(x)


class RadarFNO(nn.Module):
    """
    雷达短临预测 FNO 包装器。

    外部接口：
        x: [B, Tin, C, H, W] -> y: [B, Tout, C, H, W]
    """

    def __init__(
        self,
        tin: int,
        tout: int,
        in_channels: int = 1,
        width: int = 32,
        modes1: int = 12,
        modes2: int = 12,
        n_layers: int = 4,
    ) -> None:
        super().__init__()
        if tin < 1 or tout < 1:
            raise ValueError("tin / tout 必须 >= 1")
        if in_channels < 1:
            raise ValueError("in_channels 必须 >= 1")

        self.tin = tin
        self.tout = tout
        self.in_channels = in_channels
        self.width = width
        self.modes1 = modes1
        self.modes2 = modes2
        self.n_layers = n_layers

        self.backbone = FNO2d(
            in_channels=tin * in_channels,
            out_channels=tout * in_channels,
            width=width,
            modes1=modes1,
            modes2=modes2,
            n_layers=n_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"期望输入为 5D [B,Tin,C,H,W]，实际 ndim={x.ndim}")
        b, tin, c, h, w = x.shape
        if tin != self.tin:
            raise ValueError(f"Tin 不匹配：期望 {self.tin}，实际 {tin}")
        if c != self.in_channels:
            raise ValueError(f"C 不匹配：期望 {self.in_channels}，实际 {c}")

        x2d = x.reshape(b, tin * c, h, w)
        y2d = self.backbone(x2d)
        return y2d.reshape(b, self.tout, c, h, w)
