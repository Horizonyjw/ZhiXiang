"""
随机张量前向测试（任务 A5）。

用法（在仓库根目录）:
  .\\.venv\\Scripts\\python.exe -m train.test_unet_forward
  .\\.venv\\Scripts\\python.exe -m train.test_unet_forward --config configs/unet_baseline.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# 允许直接 python train/test_unet_forward.py 运行
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import RadarUNet
from train.config_utils import load_config, resolve_device


def run_forward(cfg: dict) -> None:
    device = resolve_device(cfg.get("device", "auto"))
    tin = int(cfg["tin"])
    tout = int(cfg["tout"])
    c = int(cfg["in_channels"])
    h = int(cfg["height"])
    w = int(cfg["width"])
    base = int(cfg.get("base_channels", 32))
    depth = int(cfg.get("depth", 4))
    seed = int(cfg.get("seed", 42))

    factor = 2 ** depth
    if h % factor != 0 or w % factor != 0:
        raise ValueError(f"H/W 需能被 {factor} 整除，当前 H={h}, W={w}")

    torch.manual_seed(seed)
    model = RadarUNet(
        tin=tin,
        tout=tout,
        in_channels=c,
        base_channels=base,
        depth=depth,
    ).to(device)
    model.eval()

    x = torch.randn(2, tin, c, h, w, dtype=torch.float32, device=device)
    with torch.no_grad():
        y = model(x)

    expect = (2, tout, c, h, w)
    print("device:", device)
    print("input :", tuple(x.shape), x.dtype)
    print("output:", tuple(y.shape), y.dtype)
    print("expect:", expect)
    assert tuple(y.shape) == expect, f"输出维度错误：{tuple(y.shape)} != {expect}"
    assert torch.isfinite(y).all(), "输出存在 NaN/Inf"
    print("forward test OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="U-Net 随机张量前向测试")
    parser.add_argument(
        "--config",
        default="configs/unet_baseline.yaml",
        help="配置文件路径（相对仓库根目录）",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_forward(cfg)


if __name__ == "__main__":
    main()
