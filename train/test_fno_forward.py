"""
FNO 随机张量前向测试。

用法（仓库根目录）:
  python -m train.test_fno_forward
  python -m train.test_fno_forward --config configs/fno_baseline.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import RadarFNO
from train.config_utils import load_config, resolve_device


def run_forward(cfg: dict) -> None:
    device = resolve_device(cfg.get("device", "auto"))
    tin = int(cfg["tin"])
    tout = int(cfg["tout"])
    c = int(cfg["in_channels"])
    h = int(cfg["height"])
    w = int(cfg["width"])
    fno_width = int(cfg.get("fno_width", cfg.get("hidden_width", 32)))
    modes1 = int(cfg.get("modes1", 12))
    modes2 = int(cfg.get("modes2", 12))
    n_layers = int(cfg.get("n_layers", 4))
    seed = int(cfg.get("seed", 42))

    torch.manual_seed(seed)
    model = RadarFNO(
        tin=tin,
        tout=tout,
        in_channels=c,
        width=fno_width,
        modes1=modes1,
        modes2=modes2,
        n_layers=n_layers,
    ).to(device)
    model.eval()

    x = torch.randn(2, tin, c, h, w, dtype=torch.float32, device=device)
    with torch.no_grad():
        y = model(x)

    expect = (2, tout, c, h, w)
    n_params = sum(p.numel() for p in model.parameters())
    print("device:", device)
    print("params:", n_params)
    print("input :", tuple(x.shape), x.dtype)
    print("output:", tuple(y.shape), y.dtype)
    print("expect:", expect)
    assert tuple(y.shape) == expect, f"输出维度错误：{tuple(y.shape)} != {expect}"
    assert torch.isfinite(y).all(), "输出存在 NaN/Inf"
    print("forward test OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="FNO 随机张量前向测试")
    parser.add_argument(
        "--config",
        default="configs/fno_baseline.yaml",
        help="配置文件路径（相对仓库根目录）",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_forward(cfg)


if __name__ == "__main__":
    main()
