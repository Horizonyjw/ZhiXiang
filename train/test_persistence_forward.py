"""
Persistence 随机张量前向测试。

用法（仓库根目录）:
  python -m train.test_persistence_forward
  python -m train.test_persistence_forward --config configs/persistence_baseline.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import Persistence
from train.config_utils import load_config, resolve_device


def run_forward(cfg: dict) -> None:
    device = resolve_device(cfg.get("device", "auto"))
    tin = int(cfg["tin"])
    tout = int(cfg["tout"])
    c = int(cfg["in_channels"])
    h = int(cfg["height"])
    w = int(cfg["width"])
    seed = int(cfg.get("seed", 42))
    b = int(cfg.get("batch_size", 2))

    torch.manual_seed(seed)
    model = Persistence(tin=tin, tout=tout, in_channels=c).to(device)
    model.eval()

    x = torch.randn(b, tin, c, h, w, dtype=torch.float32, device=device)
    with torch.no_grad():
        y = model(x)

    expect = (b, tout, c, h, w)
    print("device:", device)
    print("input :", tuple(x.shape), x.dtype)
    print("output:", tuple(y.shape), y.dtype)
    print("expect:", expect)
    assert tuple(y.shape) == expect, f"输出维度错误：{tuple(y.shape)} != {expect}"
    assert torch.isfinite(y).all(), "输出存在 NaN/Inf"

    # Persistence 语义：每一未来帧都应等于输入最后一帧
    last = x[:, -1]
    for t in range(tout):
        assert torch.allclose(y[:, t], last), f"第 {t} 帧未复制最后一帧"
    print("forward test OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="Persistence 随机张量前向测试")
    parser.add_argument(
        "--config",
        default="configs/persistence_baseline.yaml",
        help="配置文件路径（相对仓库根目录）",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_forward(cfg)


if __name__ == "__main__":
    main()
