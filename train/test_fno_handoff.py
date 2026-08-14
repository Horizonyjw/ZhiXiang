"""
读取 radar_v1 handoff_samples.npz 并做 FNO 真实前向测试。

对齐 wjh: train/test_handoff_forward.py

用法（仓库根目录）:
  python -m train.test_fno_handoff
  python -m train.test_fno_handoff --config configs/20260814-fno-radar_v1-01.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import RadarFNO
from train.config_utils import load_config, repo_root, resolve_device
from train.export_predictions import save_predictions_npz


def load_handoff(path: Path) -> dict:
    data = np.load(path, allow_pickle=False)
    required = {"inputs", "targets"}
    missing = required - set(data.files)
    if missing:
        raise KeyError(f"handoff 缺少字段：{sorted(missing)}")
    payload = {
        "inputs": np.asarray(data["inputs"]),
        "targets": np.asarray(data["targets"]),
    }
    if "metadata_json" in data.files:
        payload["metadata_json"] = data["metadata_json"]
    return payload


def run(cfg: dict) -> Path:
    device = resolve_device(cfg.get("device", "auto"))
    tin = int(cfg["tin"])
    tout = int(cfg["tout"])
    c = int(cfg["in_channels"])
    h = int(cfg["height"])
    w = int(cfg["width"])
    fno_width = int(cfg.get("fno_width", cfg.get("hidden_width", 32)))
    modes1 = int(cfg.get("modes1", 16))
    modes2 = int(cfg.get("modes2", 16))
    n_layers = int(cfg.get("n_layers", 4))

    handoff_path = Path(cfg.get("handoff_npz", "radar_v1/04_handoff/handoff_samples.npz"))
    if not handoff_path.is_absolute():
        handoff_path = repo_root() / handoff_path
    if not handoff_path.exists():
        raise FileNotFoundError(f"找不到 handoff：{handoff_path}")

    payload = load_handoff(handoff_path)
    x = payload["inputs"]
    y = payload["targets"]

    expect_x = (x.shape[0], tin, c, h, w)
    expect_y = (y.shape[0], tout, c, h, w)
    print("handoff:", handoff_path)
    print("inputs :", x.shape, x.dtype, "min/max", float(x.min()), float(x.max()))
    print("targets:", y.shape, y.dtype, "min/max", float(y.min()), float(y.max()))
    assert x.shape == expect_x, f"inputs shape 错误：{x.shape} != {expect_x}"
    assert y.shape == expect_y, f"targets shape 错误：{y.shape} != {expect_y}"
    assert x.dtype == np.float32 and y.dtype == np.float32

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

    batch = min(2, x.shape[0])
    xb = torch.from_numpy(x[:batch]).to(device)
    with torch.no_grad():
        pred = model(xb)

    expect_pred = (batch, tout, c, h, w)
    print("device:", device)
    print("pred :", tuple(pred.shape), pred.dtype)
    assert tuple(pred.shape) == expect_pred, f"输出维度错误：{tuple(pred.shape)} != {expect_pred}"
    assert torch.isfinite(pred).all(), "预测存在 NaN/Inf"

    out_dir = repo_root() / cfg.get("output_dir", f"results/{cfg.get('experiment_id')}")
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "experiment_id": cfg.get("experiment_id"),
        "model_name": "fno",
        "data_version": cfg.get("data_version", "radar_v1"),
        "stage": "untrained_forward",
        "physical_unit": cfg.get("physical_unit", "normalized_grayscale_proxy_NOT_dBZ"),
        "handoff": str(handoff_path.relative_to(repo_root())).replace("\\", "/"),
        "n_used": batch,
        "note": "未训练权重的接口前向，不得当作正式对比结论",
    }

    pred_path = save_predictions_npz(
        out_dir / "predictions_untrained.npz",
        xb.cpu(),
        torch.from_numpy(y[:batch]),
        pred.cpu(),
        metadata=meta,
    )

    summary = {
        "experiment_id": cfg.get("experiment_id"),
        "data_version": cfg.get("data_version"),
        "device": device,
        "inputs_shape": list(x.shape),
        "targets_shape": list(y.shape),
        "pred_shape": list(expect_pred),
        "forward_ok": True,
        "predictions_untrained": str(pred_path.relative_to(repo_root())).replace("\\", "/"),
        "physical_unit": cfg.get("physical_unit"),
    }
    summary_path = out_dir / "forward_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved:", pred_path)
    print("fno handoff forward OK")
    return pred_path


def main() -> None:
    parser = argparse.ArgumentParser(description="radar_v1 handoff FNO 前向测试")
    parser.add_argument("--config", default="configs/20260814-fno-radar_v1-01.yaml")
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
