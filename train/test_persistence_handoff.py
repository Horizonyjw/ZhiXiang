"""
读取 radar_v1 handoff_samples.npz 并做 Persistence 真实前向测试。

对齐 wjh: train/test_handoff_forward.py（U-Net 版）

用法（仓库根目录）:
  python -m train.test_persistence_handoff
  python -m train.test_persistence_handoff --config configs/20260814-persistence-radar_v1-01.yaml
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

from models import Persistence
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
    assert float(x.min()) >= 0.0 and float(x.max()) <= 1.0
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0

    model = Persistence(tin=tin, tout=tout, in_channels=c).to(device)
    model.eval()

    # 默认用全部 handoff 样本导出，便于评测侧直接读 predictions.npz
    xb = torch.from_numpy(x).to(device)
    yb = torch.from_numpy(y)
    with torch.no_grad():
        pred = model(xb)

    expect_pred = (x.shape[0], tout, c, h, w)
    print("device:", device)
    print("pred :", tuple(pred.shape), pred.dtype)
    assert tuple(pred.shape) == expect_pred, f"输出维度错误：{tuple(pred.shape)} != {expect_pred}"
    assert torch.isfinite(pred).all(), "预测存在 NaN/Inf"

    out_dir = repo_root() / cfg.get("output_dir", f"results/{cfg.get('experiment_id')}")
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "experiment_id": cfg.get("experiment_id"),
        "model_name": "persistence",
        "data_version": cfg.get("data_version", "radar_v1"),
        "stage": "handoff_forward",
        "physical_unit": cfg.get("physical_unit", "normalized_grayscale_proxy_NOT_dBZ"),
        "handoff": str(handoff_path.relative_to(repo_root())).replace("\\", "/"),
        "n_used": int(x.shape[0]),
        "note": "eval v0.2: MAE/MSE on [0,1] grayscale; CSI/POD/FAR not computed",
    }
    if "metadata_json" in payload:
        raw = payload["metadata_json"]
        try:
            meta["sample_metadata"] = [str(raw[i]) for i in range(int(x.shape[0]))]
        except Exception:
            pass

    pred_path = save_predictions_npz(
        out_dir / "predictions.npz",
        xb.cpu(),
        yb,
        pred.cpu(),
        metadata=meta,
    )

    # 同步保存配置副本便于追溯
    cfg_out = out_dir / "config.yaml"
    src_cfg = cfg.get("_config_path")
    if src_cfg:
        cfg_out.write_text(Path(src_cfg).read_text(encoding="utf-8"), encoding="utf-8")

    summary = {
        "experiment_id": cfg.get("experiment_id"),
        "data_version": cfg.get("data_version"),
        "device": device,
        "inputs_shape": list(x.shape),
        "targets_shape": list(y.shape),
        "pred_shape": list(expect_pred),
        "forward_ok": True,
        "predictions": str(pred_path.relative_to(repo_root())).replace("\\", "/"),
        "physical_unit": cfg.get("physical_unit"),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved:", pred_path)
    print("summary:", summary_path)
    print("persistence handoff forward OK")
    return pred_path


def main() -> None:
    parser = argparse.ArgumentParser(description="radar_v1 handoff Persistence 前向测试")
    parser.add_argument(
        "--config",
        default="configs/20260814-persistence-radar_v1-01.yaml",
    )
    args = parser.parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = repo_root() / cfg_path
    cfg = load_config(cfg_path)
    cfg["_config_path"] = str(cfg_path)
    run(cfg)


if __name__ == "__main__":
    main()
