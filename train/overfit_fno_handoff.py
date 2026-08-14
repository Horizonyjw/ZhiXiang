"""
使用 radar_v1 handoff_samples.npz 做 FNO 小样本过拟合。

对齐 wjh: train/overfit_unet_handoff.py
数据使用仓库内 radar_v1 handoff（标准化真实样本），不依赖本机原始 PNG 路径。

用法（仓库根目录）:
  python -m train.overfit_fno_handoff
  python -m train.overfit_fno_handoff --config configs/20260814-fno-radar_v1-overfit-01.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import RadarFNO
from train.config_utils import load_config, repo_root, resolve_device
from train.export_predictions import save_predictions_npz


def load_handoff_tensors(path: Path, num_samples: int, seed: int):
    data = np.load(path, allow_pickle=False)
    x = np.asarray(data["inputs"], dtype=np.float32)
    y = np.asarray(data["targets"], dtype=np.float32)
    n = min(num_samples, x.shape[0])
    rng = np.random.default_rng(seed)
    idx = np.arange(x.shape[0])
    rng.shuffle(idx)
    idx = np.sort(idx[:n])
    meta = None
    if "metadata_json" in data.files:
        raw = data["metadata_json"]
        meta = [str(raw[i]) for i in idx]
    return (
        torch.from_numpy(x[idx]),
        torch.from_numpy(y[idx]),
        meta,
        idx.tolist(),
    )


def train(cfg: dict) -> Path:
    device = resolve_device(cfg.get("device", "auto"))
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    handoff_path = Path(cfg.get("handoff_npz", "radar_v1/04_handoff/handoff_samples.npz"))
    if not handoff_path.is_absolute():
        handoff_path = repo_root() / handoff_path

    xs, ys, meta_list, sample_idx = load_handoff_tensors(
        handoff_path,
        num_samples=int(cfg.get("num_samples", 16)),
        seed=seed,
    )
    print("overfit samples:", tuple(xs.shape), "->", tuple(ys.shape), "idx:", sample_idx)
    print("source handoff:", handoff_path)

    out_dir = repo_root() / cfg.get("output_dir", f"results/{cfg.get('experiment_id')}")
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(
        TensorDataset(xs, ys),
        batch_size=int(cfg.get("batch_size", 1)),
        shuffle=True,
    )

    model = RadarFNO(
        tin=int(cfg["tin"]),
        tout=int(cfg["tout"]),
        in_channels=int(cfg["in_channels"]),
        width=int(cfg.get("fno_width", 32)),
        modes1=int(cfg.get("modes1", 16)),
        modes2=int(cfg.get("modes2", 16)),
        n_layers=int(cfg.get("n_layers", 4)),
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=float(cfg.get("learning_rate", 1e-3)))
    criterion = nn.MSELoss()
    epochs = int(cfg.get("epochs", 40))

    history = []
    model.train()
    for epoch in range(1, epochs + 1):
        total = 0.0
        n_batches = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            total += float(loss.item())
            n_batches += 1
        mean_loss = total / max(n_batches, 1)
        history.append({"epoch": epoch, "mse": mean_loss})
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"epoch {epoch:03d} | mse={mean_loss:.6f}")

    csv_path = out_dir / "train_loss.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "mse"])
        writer.writeheader()
        writer.writerows(history)

    ckpt_path = ckpt_dir / "fno_overfit.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {k: v for k, v in cfg.items() if not str(k).startswith("_")},
            "history": history,
            "sample_indices": sample_idx,
        },
        ckpt_path,
    )

    model.eval()
    with torch.no_grad():
        pred_all = model(xs.to(device)).cpu()

    # handoff 灰度 MAE/MSE（评测 v0.2 口径）
    err = pred_all.numpy().astype(np.float64) - ys.numpy().astype(np.float64)
    metrics = {
        "evaluation_version": "v0.2-local",
        "data_scale": "normalized_grayscale_[0,1]",
        "data_version": cfg.get("data_version", "radar_v1"),
        "model_name": "fno",
        "stage": "handoff_overfit",
        "n_samples": int(xs.shape[0]),
        "overall": {
            "mae": float(np.mean(np.abs(err))),
            "mse": float(np.mean(err**2)),
        },
        "per_lead_time": [
            {
                "step": i + 1,
                "lead_min": (i + 1) * 6,
                "mae": float(np.mean(np.abs(err[:, i]))),
                "mse": float(np.mean(err[:, i] ** 2)),
            }
            for i in range(err.shape[1])
        ],
        "classification_metrics_status": "not_computed",
    }
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    meta = {
        "experiment_id": cfg.get("experiment_id"),
        "data_version": cfg.get("data_version", "radar_v1"),
        "stage": "handoff_overfit",
        "physical_unit": cfg.get("physical_unit", "normalized_grayscale_proxy_NOT_dBZ"),
        "sample_indices": sample_idx,
        "sample_metadata": meta_list,
        "note": "trained on radar_v1 handoff real samples; eval v0.2 MAE/MSE only",
    }
    pred_path = save_predictions_npz(
        out_dir / "predictions.npz",
        xs,
        ys,
        pred_all,
        metadata=meta,
    )

    summary = {
        "experiment_id": cfg.get("experiment_id"),
        "data_version": cfg.get("data_version"),
        "device": device,
        "num_samples": int(xs.shape[0]),
        "sample_indices": sample_idx,
        "epochs": epochs,
        "first_mse": history[0]["mse"],
        "last_mse": history[-1]["mse"],
        "loss_decreased": history[-1]["mse"] < history[0]["mse"],
        "overall_mae": metrics["overall"]["mae"],
        "overall_mse": metrics["overall"]["mse"],
        "checkpoint": str(ckpt_path.relative_to(repo_root())).replace("\\", "/"),
        "loss_csv": str(csv_path.relative_to(repo_root())).replace("\\", "/"),
        "predictions": str(pred_path.relative_to(repo_root())).replace("\\", "/"),
        "metrics": str(metrics_path.relative_to(repo_root())).replace("\\", "/"),
        "physical_unit": cfg.get("physical_unit"),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # 保存配置副本
    src_cfg = cfg.get("_config_path")
    if src_cfg:
        (out_dir / "config.yaml").write_text(Path(src_cfg).read_text(encoding="utf-8"), encoding="utf-8")

    print("summary:", json.dumps(summary, ensure_ascii=False))
    if not summary["loss_decreased"]:
        raise RuntimeError("损失未下降，请检查学习率/模型输出/数据尺度")
    print("fno handoff overfit OK")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="radar_v1 handoff FNO 过拟合")
    parser.add_argument("--config", default="configs/20260814-fno-radar_v1-overfit-01.yaml")
    args = parser.parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = repo_root() / cfg_path
    cfg = load_config(cfg_path)
    cfg["_config_path"] = str(cfg_path)
    train(cfg)


if __name__ == "__main__":
    main()
