"""
人工数据小样本过拟合脚手架（任务 A6）。

在真实 DataLoader 到位前，用固定随机样本检查训练入口是否能使损失下降。
真实数据联调时，只需替换数据构造部分，保持保存格式不变。

用法（在仓库根目录）:
  .\\.venv\\Scripts\\python.exe -m train.overfit_unet_toy
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import RadarUNet
from train.config_utils import load_config, repo_root, resolve_device


def make_toy_batch(cfg: dict, device: str) -> TensorDataset:
    """构造固定的人工样本：输入为平滑随机场，目标为输入最后一帧的平移近似。"""
    seed = int(cfg.get("seed", 42))
    n = int(cfg.get("num_samples", 12))
    tin = int(cfg["tin"])
    tout = int(cfg["tout"])
    c = int(cfg["in_channels"])
    h = int(cfg["height"])
    w = int(cfg["width"])

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    # 低频随机场，比纯白噪声更容易过拟合观察
    noise = torch.randn(n, tin + tout, c, h // 4, w // 4, generator=g)
    fields = torch.nn.functional.interpolate(
        noise.reshape(n * (tin + tout), c, h // 4, w // 4),
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    ).reshape(n, tin + tout, c, h, w)

    inputs = fields[:, :tin]
    # 目标：将末帧向右下轻微平移后复制到未来帧，形成可学模式
    last = inputs[:, -1]
    shifted = torch.roll(last, shifts=(2, 2), dims=(-2, -1))
    targets = shifted.unsqueeze(1).repeat(1, tout, 1, 1, 1)
    targets = targets + 0.05 * torch.randn_like(targets)

    return TensorDataset(inputs.contiguous(), targets.contiguous())


def train(cfg: dict) -> Path:
    device = resolve_device(cfg.get("device", "auto"))
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    out_dir = repo_root() / cfg.get(
        "output_dir", f"results/{cfg.get('experiment_id', 'unet_toy')}"
    )
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    dataset = make_toy_batch(cfg, device)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.get("batch_size", 4)),
        shuffle=True,
        drop_last=False,
    )

    model = RadarUNet(
        tin=int(cfg["tin"]),
        tout=int(cfg["tout"]),
        in_channels=int(cfg["in_channels"]),
        base_channels=int(cfg.get("base_channels", 32)),
        depth=int(cfg.get("depth", 4)),
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=float(cfg.get("learning_rate", 1e-3)))
    criterion = nn.MSELoss()
    epochs = int(cfg.get("epochs", 50))

    history = []
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        n_batches = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            total_loss += float(loss.item())
            n_batches += 1
        mean_loss = total_loss / max(n_batches, 1)
        history.append({"epoch": epoch, "mse": mean_loss})
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"epoch {epoch:03d} | mse={mean_loss:.6f}")

    # 保存曲线
    csv_path = out_dir / "train_loss.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "mse"])
        writer.writeheader()
        writer.writerows(history)

    # 保存权重与一次预测样例
    ckpt_path = ckpt_dir / "unet_toy_last.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg,
            "history": history,
        },
        ckpt_path,
    )

    model.eval()
    with torch.no_grad():
        x0, y0 = dataset[0]
        x0 = x0.unsqueeze(0).to(device)
        y0 = y0.unsqueeze(0).to(device)
        pred0 = model(x0)
        pred_path = out_dir / "predictions_toy.pt"
        torch.save(
            {
                "inputs": x0.cpu(),
                "targets": y0.cpu(),
                "predictions": pred0.cpu(),
            },
            pred_path,
        )

    summary = {
        "experiment_id": cfg.get("experiment_id"),
        "device": device,
        "epochs": epochs,
        "num_samples": int(cfg.get("num_samples", 12)),
        "first_mse": history[0]["mse"],
        "last_mse": history[-1]["mse"],
        "loss_decreased": history[-1]["mse"] < history[0]["mse"],
        "checkpoint": str(ckpt_path.relative_to(repo_root())),
        "loss_csv": str(csv_path.relative_to(repo_root())),
        "note": "人工数据脚手架结果，不能作为正式对比实验",
    }
    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("summary:", json.dumps(summary, ensure_ascii=False))
    if not summary["loss_decreased"]:
        raise RuntimeError("损失未下降，请检查训练入口 / 学习率 / 模型输出")
    print("overfit toy scaffold OK")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="U-Net 人工数据过拟合脚手架")
    parser.add_argument("--config", default="configs/unet_baseline.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
