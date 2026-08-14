"""
运行 Persistence 并保存预测结果。

默认使用随机张量验证保存接口；获得真实样本后，用 --npz 指向
含 inputs/targets 的 npz（字段形状 [N,T,C,H,W]）即可替换。

用法（仓库根目录）:
  python -m train.run_persistence
  python -m train.run_persistence --npz path/to/real_batch.npz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import Persistence
from train.config_utils import load_config, repo_root, resolve_device
from train.export_predictions import save_predictions_npz


def _load_real_batch(path: Path) -> tuple[torch.Tensor, torch.Tensor, dict]:
    data = np.load(path, allow_pickle=False)
    if "inputs" not in data or "targets" not in data:
        raise ValueError(f"{path} 需包含 inputs 与 targets 字段")
    inputs = torch.from_numpy(np.asarray(data["inputs"], dtype=np.float32))
    targets = torch.from_numpy(np.asarray(data["targets"], dtype=np.float32))
    if inputs.ndim != 5 or targets.ndim != 5:
        raise ValueError("inputs/targets 应为 5D [N,T,C,H,W]")
    meta: dict = {"source_npz": str(path)}
    if "metadata_json" in data:
        meta["source_metadata_json"] = str(data["metadata_json"])
    return inputs, targets, meta


def run(cfg: dict, npz_path: Path | None) -> Path:
    device = resolve_device(cfg.get("device", "auto"))
    tin = int(cfg["tin"])
    tout = int(cfg["tout"])
    c = int(cfg["in_channels"])
    h = int(cfg["height"])
    w = int(cfg["width"])
    seed = int(cfg.get("seed", 42))
    b = int(cfg.get("batch_size", 2))
    exp_id = str(cfg.get("experiment_id", "persistence-run"))
    out_dir = Path(cfg.get("output_dir", f"results/{exp_id}"))
    if not out_dir.is_absolute():
        out_dir = repo_root() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    model = Persistence(tin=tin, tout=tout, in_channels=c).to(device)
    model.eval()

    data_mode = "real_npz" if npz_path is not None else "random_tensor"
    if npz_path is not None:
        inputs, targets, extra_meta = _load_real_batch(npz_path)
        if inputs.shape[1] != tin or targets.shape[1] != tout:
            raise ValueError(
                f"Tin/Tout 与配置不一致：inputs.T={inputs.shape[1]}, "
                f"targets.T={targets.shape[1]}, cfg tin/tout={tin}/{tout}"
            )
        if inputs.shape[2] != c or targets.shape[2] != c:
            raise ValueError("通道数 C 与配置不一致")
    else:
        torch.manual_seed(seed)
        inputs = torch.randn(b, tin, c, h, w, dtype=torch.float32)
        # 无真实未来帧时，targets 仅作接口占位，不作为评测结论
        targets = torch.randn(b, tout, c, h, w, dtype=torch.float32)
        extra_meta = {
            "warning": "随机张量占位，不是真实雷达数据，不得写入正式联调结论"
        }

    t0 = time.perf_counter()
    with torch.no_grad():
        preds = model(inputs.to(device)).cpu()
    elapsed = time.perf_counter() - t0

    metadata = {
        "experiment_id": exp_id,
        "model_name": "persistence",
        "data_mode": data_mode,
        "tin": tin,
        "tout": tout,
        "in_channels": c,
        "device": device,
        "inference_seconds": elapsed,
        "git_branch_hint": "see git rev-parse HEAD",
        **extra_meta,
    }

    pred_path = save_predictions_npz(
        out_dir / "predictions.npz",
        inputs=inputs,
        targets=targets,
        predictions=preds,
        metadata=metadata,
    )

    summary = {
        **metadata,
        "input_shape": list(inputs.shape),
        "target_shape": list(targets.shape),
        "prediction_shape": list(preds.shape),
        "predictions_path": str(pred_path.relative_to(repo_root())).replace("\\", "/"),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 复制本次使用的配置，便于追溯
    cfg_out = out_dir / "config.yaml"
    src_cfg = cfg.get("_config_path")
    if src_cfg:
        cfg_out.write_text(Path(src_cfg).read_text(encoding="utf-8"), encoding="utf-8")

    print("data_mode:", data_mode)
    print("device:", device)
    print("predictions:", pred_path)
    print("summary:", summary_path)
    print("inference_seconds:", f"{elapsed:.6f}")
    if data_mode != "real_npz":
        print("NOTE: 当前为随机张量结果，真实数据到位后请加 --npz 重跑")
    return pred_path


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 Persistence 并保存预测")
    parser.add_argument(
        "--config",
        default="configs/20260814-persistence-radar_v1-01.yaml",
        help="配置文件路径（相对仓库根目录）",
    )
    parser.add_argument(
        "--npz",
        default=None,
        help="可选：真实样本 npz，需含 inputs/targets；缺省时读配置 handoff_npz",
    )
    args = parser.parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = repo_root() / cfg_path
    cfg = load_config(cfg_path)
    cfg["_config_path"] = str(cfg_path)
    if args.npz:
        npz = Path(args.npz)
    elif cfg.get("handoff_npz"):
        npz = Path(cfg["handoff_npz"])
        if not npz.is_absolute():
            npz = repo_root() / npz
    else:
        npz = None
    run(cfg, npz)


if __name__ == "__main__":
    main()
