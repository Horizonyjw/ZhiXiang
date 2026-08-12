"""
将预测结果导出为评测可读的 npz（任务 B3 预留格式）。

约定字段（与 evaluate.run_from_npz 对齐）:
  inputs, targets, predictions  — float32 数组，形状 [N, T, C, H, W]
  metadata_json                 — UTF-8 JSON 字符串（可选）

评测读取：
  .\\.venv\\Scripts\\python.exe -m evaluate.run_from_npz --npz <path> --plot
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    return arr


def save_predictions_npz(
    path: str | Path,
    inputs: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
    predictions: torch.Tensor | np.ndarray,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    保存到 results/<experiment_id>/predictions.npz 一类路径。

    期望形状：
      inputs/targets/predictions: [N, T, C, H, W]
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "inputs": _to_numpy(inputs),
        "targets": _to_numpy(targets),
        "predictions": _to_numpy(predictions),
    }
    for key in ("inputs", "targets", "predictions"):
        if payload[key].ndim != 5:
            raise ValueError(f"{key} 应为 5D [N,T,C,H,W]，实际 {payload[key].shape}")

    if metadata is not None:
        payload["metadata_json"] = np.asarray(json.dumps(metadata, ensure_ascii=False))

    np.savez_compressed(out, **payload)
    return out
