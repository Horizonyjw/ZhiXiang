"""v0.1 历史评测（已弃用）。

请使用 `evaluate.metrics_v0.2.py`（经 `from evaluate import evaluate`）：灰度 [0,1] 上计算 MAE/MSE，
CSI/POD/FAR 暂不计算。本文件保留 20 dBZ 阈值实现，仅供对照，不再作为官方入口。
"""

import csv
import json
from pathlib import Path

import numpy as np


def evaluate(prediction, target, threshold_dbz=20.0):
    """输入 [B, Tout, C, H, W] 的预测结果和真实结果，返回评测指标。"""
    # 1. 接收并转成数组：prediction 是预测结果，target 是真实结果。
    prediction = np.asarray(prediction)
    target = np.asarray(target)

    # 2. 先检查二者维度是否一致；不一致时停止计算并报错。
    if prediction.shape != target.shape:
        raise ValueError(f"预测和真值维度不一致：{prediction.shape} != {target.shape}")
    if prediction.ndim != 5:
        raise ValueError("输入必须是 [B, Tout, C, H, W] 五维数组")

    # 3. 总体指标：把所有预测步长放在一起计算一次。
    overall = _calculate(prediction, target, threshold_dbz)

    # 4. 逐预测步长指标：未来第 1 帧、第 2 帧……分别计算。
    per_lead_time = []
    for step in range(prediction.shape[1]):
        per_lead_time.append(
            {"step": step + 1, **_calculate(prediction[:, step], target[:, step], threshold_dbz)}
        )

    return {"threshold_dbz": threshold_dbz, "overall": overall, "per_lead_time": per_lead_time}


def save_metrics(metrics, output_path):
    """按文件扩展名保存为 JSON 或 CSV。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".json":
        # 5a. 文件名以 .json 结尾时，保存为 JSON。
        output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    if output_path.suffix.lower() == ".csv":
        # 5b. 文件名以 .csv 结尾时，保存为 CSV 表格。
        rows = [{"scope": "overall", "step": "all", **metrics["overall"]}]
        rows.extend({"scope": "lead_time", **row} for row in metrics["per_lead_time"])
        with output_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        return
    raise ValueError("输出文件必须以 .json 或 .csv 结尾")


def _calculate(prediction, target, threshold_dbz):
    # 只保留预测值和真值都存在的格点。
    valid = np.isfinite(prediction) & np.isfinite(target)
    prediction = prediction[valid]
    target = target[valid]
    if prediction.size == 0:
        return {"mae": None, "mse": None, "csi": None, "pod": None, "far": None}

    # 计算 MAE、MSE 所需的预测误差。
    error = prediction - target

    # 按 20.0 dBZ 阈值统计命中、漏报和空报。
    hit = np.sum((prediction >= threshold_dbz) & (target >= threshold_dbz))
    miss = np.sum((prediction < threshold_dbz) & (target >= threshold_dbz))
    false_alarm = np.sum((prediction >= threshold_dbz) & (target < threshold_dbz))

    return {
        "mae": float(np.mean(np.abs(error))),
        "mse": float(np.mean(error**2)),
        "csi": _divide(hit, hit + miss + false_alarm),
        "pod": _divide(hit, hit + miss),
        "far": _divide(false_alarm, hit + false_alarm),
    }


def _divide(numerator, denominator):
    return None if denominator == 0 else float(numerator / denominator)

