"""v0.2 统一评测：灰度 MAE、MSE；CSI、POD、FAR 暂不计算。"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


# 当前缺少完整的 RGB 颜色—回波强度等级色标，无法确定统一回波阈值。
# 因此，CSI、POD、FAR 在 v0.2 中只保留结果字段，不生成指标数值。
CLASSIFICATION_METRICS_REASON = (
    "当前缺少完整的 RGB 颜色—回波强度等级色标，无法确定统一的活动回波阈值"
)


def evaluate_file(input_path, output_path=None):
    """读取 predictions.npz 并评测其中的 predictions 和 targets。

    NPZ 至少必须包含：
      - predictions: [N, Tout, C, H, W]
      - targets: [N, Tout, C, H, W]

    output_path 可选；传入 .json 或 .csv 路径时会同时保存结果。
    """
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"找不到预测文件：{input_path}")
    if input_path.suffix.lower() != ".npz":
        raise ValueError("输入文件必须以 .npz 结尾")

    try:
        with np.load(input_path, allow_pickle=False) as data:
            missing = [key for key in ("predictions", "targets") if key not in data]
            if missing:
                raise ValueError(
                    f"NPZ 缺少必需字段：{', '.join(missing)}；"
                    f"当前字段：{', '.join(data.files)}"
                )
            prediction = data["predictions"]
            target = data["targets"]
    except (OSError, EOFError) as error:
        raise ValueError(f"无法读取 NPZ 文件：{input_path}") from error

    if not np.issubdtype(prediction.dtype, np.number):
        raise ValueError("predictions 必须是数值数组")
    if not np.issubdtype(target.dtype, np.number):
        raise ValueError("targets 必须是数值数组")

    metrics = evaluate(prediction, target)
    if output_path is not None:
        save_metrics(metrics, output_path)
    return metrics


def evaluate(prediction, target):
    """输入归一化灰度预测和真值，返回 v0.2 评测结果。"""
    # 1. 接收并转成数组：prediction 是预测结果，target 是真实结果。
    prediction = np.asarray(prediction)
    target = np.asarray(target)

    # 2. 先检查二者维度是否一致；不一致时停止计算并报错。
    if prediction.shape != target.shape:
        raise ValueError(f"预测和真值维度不一致：{prediction.shape} != {target.shape}")
    if prediction.ndim != 5:
        raise ValueError("输入必须是 [B, Tout, C, H, W] 五维数组")

    # 3. 使用 [0,1] 归一化灰度值
    # NaN、正无穷和负无穷会在指标计算时作为无效格点排除；
    # 其余有限值必须位于 [0,1] 范围内。
    _check_gray_range(prediction, "预测值")
    _check_gray_range(target, "真值")

    # 4. 总体指标：把所有预测步长放在一起计算一次。
    overall = _calculate(prediction, target)

    # 5. 逐预测步长指标：未来第 1 帧、第 2 帧……分别计算。
    per_lead_time = []
    for step in range(prediction.shape[1]):
        per_lead_time.append(
            {"step": step + 1, **_calculate(prediction[:, step], target[:, step])}
        )

    # CSI、POD、FAR 保留字段但返回 None，表示当前条件下未计算，不能按 0 解读。
    return {
        "evaluation_version": "v0.2",
        "data_scale": "normalized_grayscale_[0,1]",
        "classification_metrics_status": "not_computed",
        "classification_metrics_reason": CLASSIFICATION_METRICS_REASON,
        "overall": overall,
        "per_lead_time": per_lead_time,
    }


def save_metrics(metrics, output_path):
    """按文件扩展名保存为 JSON 或 CSV。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".json":
        # 7a. 文件名以 .json 结尾时，保存完整的分层结果和暂不计算原因。
        output_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return
    if output_path.suffix.lower() == ".csv":
        # 7b. 文件名以 .csv 结尾时，整体结果占一行，各预测步长各占一行。
        rows = [{"scope": "overall", "step": "all", **metrics["overall"]}]
        rows.extend(
            {"scope": "lead_time", **row} for row in metrics["per_lead_time"]
        )
        fieldnames = [
            "scope",
            "step",
            "mae",
            "mse",
            "csi",
            "pod",
            "far",
            "classification_metrics_status",
            "classification_metrics_reason",
        ]
        with output_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return
    raise ValueError("输出文件必须以 .json 或 .csv 结尾")


def _calculate(prediction, target):
    # 只保留预测值和真值都存在的格点
    valid = np.isfinite(prediction) & np.isfinite(target)
    prediction = prediction[valid]
    target = target[valid]

    # 没有任何有效格点时，MAE 和 MSE 无法计算，返回 None。
    # CSI、POD、FAR 同样返回 None，但原因是当前未确定统一回波阈值。
    if prediction.size == 0:
        return _empty_metrics()

    # 计算 MAE、MSE 所需的灰度预测误差
    # 转为 float64 后再计算，减少 float32 累加造成的数值误差。
    error = prediction.astype(np.float64) - target.astype(np.float64)

    return {
        "mae": float(np.mean(np.abs(error))),
        "mse": float(np.mean(error**2)),
        "csi": None,
        "pod": None,
        "far": None,
        "classification_metrics_status": "not_computed",
        "classification_metrics_reason": CLASSIFICATION_METRICS_REASON,
    }


def _empty_metrics():
    """返回没有有效连续指标、且分类指标暂不计算时的统一空结果。"""
    return {
        "mae": None,
        "mse": None,
        "csi": None,
        "pod": None,
        "far": None,
        "classification_metrics_status": "not_computed",
        "classification_metrics_reason": CLASSIFICATION_METRICS_REASON,
    }


def _check_gray_range(values, name):
    """检查所有有限灰度值是否都位于 [0,1]。"""
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return
    if np.any(finite_values < 0.0) or np.any(finite_values > 1.0):
        raise ValueError(f"{name}必须是 [0,1] 范围内的归一化灰度值")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="读取 predictions.npz 并计算 v0.2 评测指标"
    )
    parser.add_argument("input", help="predictions.npz 文件路径")
    parser.add_argument(
        "-o",
        "--output",
        help="可选的输出路径，必须以 .json 或 .csv 结尾",
    )
    args = parser.parse_args(argv)

    metrics = evaluate_file(args.input, args.output)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
