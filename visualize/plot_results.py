"""雷达回波预测结果图模板"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_result_figures_from_file(
    predictions_path,
    metrics_path=None,
    output_dir=None,
    experiment_id=None,
    prediction_label="Model prediction",
    lead_interval_minutes=6,
    sample_index=0,
    channel_index=0,
    overwrite=False,
):
    """直接读取 predictions.npz 并生成约定的两张结果图。

    默认优先从 metadata_json 读取 experiment_id；如果文件位于
    results/<experiment_id>/predictions.npz，也可从父目录推断。
    未显式传入 output_dir 时，图片保存到
    results/<experiment_id>/figures/。
    """
    predictions_path = Path(predictions_path)
    if not predictions_path.is_file():
        raise FileNotFoundError(f"找不到预测文件：{predictions_path}")
    if predictions_path.suffix.lower() != ".npz":
        raise ValueError("输入文件必须以 .npz 结尾")

    try:
        with np.load(predictions_path, allow_pickle=False) as data:
            required = ("inputs", "targets", "predictions")
            missing = [key for key in required if key not in data]
            if missing:
                raise ValueError(
                    f"NPZ 缺少必需字段：{', '.join(missing)}；"
                    f"当前字段：{', '.join(data.files)}"
                )
            inputs = data["inputs"]
            target = data["targets"]
            prediction = data["predictions"]
            metadata = _read_metadata_json(data)
    except (OSError, EOFError) as error:
        raise ValueError(f"无法读取 NPZ 文件：{predictions_path}") from error

    experiment_id = _resolve_experiment_id(
        experiment_id, metadata, predictions_path.parent
    )
    metrics = _resolve_metrics(
        metrics_path, predictions_path, experiment_id, prediction, target
    )
    _validate_metrics(metrics, prediction.shape[1])

    if output_dir is None:
        if predictions_path.parent.name == experiment_id:
            output_dir = predictions_path.parent / "figures"
        else:
            output_dir = PROJECT_ROOT / "results" / experiment_id / "figures"

    return make_result_figures(
        inputs,
        target,
        prediction,
        metrics,
        output_dir,
        experiment_id,
        prediction_label=prediction_label,
        lead_interval_minutes=lead_interval_minutes,
        sample_index=sample_index,
        channel_index=channel_index,
        overwrite=overwrite,
    )


def make_result_figures(
    inputs,
    target,
    prediction,
    metrics,
    output_dir,
    experiment_id,
    prediction_label="Model prediction",
    lead_interval_minutes=6,
    sample_index=0,
    channel_index=0,
    overwrite=False,
):
    """按实验编号生成序列对比图和逐预测步长指标图。"""
    _validate_experiment_id(experiment_id)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sequence_path = output_dir / f"{experiment_id}_sequence-comparison.png"
    metric_path = output_dir / f"{experiment_id}_metric-changes.png"
    _check_output_paths((sequence_path, metric_path), overwrite)
    plot_sequence_comparison(
        inputs,
        target,
        prediction,
        sequence_path,
        prediction_label=prediction_label,
        lead_interval_minutes=lead_interval_minutes,
        sample_index=sample_index,
        channel_index=channel_index,
    )
    plot_metric_changes(
        metrics["per_lead_time"],
        metric_path,
        lead_interval_minutes=lead_interval_minutes,
    )
    return {"sequence_comparison": str(sequence_path), "metric_changes": str(metric_path)}


def plot_sequence_comparison(
    inputs,
    target,
    prediction,
    output_path,
    prediction_label="Model prediction",
    lead_interval_minutes=6,
    sample_index=0,
    channel_index=0,
):
    """绘制历史输入、真实未来、预测未来和预测误差。"""
    _validate_display_options(prediction_label, lead_interval_minutes)
    inputs, target, prediction = _check_sequences(inputs, target, prediction, sample_index, channel_index)
    history = inputs[sample_index, :, channel_index]
    truth = target[sample_index, :, channel_index]
    forecast = prediction[sample_index, :, channel_index]
    error = np.abs(forecast - truth)

    # 每列代表一个时间帧；四行依次为历史输入、真实未来、预测未来和预测误差。
    columns = max(len(history), len(truth))
    tile_size, label_width, row_header_height = 150, 150, 28
    row_height = row_header_height + tile_size
    image = Image.new("RGB", (label_width + columns * tile_size, 4 * row_height), "white")
    draw = ImageDraw.Draw(image)
    font = _font(14)
    labels = ["Historical input", "Future truth", prediction_label, "Absolute error"]
    for row, label in enumerate(labels):
        draw.text((6, row * row_height + row_header_height + 8), label, fill="black", font=font)

    # 历史帧以预测起点为 T0 倒序标记，未来帧按实际预测分钟数标记。
    for column in range(len(history)):
        minutes_before = (len(history) - column - 1) * lead_interval_minutes
        label = "T0" if minutes_before == 0 else f"T-{minutes_before} min"
        draw.text((label_width + column * tile_size + 6, 8), label, fill="black", font=font)
    for row in (1, 2, 3):
        for column in range(len(truth)):
            label = f"T+{(column + 1) * lead_interval_minutes} min"
            draw.text(
                (label_width + column * tile_size + 6, row * row_height + 8),
                label,
                fill="black",
                font=font,
            )

    all_fields = np.concatenate([history.ravel(), truth.ravel(), forecast.ravel()])
    field_min, field_max = _range(all_fields)
    error_min, error_max = _error_range(error)
    _paste_row(image, history, 0, label_width, row_header_height, row_height, tile_size, field_min, field_max, _field_color)
    _paste_row(image, truth, 1, label_width, row_header_height, row_height, tile_size, field_min, field_max, _field_color)
    _paste_row(image, forecast, 2, label_width, row_header_height, row_height, tile_size, field_min, field_max, _field_color)
    _paste_row(image, error, 3, label_width, row_header_height, row_height, tile_size, error_min, error_max, _error_color)
    image.save(output_path)


def plot_metric_changes(per_lead_time, output_path, lead_interval_minutes=6):
    """绘制当前评测版本中实际可计算指标随预测步长的变化。"""
    if not isinstance(lead_interval_minutes, int) or lead_interval_minutes <= 0:
        raise ValueError("lead_interval_minutes 必须是正整数")
    preferred_names = ["mae", "mse", "csi", "pod", "far"]
    # v0.2 的 CSI、POD、FAR 为 None；这些未计算指标不生成空白面板。
    # 如果后续版本恢复后三项且结果中存在有限数值，面板会自动重新出现。
    names = [
        name
        for name in preferred_names
        if any(_is_finite_number(row.get(name)) for row in per_lead_time)
    ]
    if not names:
        names = ["mae", "mse"]
    panel_width, panel_height, margin = 210, 180, 18
    image = Image.new("RGB", (len(names) * panel_width, panel_height + 2 * margin), "white")
    draw = ImageDraw.Draw(image)
    font = _font(14)

    steps = [
        (row["step"] if "step" in row else row["lead_time_index"])
        * lead_interval_minutes
        for row in per_lead_time
    ]
    for index, name in enumerate(names):
        values = [row.get(name) for row in per_lead_time]
        _draw_line_chart(draw, index * panel_width, margin, panel_width, panel_height, name.upper(), steps, values, font)
    image.save(output_path)


def _check_sequences(inputs, target, prediction, sample_index, channel_index):
    inputs, target, prediction = map(np.asarray, (inputs, target, prediction))
    if target.shape != prediction.shape:
        raise ValueError("真实未来序列与预测序列的维度必须一致")
    if any(array.ndim != 5 for array in (inputs, target, prediction)):
        raise ValueError("输入、真值和预测均必须是 [B, T, C, H, W] 五维数组")
    if inputs.shape[0] != target.shape[0] or inputs.shape[2:] != target.shape[2:]:
        raise ValueError("历史输入与真实未来的批次、通道和空间维度必须一致")
    if not 0 <= sample_index < target.shape[0] or not 0 <= channel_index < target.shape[2]:
        raise ValueError("sample_index 或 channel_index 超出数组范围")
    return inputs, target, prediction


def _paste_row(image, frames, row, left, top, row_height, tile_size, lower, upper, color_function):
    for column, frame in enumerate(frames):
        tile = Image.fromarray(color_function(frame, lower, upper)).resize((tile_size, tile_size), Image.Resampling.NEAREST)
        image.paste(tile, (left + column * tile_size, top + row * row_height))


def _field_color(values, lower, upper):
    ratio = _normalise(values, lower, upper)
    return np.uint8(np.stack([255 * ratio, 200 * np.sqrt(ratio), 255 * (1 - ratio)], axis=-1))


def _error_color(values, lower, upper):
    ratio = _normalise(values, lower, upper)
    return np.uint8(np.stack([255 * np.ones_like(ratio), 255 * (1 - ratio), 255 * (1 - ratio)], axis=-1))


def _normalise(values, lower, upper):
    return np.clip((np.nan_to_num(values, nan=lower) - lower) / (upper - lower), 0, 1)


def _range(values):
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    lower, upper = float(np.min(values)), float(np.max(values))
    return (lower, upper) if lower < upper else (lower - 0.5, upper + 0.5)


def _error_range(values):
    values = values[np.isfinite(values)]
    upper = float(np.max(values)) if values.size else 1.0
    return 0.0, upper if upper > 0 else 1.0


def _draw_line_chart(draw, x, y, width, height, title, steps, values, font):
    left, right, top, bottom = x + 30, x + width - 10, y + 24, y + height - 24
    draw.text((left, y), title, fill="black", font=font)
    draw.line((left, top, left, bottom, right, bottom), fill="black", width=1)
    valid = [
        (step, float(value))
        for step, value in zip(steps, values)
        if _is_finite_number(value)
    ]
    if not valid:
        draw.text((left, (top + bottom) // 2), "No data", fill="black", font=font)
        return
    numbers = [float(value) for _, value in valid]
    lower, upper = _range(np.array(numbers))
    step_min, step_max = min(steps), max(steps)
    step_span = max(step_max - step_min, 1)
    points = [
        (left + (step - step_min) / step_span * (right - left), bottom - (value - lower) / (upper - lower) * (bottom - top))
        for step, value in valid
    ]
    if len(points) > 1:
        draw.line(points, fill="blue", width=2)
    for point in points:
        draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill="blue")
    draw.text((left, bottom + 4), "Lead time (min)", fill="black", font=font)


def _font(size):
    for path in ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/msyh.ttc"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _is_finite_number(value):
    """仅把可转换为有限浮点数的指标值视为可绘制数据。"""
    if value is None:
        return False
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _validate_experiment_id(experiment_id):
    """检查实验编号是否符合 YYYYMMDD-模型-数据版本-序号。"""
    if not isinstance(experiment_id, str):
        raise TypeError("experiment_id 必须是字符串")
    pattern = r"^\d{8}-[A-Za-z0-9_\u4e00-\u9fff.]+-[A-Za-z0-9_\u4e00-\u9fff.]+-\d+$"
    if re.fullmatch(pattern, experiment_id) is None:
        raise ValueError(
            "实验编号必须符合 YYYYMMDD-模型-数据版本-序号，"
            "例如 20260813-unet-radar_v1-01"
        )


def _check_output_paths(paths, overwrite):
    """默认禁止覆盖已有图片；只有 overwrite=True 时允许覆盖。"""
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "以下结果图已存在，未执行覆盖：" + "，".join(existing)
        )


def _validate_display_options(prediction_label, lead_interval_minutes):
    """检查预测行标签和时间间隔是否可用于结果图。"""
    if not isinstance(prediction_label, str) or not prediction_label.strip():
        raise ValueError("prediction_label 必须是非空字符串，例如 Persistence prediction")
    if not isinstance(lead_interval_minutes, int) or lead_interval_minutes <= 0:
        raise ValueError("lead_interval_minutes 必须是正整数")


def _read_metadata_json(data):
    if "metadata_json" not in data:
        return {}
    raw = data["metadata_json"]
    if raw.size != 1:
        raise ValueError("metadata_json 必须是单个 JSON 字符串")
    try:
        metadata = json.loads(str(raw.item()))
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError("metadata_json 不是有效的 JSON") from error
    if not isinstance(metadata, dict):
        raise ValueError("metadata_json 的顶层结构必须是对象")
    return metadata


def _resolve_experiment_id(explicit_id, metadata, parent_dir):
    candidates = [explicit_id, metadata.get("experiment_id")]
    if parent_dir.name != "results":
        candidates.append(parent_dir.name)
    for candidate in candidates:
        if candidate is None:
            continue
        _validate_experiment_id(candidate)
        return candidate
    raise ValueError(
        "无法确定 experiment_id：请在 metadata_json 中提供，"
        "将文件放入 results/<experiment_id>/，或显式传入"
    )


def _resolve_metrics(metrics_path, predictions_path, experiment_id, prediction, target):
    if metrics_path is not None:
        candidate_paths = [Path(metrics_path)]
    else:
        candidate_paths = [
            predictions_path.parent / "metrics.json",
            PROJECT_ROOT / "results" / experiment_id / "metrics.json",
        ]
    for path in candidate_paths:
        if path.is_file():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(f"无法读取指标文件：{path}") from error
    if metrics_path is not None:
        raise FileNotFoundError(f"找不到指标文件：{metrics_path}")

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from evaluate import evaluate

    return evaluate(prediction, target)


def _validate_metrics(metrics, expected_steps):
    if not isinstance(metrics, dict) or not isinstance(
        metrics.get("per_lead_time"), list
    ):
        raise ValueError("指标结果必须包含 per_lead_time 列表")
    if len(metrics["per_lead_time"]) != expected_steps:
        raise ValueError(
            "指标的预测步数与 predictions 的 Tout 不一致："
            f"{len(metrics['per_lead_time'])} != {expected_steps}"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="从 predictions.npz 生成序列对比图和指标变化图"
    )
    parser.add_argument("predictions", help="predictions.npz 文件路径")
    parser.add_argument("--metrics", help="可选的 metrics.json 路径")
    parser.add_argument("--output-dir", help="可选的图片输出目录")
    parser.add_argument("--experiment-id", help="可选；默认从元数据或父目录读取")
    parser.add_argument("--prediction-label", default="Model prediction")
    parser.add_argument("--lead-interval-minutes", type=int, default=6)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--channel-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    paths = make_result_figures_from_file(
        args.predictions,
        metrics_path=args.metrics,
        output_dir=args.output_dir,
        experiment_id=args.experiment_id,
        prediction_label=args.prediction_label,
        lead_interval_minutes=args.lead_interval_minutes,
        sample_index=args.sample_index,
        channel_index=args.channel_index,
        overwrite=args.overwrite,
    )
    print(json.dumps(paths, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

