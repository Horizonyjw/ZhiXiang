"""雷达回波预测结果图模板"""

import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


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
