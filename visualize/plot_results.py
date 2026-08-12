"""雷达回波预测结果图模板。仅依赖 NumPy 和 Pillow。"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def make_result_figures(inputs, target, prediction, metrics, output_dir, sample_index=0, channel_index=0):
    """生成序列对比图和逐预测步长指标图。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sequence_path = output_dir / "sequence_comparison.png"
    metric_path = output_dir / "metric_changes.png"
    plot_sequence_comparison(inputs, target, prediction, sequence_path, sample_index, channel_index)
    plot_metric_changes(metrics["per_lead_time"], metric_path)
    return {"sequence_comparison": str(sequence_path), "metric_changes": str(metric_path)}


def plot_sequence_comparison(inputs, target, prediction, output_path, sample_index=0, channel_index=0):
    """绘制历史输入、真实未来、预测未来和预测误差。"""
    inputs, target, prediction = _check_sequences(inputs, target, prediction, sample_index, channel_index)
    history = inputs[sample_index, :, channel_index]
    truth = target[sample_index, :, channel_index]
    forecast = prediction[sample_index, :, channel_index]
    error = np.abs(forecast - truth)

    # 每列代表一个时间帧；四行依次为输入、真值、预测和误差。
    columns = max(len(history), len(truth))
    tile_size, label_width, top_height = 150, 75, 34
    image = Image.new("RGB", (label_width + columns * tile_size, top_height + 4 * tile_size), "white")
    draw = ImageDraw.Draw(image)
    font = _font(14)
    labels = ["Input", "Truth", "Prediction", "Abs error"]
    for row, label in enumerate(labels):
        draw.text((6, top_height + row * tile_size + 8), label, fill="black", font=font)
    for column in range(columns):
        draw.text((label_width + column * tile_size + 6, 8), f"Step {column + 1}", fill="black", font=font)

    all_fields = np.concatenate([history.ravel(), truth.ravel(), forecast.ravel()])
    field_min, field_max = _range(all_fields)
    error_min, error_max = _error_range(error)
    _paste_row(image, history, 0, label_width, top_height, tile_size, field_min, field_max, _field_color)
    _paste_row(image, truth, 1, label_width, top_height, tile_size, field_min, field_max, _field_color)
    _paste_row(image, forecast, 2, label_width, top_height, tile_size, field_min, field_max, _field_color)
    _paste_row(image, error, 3, label_width, top_height, tile_size, error_min, error_max, _error_color)
    image.save(output_path)


def plot_metric_changes(per_lead_time, output_path):
    """绘制 MAE、MSE、CSI、POD、FAR 随预测步长的变化。"""
    names = ["mae", "mse", "csi", "pod", "far"]
    panel_width, panel_height, margin = 210, 180, 18
    image = Image.new("RGB", (len(names) * panel_width, panel_height + 2 * margin), "white")
    draw = ImageDraw.Draw(image)
    font = _font(14)

    steps = [row["step"] if "step" in row else row["lead_time_index"] for row in per_lead_time]
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


def _paste_row(image, frames, row, left, top, tile_size, lower, upper, color_function):
    for column, frame in enumerate(frames):
        tile = Image.fromarray(color_function(frame, lower, upper)).resize((tile_size, tile_size), Image.Resampling.NEAREST)
        image.paste(tile, (left + column * tile_size, top + row * tile_size))


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
    valid = [(step, value) for step, value in zip(steps, values) if value is not None]
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
    draw.text((left, bottom + 4), "Lead time", fill="black", font=font)


def _font(size):
    for path in ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/msyh.ttc"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()
