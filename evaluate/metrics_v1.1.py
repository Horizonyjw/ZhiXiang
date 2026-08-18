"""eval_v1.1 的独立雷达评测器。

模型提交文件（predictions.npz）只允许含：
  pred:        float32 [N, 3, 1, 352, 512]
  sample_id:   string [N]
  horizon_min: int [3] == [6, 12, 18]

GitHub 仓库 radar_v1/07_pilot_contract/sample_manifest.csv 提供冻结 split 与 target 文件名
气象大模型数据 ZIP 提供对应的原始雷达 PNG

评测范围为整图；不使用 alpha 或其他空间 mask。输出 metrics.json 和
evaluation_manifest.json。CSI/POD/FAR 使用公共配置中 train-only 冻结阈值计算。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image
from skimage import __version__ as SKIMAGE_VERSION
from skimage.metrics import structural_similarity


EVALUATOR_VERSION = "metrics_v1.1"
REQUIRED_SKIMAGE_VERSION = "0.24.0"
EXPECTED_HORIZONS = (6, 12, 18)
EXPECTED_IMAGE_SHAPE = (3, 1, 352, 512)
MASK_DEFINITION = "all_pixels_valid_no_mask"

# ===== 模型侧运行前只需填写下面一个本地路径 =====
# 示例：RADAR_DATA_ZIP_PATH = r"D:\data\气象大模型数据（中南）-20260224.zip"
RADAR_DATA_ZIP_PATH = r""
# =================================================

DEFAULT_SPEC_PATH = Path(__file__).resolve().parents[1] / "docs" / "eval_v1.1.md"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "evaluation_v1.1.yaml"
DEFAULT_SAMPLE_MANIFEST_PATH = (
    Path(__file__).resolve().parents[1]
    / "radar_v1"
    / "07_pilot_contract"
    / "sample_manifest.csv"
)


class EvaluationInvalid(ValueError):
    """提交文件或冻结真值不符合 v1.1 接口。"""


@dataclass(frozen=True)
class Submission:
    pred: np.ndarray
    sample_ids: list[str]
    horizons: tuple[int, int, int]


@dataclass(frozen=True)
class FrozenTargets:
    target: np.ndarray
    sample_ids: list[str]
    radar_zip: Path
    sample_manifest: Path
    sample_manifest_sha256: str
    target_digest: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_evaluation_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path)
    if not config_path.is_file():
        raise EvaluationInvalid(f"找不到公共评测配置：{config_path}")
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise EvaluationInvalid(f"无法读取公共评测配置：{config_path}") from error
    if not isinstance(config, dict):
        raise EvaluationInvalid("公共评测配置必须是 YAML 对象")
    if config.get("evaluation_version") != "v1.1":
        raise EvaluationInvalid("公共评测配置 evaluation_version 必须为 v1.1")
    mask = config.get("mask")
    if not isinstance(mask, dict) or mask.get("enabled") is not False:
        raise EvaluationInvalid("v1.1 评测不得启用 mask")
    if mask.get("definition") != MASK_DEFINITION:
        raise EvaluationInvalid(
            f"mask.definition 必须为 {MASK_DEFINITION}"
        )
    activity = config.get("activity_threshold")
    if not isinstance(activity, dict) or activity.get("enabled") is not True:
        raise EvaluationInvalid("公共评测配置未启用 activity_threshold")
    try:
        threshold = float(activity["value"])
    except (KeyError, TypeError, ValueError) as error:
        raise EvaluationInvalid("activity_threshold.value 必须是数值") from error
    if not 0.0 < threshold <= 1.0:
        raise EvaluationInvalid("activity_threshold.value 必须位于 (0,1]")
    if activity.get("source") != "train_only":
        raise EvaluationInvalid("activity_threshold.source 必须为 train_only")
    metrics = config.get("classification_metrics")
    if not isinstance(metrics, dict) or metrics.get("enabled") is not True:
        raise EvaluationInvalid("公共评测配置未启用 CSI/POD/FAR")
    expected_classification_config = {
        "metrics": ["CSI", "POD", "FAR"],
        "event_definition": "value >= activity_threshold",
        "per_horizon_aggregation": "micro_merge_tp_fp_fn",
        "overall_aggregation": "micro_merge_tp_fp_fn_across_horizons",
        "zero_denominator_value": None,
        "zero_denominator_status": "undefined_zero_denominator",
    }
    for key, expected in expected_classification_config.items():
        if metrics.get(key) != expected:
            raise EvaluationInvalid(
                f"classification_metrics.{key} 必须为 {expected!r}"
            )
    return config, config_path


def _as_ids(values: np.ndarray, field_name: str) -> list[str]:
    if values.ndim != 1:
        raise EvaluationInvalid(f"{field_name} 必须是一维数组，实际形状为 {values.shape}")
    if values.dtype.kind not in {"U", "S"}:
        raise EvaluationInvalid(
            f"{field_name} 必须是字符串数组，实际类型为 {values.dtype}"
        )
    ids = [
        item.decode("utf-8") if isinstance(item, bytes) else str(item)
        for item in values.tolist()
    ]
    if not ids or any(not item for item in ids):
        raise EvaluationInvalid(f"{field_name} 不能为空")
    seen: set[str] = set()
    duplicates: set[str] = set()
    for item in ids:
        if item in seen:
            duplicates.add(item)
        seen.add(item)
    if duplicates:
        raise EvaluationInvalid(f"{field_name} 有重复值：{sorted(duplicates)[:5]}")
    return ids


def _validate_tensor(values: np.ndarray, field_name: str) -> np.ndarray:
    if values.dtype != np.float32:
        raise EvaluationInvalid(f"{field_name} 必须为 float32，实际为 {values.dtype}")
    if values.ndim != 5 or tuple(values.shape[1:]) != EXPECTED_IMAGE_SHAPE:
        raise EvaluationInvalid(
            f"{field_name} 形状必须为 [N,3,1,352,512]，实际为 {values.shape}"
        )
    if values.shape[0] == 0:
        raise EvaluationInvalid(f"{field_name} 的样本数 N 不能为 0")
    if not np.isfinite(values).all():
        raise EvaluationInvalid(f"{field_name} 含 NaN 或 Inf")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise EvaluationInvalid(f"{field_name} 必须在 [0,1] 范围内")
    return values


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    npz_path = Path(path)
    if not npz_path.is_file() or npz_path.suffix.lower() != ".npz":
        raise EvaluationInvalid(f"找不到 NPZ 文件：{npz_path}")
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            return {name: data[name] for name in data.files}
    except (OSError, EOFError, ValueError) as error:
        raise EvaluationInvalid(f"无法读取 NPZ 文件：{npz_path}") from error


def load_submission(path: str | Path) -> Submission:
    data = _load_npz(path)
    required = {"pred", "sample_id", "horizon_min"}
    missing = sorted(required - set(data))
    if missing:
        raise EvaluationInvalid(f"predictions.npz 缺少字段：{', '.join(missing)}")
    extra = sorted(set(data) - required)
    if extra:
        raise EvaluationInvalid(
            "predictions.npz 只能包含 pred、sample_id、horizon_min；"
            f"发现额外字段：{', '.join(extra)}"
        )
    pred = _validate_tensor(data["pred"], "pred")
    ids = _as_ids(data["sample_id"], "sample_id")
    if len(ids) != pred.shape[0]:
        raise EvaluationInvalid("sample_id 数量必须与 pred 的 N 维度一致")
    horizon_values = np.asarray(data["horizon_min"])
    if horizon_values.shape != (3,) or not np.issubdtype(
        horizon_values.dtype, np.integer
    ):
        raise EvaluationInvalid(
            "horizon_min 必须是长度为 3 的整数数组，"
            f"实际形状和类型为 {horizon_values.shape}、{horizon_values.dtype}"
        )
    horizons = tuple(int(item) for item in horizon_values.tolist())
    if horizons != EXPECTED_HORIZONS:
        raise EvaluationInvalid(
            f"horizon_min 必须为 {list(EXPECTED_HORIZONS)}，实际为 {list(horizons)}"
        )
    return Submission(pred=pred, sample_ids=ids, horizons=horizons)


def _preprocess_radar_png(content: bytes) -> np.ndarray:
    """执行 radar_v1 冻结预处理，输出 float32 [1,352,512]。"""
    try:
        with Image.open(io.BytesIO(content)) as image:
            rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    except (OSError, ValueError) as error:
        raise EvaluationInvalid("无法解码冻结雷达 PNG") from error
    rgb = rgba[..., :3].astype(np.float32) / 255.0
    alpha = rgba[..., 3]
    gray = (
        0.299 * rgb[..., 0]
        + 0.587 * rgb[..., 1]
        + 0.114 * rgb[..., 2]
    ).astype(np.float32)
    valid = alpha > 0
    gray[~valid] = 0.0
    gray_small = np.asarray(
        Image.fromarray(gray, mode="F").resize((512, 352), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ).copy()
    valid_small = np.asarray(
        Image.fromarray(valid.astype(np.uint8) * 255, mode="L").resize(
            (512, 352), Image.Resampling.NEAREST
        ),
        dtype=np.uint8,
    ) > 0
    gray_small[~valid_small] = 0.0
    return np.clip(gray_small, 0.0, 1.0).astype(np.float32)[None, ...]


def load_frozen_targets_from_archives(
    radar_zip: str | Path,
    sample_manifest: str | Path,
    split: str,
) -> FrozenTargets:
    """按 GitHub 冻结清单从原始雷达 ZIP 直接读取指定 split 的 target。"""
    if split not in {"val", "test"}:
        raise EvaluationInvalid("冻结 target 只能读取 val 或 test")
    radar_path = Path(radar_zip).resolve()
    manifest_path = Path(sample_manifest).resolve()
    if not radar_path.is_file() or radar_path.suffix.lower() != ".zip":
        raise EvaluationInvalid(f"找不到原始雷达 ZIP：{radar_path}")
    if not manifest_path.is_file() or manifest_path.suffix.lower() != ".csv":
        raise EvaluationInvalid(f"找不到 GitHub 冻结样本清单：{manifest_path}")
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as error:
        raise EvaluationInvalid(f"无法读取冻结样本清单：{manifest_path}") from error
    try:
        manifest_text = manifest_bytes.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(manifest_text))
        required_columns = {"sample_id", "split", "target_paths"}
        missing_columns = sorted(required_columns - set(reader.fieldnames or []))
        if missing_columns:
            raise EvaluationInvalid(
                f"sample_manifest.csv 缺少列：{', '.join(missing_columns)}"
            )
        rows = [row for row in reader if row["split"] == split]
    except UnicodeDecodeError as error:
        raise EvaluationInvalid("sample_manifest.csv 不是 UTF-8") from error
    if not rows:
        raise EvaluationInvalid(f"sample_manifest.csv 中没有 split={split} 样本")

    sample_ids = [row["sample_id"] for row in rows]
    if any(not item for item in sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise EvaluationInvalid(f"split={split} 的 sample_id 为空或重复")

    try:
        with zipfile.ZipFile(radar_path) as radar_archive:
            radar_entries: dict[str, zipfile.ZipInfo] = {}
            for entry in radar_archive.infolist():
                name = Path(entry.filename).name
                if not name.lower().endswith(".png"):
                    continue
                if name in radar_entries:
                    raise EvaluationInvalid(f"原始雷达 ZIP 中图片文件名重复：{name}")
                radar_entries[name] = entry

            cache: dict[str, np.ndarray] = {}
            target_samples: list[np.ndarray] = []
            digest = hashlib.sha256()
            for row in rows:
                try:
                    target_names = json.loads(row["target_paths"])
                except json.JSONDecodeError as error:
                    raise EvaluationInvalid(
                        f"sample_id={row['sample_id']} 的 target_paths 非法"
                    ) from error
                if not isinstance(target_names, list) or len(target_names) != 3:
                    raise EvaluationInvalid(
                        f"sample_id={row['sample_id']} 必须恰好有 3 个 target"
                    )
                frames = []
                for target_name in target_names:
                    filename = Path(target_name).name if isinstance(target_name, str) else ""
                    if not filename or filename not in radar_entries:
                        raise EvaluationInvalid(
                            f"原始雷达 ZIP 缺少 target：{target_name}"
                        )
                    if filename not in cache:
                        cache[filename] = _preprocess_radar_png(
                            radar_archive.read(radar_entries[filename])
                        )
                    frame = cache[filename]
                    digest.update(row["sample_id"].encode("utf-8"))
                    digest.update(filename.encode("utf-8"))
                    digest.update(frame.tobytes(order="C"))
                    frames.append(frame)
                target_samples.append(np.stack(frames, axis=0))
    except (OSError, zipfile.BadZipFile, KeyError) as error:
        raise EvaluationInvalid(f"无法读取原始雷达 ZIP：{radar_path}") from error

    target = _validate_tensor(np.stack(target_samples, axis=0), "target")
    return FrozenTargets(
        target=target,
        sample_ids=sample_ids,
        radar_zip=radar_path,
        sample_manifest=manifest_path,
        sample_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        target_digest=digest.hexdigest(),
    )


def align_targets(submission: Submission, frozen: FrozenTargets) -> np.ndarray:
    target_index = {sample_id: index for index, sample_id in enumerate(frozen.sample_ids)}
    submitted_id_set = set(submission.sample_ids)
    unknown = [
        sample_id for sample_id in submission.sample_ids if sample_id not in target_index
    ]
    if unknown:
        raise EvaluationInvalid(f"提交中存在冻结真值没有的 sample_id：{unknown[:5]}")
    missing = [
        sample_id for sample_id in frozen.sample_ids if sample_id not in submitted_id_set
    ]
    if missing:
        raise EvaluationInvalid(
            f"预测样本不完整，缺少 {len(missing)} 个 sample_id：{missing[:5]}"
        )
    return frozen.target[[target_index[sample_id] for sample_id in submission.sample_ids]]


def _continuous_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = pred.astype(np.float64) - target.astype(np.float64)
    return {
        "mae": float(np.mean(np.abs(error))),
        "mse": float(np.mean(np.square(error))),
    }


def _mean_ssim(pred: np.ndarray, target: np.ndarray) -> float:
    """把 N（及可选的 T）展开，对每张二维图算 SSIM 后取平均。"""
    # per-horizon 输入是 [N, 1, H, W]，overall 输入是 [N, T, 1, H, W]。
    # 两者都展开为 [图像数, H, W]，避免把 T 或通道维误当成空间维。
    pred_images = pred.reshape(-1, pred.shape[-2], pred.shape[-1])
    target_images = target.reshape(-1, target.shape[-2], target.shape[-1])
    scores = [
        structural_similarity(prediction, truth, data_range=1.0)
        for prediction, truth in zip(pred_images, target_images)
    ]
    return float(np.mean(scores))


def _ratio_or_null(numerator: int, denominator: int) -> tuple[float | None, str]:
    if denominator == 0:
        return None, "undefined_zero_denominator"
    return float(numerator / denominator), "computed"


def _classification_metrics(
    pred: np.ndarray, target: np.ndarray, threshold: float
) -> dict[str, Any]:
    pred_event = pred >= threshold
    target_event = target >= threshold
    tp = int(np.count_nonzero(pred_event & target_event))
    fp = int(np.count_nonzero(pred_event & ~target_event))
    fn = int(np.count_nonzero(~pred_event & target_event))
    tn = int(np.count_nonzero(~pred_event & ~target_event))
    csi, csi_status = _ratio_or_null(tp, tp + fp + fn)
    pod, pod_status = _ratio_or_null(tp, tp + fn)
    far, far_status = _ratio_or_null(fp, tp + fp)
    statuses = {"csi": csi_status, "pod": pod_status, "far": far_status}
    return {
        "csi": csi,
        "pod": pod,
        "far": far,
        "event_counts": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "classification_metric_status": statuses,
        "classification_metrics_status": (
            "computed"
            if all(status == "computed" for status in statuses.values())
            else "partial_undefined"
        ),
    }


def _metric_block(
    pred: np.ndarray, target: np.ndarray, activity_threshold: float
) -> dict[str, Any]:
    block = _continuous_metrics(pred, target)
    block["ssim"] = _mean_ssim(pred, target)
    block["valid_image_count"] = int(np.prod(pred.shape[:-2]))
    block["valid_pixel_count"] = int(pred.size)
    block.update(_classification_metrics(pred, target, activity_threshold))
    return block


def evaluate_arrays(
    pred: np.ndarray,
    target: np.ndarray,
    horizons: tuple[int, int, int] = EXPECTED_HORIZONS,
    activity_threshold: float = 0.05,
) -> dict[str, Any]:
    """计算整图 MAE/MSE/SSIM 及 train-only 阈值 CSI/POD/FAR。"""
    pred = _validate_tensor(pred, "pred")
    target = _validate_tensor(target, "target")
    if pred.shape != target.shape:
        raise EvaluationInvalid(f"pred 和 target 形状不同：{pred.shape} != {target.shape}")
    if tuple(horizons) != EXPECTED_HORIZONS:
        raise EvaluationInvalid(f"horizons 必须为 {EXPECTED_HORIZONS}")

    per_horizon = {}
    for step, minutes in enumerate(horizons):
        per_horizon[f"T+{minutes}"] = _metric_block(
            pred[:, step], target[:, step], activity_threshold
        )

    return {
        "per_horizon": per_horizon,
        "overall": _metric_block(pred, target, activity_threshold),
        "threshold": activity_threshold,
        "threshold_source": "train_only",
        "mask_definition": MASK_DEFINITION,
        "valid_count": {
            "sample_count": int(pred.shape[0]),
            "image_count": int(pred.shape[0] * pred.shape[1]),
            "pixel_count": int(pred.size),
        },
    }


def _decision_inputs(metrics: dict[str, Any], split: str) -> dict[str, Any]:
    return {
        "metric_split": split,
        "primary_metric": f"{split}_overall_mae",
        "secondary_metrics": [f"{split}_overall_mse", f"{split}_overall_ssim"],
        "per_horizon_metrics": ["mae", "mse", "ssim"],
        "comparison_baseline": "Persistence",
        "minimum_relative_mae_improvement": 0.005,
        "maximum_ssim_drop": 0.002,
        "candidate_selection_allowed": split == "val",
        "current_metrics": {
            "overall_mae": metrics["overall"]["mae"],
            "overall_mse": metrics["overall"]["mse"],
            "overall_ssim": metrics["overall"]["ssim"],
        },
        "decision": (
            "not_compared_to_persistence"
            if split == "val"
            else "final_certification_only"
        ),
    }


def _spec_digest(spec_path: Path | None) -> str | None:
    return _sha256_file(spec_path) if spec_path and spec_path.is_file() else None


def evaluate_files(
    predictions_path: str | Path,
    radar_zip: str | Path,
    sample_manifest: str | Path,
    *,
    experiment_id: str,
    split: str,
    dataset_version: str,
    spec_path: str | Path | None = None,
    config_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if split not in {"val", "test"}:
        raise EvaluationInvalid("split 只能为 val 或 test")
    if SKIMAGE_VERSION != REQUIRED_SKIMAGE_VERSION:
        raise EvaluationInvalid(
            "SSIM 依赖版本不符合规范："
            f"需要 scikit-image=={REQUIRED_SKIMAGE_VERSION}，实际为 {SKIMAGE_VERSION}"
    )
    submission = load_submission(predictions_path)
    frozen = load_frozen_targets_from_archives(radar_zip, sample_manifest, split)
    target = align_targets(submission, frozen)
    config, resolved_config_path = load_evaluation_config(
        config_path or DEFAULT_CONFIG_PATH
    )
    activity_threshold = float(config["activity_threshold"]["value"])
    evaluated = evaluate_arrays(
        submission.pred, target, submission.horizons, activity_threshold
    )

    spec = Path(spec_path) if spec_path else DEFAULT_SPEC_PATH
    if not spec.is_file():
        raise EvaluationInvalid(f"找不到评测规范文件：{spec}")
    spec_digest = _spec_digest(spec)
    metrics = {
        "experiment_id": experiment_id,
        "spec_digest": spec_digest,
        "dataset_version": dataset_version,
        "evaluator_version": EVALUATOR_VERSION,
        "split": split,
        **evaluated,
        "decision_inputs": _decision_inputs(evaluated, split),
        "evaluation_status": "completed",
    }
    manifest = {
        "experiment_id": experiment_id,
        "dataset_version": dataset_version,
        "evaluator_version": EVALUATOR_VERSION,
        "spec_path": str(spec),
        "spec_digest": spec_digest,
        "predictions_path": str(Path(predictions_path)),
        "predictions_sha256": _sha256_file(Path(predictions_path)),
        "radar_zip_path": str(frozen.radar_zip),
        "radar_zip_sha256": _sha256_file(frozen.radar_zip),
        "sample_manifest_path": str(frozen.sample_manifest),
        "sample_manifest_sha256": frozen.sample_manifest_sha256,
        "aligned_target_content_sha256": frozen.target_digest,
        "evaluation_config_path": str(resolved_config_path),
        "evaluation_config_sha256": _sha256_file(resolved_config_path),
        "activity_threshold": activity_threshold,
        "activity_threshold_source": config["activity_threshold"]["source"],
        "numpy_version": np.__version__,
        "scikit_image_version": SKIMAGE_VERSION,
        "evaluator_script_sha256": _sha256_file(Path(__file__)),
        "mask_definition": MASK_DEFINITION,
        "horizon_min": list(submission.horizons),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_status": "completed",
    }
    return metrics, manifest


def _invalid_result(
    experiment_id: str,
    split: str,
    dataset_version: str,
    error: Exception,
    spec_path: str | Path | None,
) -> dict[str, Any]:
    spec = Path(spec_path) if spec_path else DEFAULT_SPEC_PATH
    return {
        "experiment_id": experiment_id,
        "spec_digest": _spec_digest(spec),
        "dataset_version": dataset_version,
        "evaluator_version": EVALUATOR_VERSION,
        "split": split,
        "per_horizon": None,
        "overall": None,
        "threshold": None,
        "mask_definition": MASK_DEFINITION,
        "valid_count": None,
        "decision_inputs": None,
        "evaluation_status": "invalid",
        "error_type": type(error).__name__,
        "error_message": str(error),
    }


def _write_json(path: Path, content: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")


def run_self_test() -> None:
    """运行 v1.1 规定的数值、分类指标、接口与 sample_id 边界测试。"""
    shape = (2, 3, 1, 352, 512)
    target = np.zeros(shape, dtype=np.float32)
    target[:, :, :, 100:120, 100:120] = 0.5
    perfect = evaluate_arrays(target.copy(), target)
    assert perfect["overall"]["mae"] == 0.0
    assert perfect["overall"]["mse"] == 0.0
    assert abs(perfect["overall"]["ssim"] - 1.0) < 1e-12
    assert perfect["overall"]["csi"] == 1.0
    assert perfect["overall"]["pod"] == 1.0
    assert perfect["overall"]["far"] == 0.0
    zeros = evaluate_arrays(np.zeros_like(target), target)
    assert zeros["overall"]["mae"] > 0.0
    no_activity = evaluate_arrays(np.zeros_like(target), np.zeros_like(target))
    assert no_activity["overall"]["mae"] == 0.0
    assert no_activity["overall"]["csi"] is None
    assert no_activity["overall"]["pod"] is None
    assert no_activity["overall"]["far"] is None
    all_active = np.ones_like(target)
    all_active_metrics = evaluate_arrays(all_active, all_active)["overall"]
    assert all_active_metrics["mae"] == 0.0
    assert all_active_metrics["csi"] == 1.0
    assert all_active_metrics["pod"] == 1.0
    assert all_active_metrics["far"] == 0.0

    # 1 个 TP、1 个 FP、1 个 FN：验证公式和 micro 计数没有写反。
    event_target = np.zeros(shape, dtype=np.float32)
    event_pred = np.zeros(shape, dtype=np.float32)
    event_target[0, 0, 0, 0, 0:2] = 0.5
    event_pred[0, 0, 0, 0, 0] = 0.5
    event_pred[0, 0, 0, 0, 2] = 0.5
    classification = evaluate_arrays(event_pred, event_target)["overall"]
    assert classification["event_counts"]["tp"] == 1
    assert classification["event_counts"]["fp"] == 1
    assert classification["event_counts"]["fn"] == 1
    assert classification["csi"] == 1 / 3
    assert classification["pod"] == 1 / 2
    assert classification["far"] == 1 / 2
    for broken in (
        np.full_like(target, np.nan),
        np.full_like(target, np.inf),
        target[:, :, :, :-1, :],
    ):
        try:
            evaluate_arrays(broken, target)
        except EvaluationInvalid:
            pass
        else:
            raise AssertionError("边界测试应当返回 EvaluationInvalid")

    # 文件级接口测试：乱序可对齐；缺失、重复、额外字段均拒绝。
    with tempfile.TemporaryDirectory() as temporary_directory:
        temp_dir = Path(temporary_directory)
        radar_zip = temp_dir / "radar.zip"
        sample_manifest = temp_dir / "sample_manifest.csv"
        frozen_ids = np.asarray(["S000001", "S000002"])
        rows = []
        with zipfile.ZipFile(radar_zip, "w") as radar_archive:
            for sample_index, sample_id in enumerate(frozen_ids.tolist()):
                target_paths = []
                pixel_value = 64 if sample_index == 0 else 192
                rgba = np.full((352, 512, 4), pixel_value, dtype=np.uint8)
                rgba[..., 3] = 255
                for horizon_index in range(3):
                    filename = f"{sample_id}_{horizon_index}.png"
                    buffer = io.BytesIO()
                    Image.fromarray(rgba, mode="RGBA").save(buffer, format="PNG")
                    radar_archive.writestr(f"radar/{filename}", buffer.getvalue())
                    target_paths.append(filename)
                rows.append(
                    {
                        "sample_id": sample_id,
                        "split": "val",
                        "target_paths": json.dumps(target_paths),
                    }
                )
        manifest_buffer = io.StringIO()
        writer = csv.DictWriter(
            manifest_buffer,
            fieldnames=["sample_id", "split", "target_paths"],
        )
        writer.writeheader()
        writer.writerows(rows)
        sample_manifest.write_bytes(manifest_buffer.getvalue().encode("utf-8-sig"))
        frozen = load_frozen_targets_from_archives(radar_zip, sample_manifest, "val")
        assert frozen.target.shape == shape
        frozen_target = frozen.target

        shuffled_path = temp_dir / "predictions_shuffled.npz"
        np.savez(
            shuffled_path,
            pred=frozen_target[[1, 0]],
            sample_id=frozen_ids[[1, 0]],
            horizon_min=np.asarray(EXPECTED_HORIZONS, dtype=np.int64),
        )
        shuffled = load_submission(shuffled_path)
        aligned = align_targets(shuffled, frozen)
        shuffled_metrics = evaluate_arrays(shuffled.pred, aligned)
        assert shuffled_metrics["overall"]["mae"] == 0.0
        assert abs(shuffled_metrics["overall"]["ssim"] - 1.0) < 1e-12

        invalid_submissions = {
            "missing_sample": {
                "pred": frozen_target[:1],
                "sample_id": frozen_ids[:1],
                "horizon_min": np.asarray(EXPECTED_HORIZONS, dtype=np.int64),
            },
            "duplicate_id": {
                "pred": frozen_target,
                "sample_id": np.asarray(["S000001", "S000001"]),
                "horizon_min": np.asarray(EXPECTED_HORIZONS, dtype=np.int64),
            },
            "extra_target_field": {
                "pred": frozen_target,
                "sample_id": frozen_ids,
                "horizon_min": np.asarray(EXPECTED_HORIZONS, dtype=np.int64),
                "target": frozen_target,
            },
        }
        for case_name, arrays in invalid_submissions.items():
            case_path = temp_dir / f"{case_name}.npz"
            np.savez(case_path, **arrays)
            try:
                candidate = load_submission(case_path)
                align_targets(candidate, frozen)
            except EvaluationInvalid:
                pass
            else:
                raise AssertionError(f"边界测试 {case_name} 应当返回 EvaluationInvalid")
    print("self-test passed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="eval_v1.1 独立雷达评测器")
    parser.add_argument("--predictions", help="模型提交的 predictions.npz")
    parser.add_argument(
        "--radar-zip",
        help="包含 7396 张原始雷达 PNG 的气象大模型数据 ZIP",
    )
    parser.add_argument(
        "--sample-manifest",
        default=str(DEFAULT_SAMPLE_MANIFEST_PATH),
        help="GitHub 仓库中的 radar_v1/07_pilot_contract/sample_manifest.csv",
    )
    parser.add_argument("--output-dir", default="evaluate_output")
    parser.add_argument("--experiment-id", default="unknown-experiment")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--dataset-version", default="unknown-dataset")
    parser.add_argument("--spec", default=str(DEFAULT_SPEC_PATH))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.self_test:
        run_self_test()
        return 0
    radar_zip = args.radar_zip or RADAR_DATA_ZIP_PATH
    if not args.predictions:
        parser.error("评测时必须提供 --predictions")
    if not radar_zip:
        parser.error(
            "请先在脚本开头填写 RADAR_DATA_ZIP_PATH，或提供 --radar-zip"
        )

    output_dir = Path(args.output_dir)
    try:
        metrics, manifest = evaluate_files(
            args.predictions,
            radar_zip,
            args.sample_manifest,
            experiment_id=args.experiment_id,
            split=args.split,
            dataset_version=args.dataset_version,
            spec_path=args.spec,
            config_path=args.config,
        )
    except EvaluationInvalid as error:
        metrics = _invalid_result(
            args.experiment_id,
            args.split,
            args.dataset_version,
            error,
            args.spec,
        )
        manifest = {
            "experiment_id": args.experiment_id,
            "evaluator_version": EVALUATOR_VERSION,
            "evaluation_status": "invalid",
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
        _write_json(output_dir / "metrics.json", metrics)
        _write_json(output_dir / "evaluation_manifest.json", manifest)
        print(json.dumps(metrics, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2

    _write_json(output_dir / "metrics.json", metrics)
    _write_json(output_dir / "evaluation_manifest.json", manifest)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
