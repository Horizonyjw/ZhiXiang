"""
从 predictions.npz 读取预测并调用统一评测（v0.2）。

用法（仓库根目录）:
  .\\.venv\\Scripts\\python.exe -m evaluate.run_from_npz ^
      --npz results/<experiment_id>/predictions.npz ^
      --output-dir results/<experiment_id> ^
      --plot
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate import evaluate, save_metrics


def load_predictions_npz(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    required = ("targets", "predictions")
    for key in required:
        if key not in data.files:
            raise KeyError(f"npz 缺少字段 `{key}`，现有字段：{list(data.files)}")
    payload = {
        "targets": np.asarray(data["targets"]),
        "predictions": np.asarray(data["predictions"]),
    }
    if "inputs" in data.files:
        payload["inputs"] = np.asarray(data["inputs"])
    if "metadata_json" in data.files:
        raw = data["metadata_json"]
        text = raw.item() if getattr(raw, "shape", ()) == () else str(raw)
        payload["metadata"] = json.loads(str(text))
    return payload


def resolve_experiment_id(payload: dict, output_dir: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    metadata = payload.get("metadata") or {}
    if isinstance(metadata, dict) and metadata.get("experiment_id"):
        return str(metadata["experiment_id"])
    return output_dir.name


def clip_to_unit_interval(array: np.ndarray, name: str) -> tuple[np.ndarray, bool]:
    """线性头可能略超出 [0,1]；评测 v0.2 要求有限值必须在该区间。"""
    finite = np.isfinite(array)
    if not np.any(finite):
        return array, False
    finite_vals = array[finite]
    out_of_range = bool(np.any(finite_vals < 0.0) or np.any(finite_vals > 1.0))
    if not out_of_range:
        return array, False
    clipped = np.clip(array, 0.0, 1.0)
    print(f"warning: {name} 存在超出 [0,1] 的有限值，已 clip 后再评测")
    return clipped, True


def run(
    npz_path: Path,
    output_dir: Path,
    plot: bool = False,
    sample_index: int = 0,
    experiment_id: str | None = None,
    overwrite: bool = False,
    clip: bool = True,
) -> dict:
    payload = load_predictions_npz(npz_path)
    predictions = payload["predictions"]
    targets = payload["targets"]
    clipped_keys = []
    if clip:
        predictions, pred_clipped = clip_to_unit_interval(predictions, "预测值")
        targets, tgt_clipped = clip_to_unit_interval(targets, "真值")
        if pred_clipped:
            clipped_keys.append("predictions")
        if tgt_clipped:
            clipped_keys.append("targets")

    metrics = evaluate(predictions, targets)

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "metrics.json"
    csv_path = output_dir / "metrics.csv"
    save_metrics(metrics, json_path)
    save_metrics(metrics, csv_path)

    exp_id = resolve_experiment_id(payload, output_dir, experiment_id)
    result = {
        "npz": str(npz_path),
        "metrics_json": str(json_path),
        "metrics_csv": str(csv_path),
        "evaluation_version": metrics.get("evaluation_version"),
        "experiment_id": exp_id,
        "overall": metrics["overall"],
        "clipped_to_unit_interval": clipped_keys,
    }

    if plot:
        if "inputs" not in payload:
            raise KeyError("绘图需要 npz 中的 `inputs` 字段")
        from visualize.plot_results import make_result_figures

        figures_dir = output_dir / "figures"
        paths = make_result_figures(
            payload["inputs"],
            targets,
            predictions,
            metrics,
            figures_dir,
            experiment_id=exp_id,
            sample_index=sample_index,
            overwrite=overwrite,
        )
        result["figures"] = paths

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="从 predictions.npz 运行统一评测（v0.2）")
    parser.add_argument("--npz", required=True, help="predictions.npz 路径")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="指标与图片输出目录（默认与 npz 同目录）",
    )
    parser.add_argument("--plot", action="store_true", help="同时生成结果图")
    parser.add_argument("--sample-index", type=int, default=0, help="绘图使用的样本下标")
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="结果图实验编号；默认读 npz metadata 或输出目录名",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已有结果图（默认不覆盖）",
    )
    parser.add_argument(
        "--no-clip",
        action="store_true",
        help="不对超出 [0,1] 的值做 clip（可能触发评测报错）",
    )
    args = parser.parse_args()

    npz_path = Path(args.npz)
    if not npz_path.is_absolute():
        npz_path = ROOT / npz_path
    output_dir = Path(args.output_dir) if args.output_dir else npz_path.parent
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    result = run(
        npz_path=npz_path,
        output_dir=output_dir,
        plot=args.plot,
        sample_index=args.sample_index,
        experiment_id=args.experiment_id,
        overwrite=args.overwrite,
        clip=not args.no_clip,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
