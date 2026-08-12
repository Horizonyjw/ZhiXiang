"""
从 predictions.npz 读取预测并调用统一评测（对接 zjr 分支 metrics）。

用法（仓库根目录）:
  .\\.venv\\Scripts\\python.exe -m evaluate.run_from_npz ^
      --npz results/<experiment_id>/predictions.npz ^
      --output-dir results/<experiment_id> ^
      --threshold 20.0 ^
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

from evaluate.metrics import evaluate, save_metrics


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


def run(
    npz_path: Path,
    output_dir: Path,
    threshold_dbz: float = 20.0,
    plot: bool = False,
    sample_index: int = 0,
) -> dict:
    payload = load_predictions_npz(npz_path)
    metrics = evaluate(
        payload["predictions"],
        payload["targets"],
        threshold_dbz=threshold_dbz,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "metrics.json"
    csv_path = output_dir / "metrics.csv"
    save_metrics(metrics, json_path)
    save_metrics(metrics, csv_path)

    result = {
        "npz": str(npz_path),
        "metrics_json": str(json_path),
        "metrics_csv": str(csv_path),
        "overall": metrics["overall"],
        "threshold_dbz": threshold_dbz,
    }

    if plot:
        if "inputs" not in payload:
            raise KeyError("绘图需要 npz 中的 `inputs` 字段")
        from visualize.plot_results import make_result_figures

        figures_dir = output_dir / "figures"
        paths = make_result_figures(
            payload["inputs"],
            payload["targets"],
            payload["predictions"],
            metrics,
            figures_dir,
            sample_index=sample_index,
        )
        result["figures"] = paths

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="从 predictions.npz 运行统一评测")
    parser.add_argument("--npz", required=True, help="predictions.npz 路径")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="指标与图片输出目录（默认与 npz 同目录）",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=20.0,
        help="回波阈值（dBZ），默认 20.0，与评测说明 v0.1 一致",
    )
    parser.add_argument("--plot", action="store_true", help="同时生成结果图")
    parser.add_argument("--sample-index", type=int, default=0, help="绘图使用的样本下标")
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
        threshold_dbz=args.threshold,
        plot=args.plot,
        sample_index=args.sample_index,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
