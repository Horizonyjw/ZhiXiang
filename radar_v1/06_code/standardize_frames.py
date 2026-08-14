# -*- coding: utf-8 -*-
"""
把 7396 张原始 RGBA PNG 批量规范化为模型直接可读的 float32 NPY。

输入：
  delivery_output/frames.csv
  delivery_output/samples.csv
  原始雷达 PNG 根目录
  config.json
  radar_delivery.py（复用完全相同的 preprocess_png）

输出：
  standardized_output/
    frames/train/*.npy
    frames/val/*.npy
    frames/test/*.npy
    standardized_frames.csv
    samples_standardized.csv
    standardization_report.json

运行：
python standardize_frames.py ^
  --radar-dir "D:\BaiduNetdiskDownload\气象大模型数据（中南）-20260224\气象雷达拼图（UTC）" ^
  --frames-csv "delivery_output\frames.csv" ^
  --samples-csv "delivery_output\samples.csv" ^
  --config "config.json" ^
  --out-dir "standardized_output"
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from radar_delivery import load_config, preprocess_png


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radar-dir", required=True)
    ap.add_argument("--frames-csv", required=True)
    ap.add_argument("--samples-csv", required=True)
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--out-dir", default="standardized_output")
    args = ap.parse_args()

    cfg = load_config(args.config)
    radar_root = Path(args.radar_dir).resolve()
    out_root = Path(args.out_dir)
    frame_out_root = out_root / "frames"
    out_root.mkdir(parents=True, exist_ok=True)

    frames = read_csv(args.frames_csv)
    samples = read_csv(args.samples_csv)

    expected_shape = (
        int(cfg["task"]["channels"]),
        int(cfg["task"]["target_height"]),
        int(cfg["task"]["target_width"])
    )

    mapping = {}
    standardized_rows = []
    split_counts = Counter()
    bad = []

    print(f"开始规范化 {len(frames)} 张图片 ...")

    for i, r in enumerate(frames, start=1):
        raw_rel = r["relative_path"]
        split = r["split"]
        raw_path = radar_root / raw_rel

        # 保留原始文件名主干，改为 .npy；按 split 分目录
        out_rel = Path("frames") / split / (Path(raw_rel).stem + ".npy")
        out_path = out_root / out_rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            arr = preprocess_png(raw_path, cfg)
            if arr.shape != expected_shape:
                raise ValueError(f"shape={arr.shape}, expected={expected_shape}")
            if arr.dtype != np.float32:
                raise ValueError(f"dtype={arr.dtype}, expected=float32")
            if not np.isfinite(arr).all():
                raise ValueError("存在 NaN/Inf")
            if float(arr.min()) < 0.0 or float(arr.max()) > 1.0:
                raise ValueError(f"值范围异常 [{arr.min()}, {arr.max()}]")

            np.save(out_path, arr, allow_pickle=False)
        except Exception as e:
            bad.append({
                "frame_id": r["frame_id"],
                "relative_path": raw_rel,
                "reason": str(e)
            })
            continue

        mapping[raw_rel] = str(out_rel).replace("\\", "/")
        split_counts[split] += 1

        standardized_rows.append({
            "frame_id": r["frame_id"],
            "timestamp": r["timestamp"],
            "split": split,
            "raw_relative_path": raw_rel,
            "standardized_relative_path": str(out_rel).replace("\\", "/"),
            "shape": "1x352x512",
            "dtype": "float32",
            "value_min": f"{float(arr.min()):.8f}",
            "value_max": f"{float(arr.max()):.8f}",
            "center_code": r["center_code"],
            "product_code": r["product_code"],
            "lat_min": r["lat_min"],
            "lat_max": r["lat_max"],
            "lon_min": r["lon_min"],
            "lon_max": r["lon_max"]
        })

        if i % 250 == 0 or i == len(frames):
            print(f"  {i}/{len(frames)}")

    # 生成标准化样本索引：样本直接指向 .npy
    std_samples = []
    for r in samples:
        input_raw = json.loads(r["input_rel_paths_json"])
        target_raw = json.loads(r["target_rel_paths_json"])

        missing = [p for p in input_raw + target_raw if p not in mapping]
        if missing:
            continue

        std_samples.append({
            "sample_id": r["sample_id"],
            "split": r["split"],
            "segment_id": r["segment_id"],
            "start_time": r["start_time"],
            "end_time": r["end_time"],
            "input_std_paths_json": json.dumps([mapping[p] for p in input_raw], ensure_ascii=False),
            "target_std_paths_json": json.dumps([mapping[p] for p in target_raw], ensure_ascii=False),
            "input_times_json": r["input_times_json"],
            "target_times_json": r["target_times_json"],
            "center_code": r["center_code"],
            "product_code": r["product_code"],
            "lat_min": r["lat_min"],
            "lat_max": r["lat_max"],
            "lon_min": r["lon_min"],
            "lon_max": r["lon_max"]
        })

    write_csv(
        out_root / "standardized_frames.csv",
        standardized_rows,
        [
            "frame_id", "timestamp", "split",
            "raw_relative_path", "standardized_relative_path",
            "shape", "dtype", "value_min", "value_max",
            "center_code", "product_code",
            "lat_min", "lat_max", "lon_min", "lon_max"
        ]
    )

    write_csv(
        out_root / "samples_standardized.csv",
        std_samples,
        [
            "sample_id", "split", "segment_id", "start_time", "end_time",
            "input_std_paths_json", "target_std_paths_json",
            "input_times_json", "target_times_json",
            "center_code", "product_code",
            "lat_min", "lat_max", "lon_min", "lon_max"
        ]
    )

    if bad:
        write_csv(
            out_root / "standardization_rejected.csv",
            bad,
            ["frame_id", "relative_path", "reason"]
        )

    sample_counts = Counter(r["split"] for r in std_samples)
    exp_frames = int(cfg["expected"]["png_count"])
    exp_samples = cfg["expected"]["sample_counts"]

    checks = {
        "standardized_frame_count_match": len(standardized_rows) == exp_frames,
        "no_standardization_rejects": len(bad) == 0,
        "train_sample_match": sample_counts.get("train", 0) == exp_samples["train"],
        "val_sample_match": sample_counts.get("val", 0) == exp_samples["val"],
        "test_sample_match": sample_counts.get("test", 0) == exp_samples["test"],
        "total_sample_match": len(std_samples) == exp_samples["total"]
    }

    report = {
        "standardization_ready": all(checks.values()),
        "checks": checks,
        "standardized_frames": len(standardized_rows),
        "rejected_frames": len(bad),
        "frame_split_counts": dict(split_counts),
        "sample_split_counts": dict(sample_counts),
        "sample_total": len(std_samples),
        "tensor_format": {
            "shape": [1, 352, 512],
            "dtype": "float32",
            "value_range": [0.0, 1.0],
            "physical_unit": cfg["metadata"]["physical_unit"]
        }
    }

    (out_root / "standardization_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("\n完成。")
    print("standardization_ready:", report["standardization_ready"])
    print("standardized_frames:", len(standardized_rows))
    print("samples:", dict(sample_counts), "total=", len(std_samples))
    print("输出目录:", out_root.resolve())


class StandardizedRadarDataset:
    """
    直接读取预先规范化好的 .npy，不再实时处理 PNG。
    单样本：
      inputs  [5,1,352,512]
      targets [3,1,352,512]
    """
    def __init__(self, samples_csv, standardized_root, split=None):
        try:
            import torch
        except ImportError as e:
            raise ImportError("请在项目统一 PyTorch 环境中使用该 Dataset。") from e

        self.torch = torch
        self.root = Path(standardized_root)
        rows = read_csv(samples_csv)
        self.rows = [r for r in rows if not split or r["split"] == split]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        xpaths = json.loads(r["input_std_paths_json"])
        ypaths = json.loads(r["target_std_paths_json"])

        x = np.stack(
            [np.load(self.root / p, allow_pickle=False) for p in xpaths],
            axis=0
        ).astype(np.float32)

        y = np.stack(
            [np.load(self.root / p, allow_pickle=False) for p in ypaths],
            axis=0
        ).astype(np.float32)

        metadata = {
            "sample_id": r["sample_id"],
            "split": r["split"],
            "segment_id": int(r["segment_id"]),
            "input_times": json.loads(r["input_times_json"]),
            "target_times": json.loads(r["target_times_json"]),
            "center_code": r["center_code"],
            "product_code": r["product_code"],
            "bbox": [
                float(r["lat_min"]), float(r["lat_max"]),
                float(r["lon_min"]), float(r["lon_max"])
            ]
        }

        return self.torch.from_numpy(x), self.torch.from_numpy(y), metadata


if __name__ == "__main__":
    main()
