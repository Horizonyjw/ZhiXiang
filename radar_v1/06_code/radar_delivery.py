# -*- coding: utf-8 -*-
"""
雷达数据最终交付脚本
================================================
功能：
1. 扫描并清洗全部 PNG
2. 解析文件名中的：中心、产品、资料时次、空间范围
3. 检查尺寸/模式/重复时次
4. 6 分钟 ±1 分钟连续性检查
5. 生成 5→3 监督样本
6. 按固定时间边界划分 train/val/test
7. 建立 SQLite 索引数据库
8. 生成 DataLoader
9. 导出 16 组 handoff_samples.npz
10. 验证对接 NPZ

推荐一次运行：
python radar_delivery.py all --radar-dir "D:\\你的7396张雷达图"

文件名示例：
Z_RADA_C_BABJ_P_ACHN_CREF000_20260123180000_12.2_54.2_73.0_135.0.png
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image


FILENAME_RE = re.compile(
    r"^Z_RADA_C_(?P<center>[^_]+)_P_(?P<product>.+?)_"
    r"(?P<time>\d{14})_"
    r"(?P<v1>-?\d+(?:\.\d+)?)_"
    r"(?P<v2>-?\d+(?:\.\d+)?)_"
    r"(?P<v3>-?\d+(?:\.\d+)?)_"
    r"(?P<v4>-?\d+(?:\.\d+)?)$"
)


def load_config(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_filename(path: Path, cfg: dict):
    m = FILENAME_RE.match(path.stem)
    if not m:
        return None, "文件名不符合 Z_RADA_C_..._YYYYMMDDHHMMSS_四个空间范围值 的格式"

    try:
        dt = datetime.strptime(m.group("time"), "%Y%m%d%H%M%S")
    except ValueError:
        return None, "文件名中的14位资料时次无法解析"

    vals = [float(m.group(f"v{i}")) for i in range(1, 5)]
    order = cfg["filename"]["bbox_order"]
    bbox = dict(zip(order, vals))

    # 基本合法性检查；只检查数值范围，不替代项目内部正式空间定义。
    if "lat_min" in bbox and not (-90 <= bbox["lat_min"] <= 90):
        return None, f"lat_min 超出范围: {bbox['lat_min']}"
    if "lat_max" in bbox and not (-90 <= bbox["lat_max"] <= 90):
        return None, f"lat_max 超出范围: {bbox['lat_max']}"
    if "lon_min" in bbox and not (-180 <= bbox["lon_min"] <= 180):
        return None, f"lon_min 超出范围: {bbox['lon_min']}"
    if "lon_max" in bbox and not (-180 <= bbox["lon_max"] <= 180):
        return None, f"lon_max 超出范围: {bbox['lon_max']}"

    return {
        "center": m.group("center"),
        "product": m.group("product"),
        "timestamp": dt,
        "timestamp_text": dt.strftime("%Y-%m-%d %H:%M:%S"),
        **bbox
    }, ""


def inspect_image(path: Path, cfg: dict, full_verify=True):
    exp = cfg["expected"]
    try:
        with Image.open(path) as img:
            width, height = img.size
            mode = img.mode
            if full_verify:
                img.verify()
    except Exception as e:
        return None, f"PNG 无法正常读取: {type(e).__name__}: {e}"

    if width != exp["width"] or height != exp["height"]:
        return None, f"尺寸异常：期望 {exp['width']}x{exp['height']}，实际 {width}x{height}"
    if mode != exp["mode"]:
        return None, f"模式异常：期望 {exp['mode']}，实际 {mode}"

    return {"width": width, "height": height, "mode": mode}, ""


def assign_split(dt: datetime, cfg: dict):
    train_end = datetime.fromisoformat(cfg["split"]["train_end"])
    val_end = datetime.fromisoformat(cfg["split"]["val_end"])
    if dt <= train_end:
        return "train"
    if dt <= val_end:
        return "val"
    return "test"


def preprocess_png(path: str | Path, cfg: dict):
    """
    输出 float32 [1, 352, 512]，范围 [0,1]

    实现：
    RGBA -> RGB/255 -> 灰度 -> alpha无效区=0 -> resize
    resize 的灰度使用 bilinear；有效区 mask 使用 nearest。
    """
    pc = cfg["preprocess"]
    with Image.open(path) as img:
        rgba = np.asarray(img.convert("RGBA"), dtype=np.uint8)

    rgb = rgba[..., :3].astype(np.float32) / float(pc["rgb_divisor"])
    alpha = rgba[..., 3]
    gray = (
        0.299 * rgb[..., 0]
        + 0.587 * rgb[..., 1]
        + 0.114 * rgb[..., 2]
    ).astype(np.float32)

    valid = alpha > int(pc["alpha_invalid_threshold"])
    gray[~valid] = float(pc["invalid_fill_value"])

    h = int(cfg["task"]["target_height"])
    w = int(cfg["task"]["target_width"])

    gray_small = np.asarray(
        Image.fromarray(gray, mode="F").resize((w, h), Image.Resampling.BILINEAR),
        dtype=np.float32
    ).copy()

    valid_small = np.asarray(
        Image.fromarray((valid.astype(np.uint8) * 255), mode="L").resize(
            (w, h), Image.Resampling.NEAREST
        ),
        dtype=np.uint8
    ) > 0

    gray_small[~valid_small] = float(pc["invalid_fill_value"])
    gray_small = np.clip(gray_small, 0.0, 1.0).astype(np.float32)
    return gray_small[None, ...]


def write_csv(path: Path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def create_database(path: Path, frames, rejected, samples, cfg):
    if path.exists():
        path.unlink()

    con = sqlite3.connect(path)
    con.executescript("""
    CREATE TABLE metadata(
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );

    CREATE TABLE frames(
        frame_id INTEGER PRIMARY KEY,
        filename TEXT NOT NULL,
        relative_path TEXT NOT NULL,
        timestamp TEXT NOT NULL UNIQUE,
        split TEXT NOT NULL,
        segment_id INTEGER NOT NULL,
        center_code TEXT NOT NULL,
        product_code TEXT NOT NULL,
        lat_min REAL,
        lat_max REAL,
        lon_min REAL,
        lon_max REAL,
        width INTEGER NOT NULL,
        height INTEGER NOT NULL,
        mode TEXT NOT NULL
    );

    CREATE TABLE rejected_frames(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        relative_path TEXT NOT NULL,
        reason TEXT NOT NULL
    );

    CREATE TABLE samples(
        sample_id TEXT PRIMARY KEY,
        split TEXT NOT NULL,
        segment_id INTEGER NOT NULL,
        start_time TEXT NOT NULL,
        end_time TEXT NOT NULL,
        input_rel_paths_json TEXT NOT NULL,
        target_rel_paths_json TEXT NOT NULL,
        input_times_json TEXT NOT NULL,
        target_times_json TEXT NOT NULL,
        center_code TEXT NOT NULL,
        product_code TEXT NOT NULL,
        lat_min REAL,
        lat_max REAL,
        lon_min REAL,
        lon_max REAL
    );

    CREATE INDEX idx_frames_time ON frames(timestamp);
    CREATE INDEX idx_frames_split ON frames(split);
    CREATE INDEX idx_samples_split ON samples(split);
    """)

    meta = {
        "data_version": cfg["data_version"],
        "Tin": cfg["task"]["tin"],
        "Tout": cfg["task"]["tout"],
        "C": cfg["task"]["channels"],
        "H": cfg["task"]["target_height"],
        "W": cfg["task"]["target_width"],
        "interval_seconds": cfg["task"]["interval_seconds"],
        "tolerance_seconds": cfg["task"]["tolerance_seconds"],
        "physical_unit": cfg["metadata"]["physical_unit"],
        "time_standard": cfg["metadata"]["time_standard"],
        "cloud_upload": cfg["metadata"]["cloud_upload"]
    }
    con.executemany(
        "INSERT INTO metadata(key,value) VALUES(?,?)",
        [(k, str(v)) for k, v in meta.items()]
    )

    con.executemany("""
        INSERT INTO frames(
            frame_id, filename, relative_path, timestamp, split, segment_id,
            center_code, product_code, lat_min, lat_max, lon_min, lon_max,
            width, height, mode
        ) VALUES(
            :frame_id, :filename, :relative_path, :timestamp, :split, :segment_id,
            :center_code, :product_code, :lat_min, :lat_max, :lon_min, :lon_max,
            :width, :height, :mode
        )
    """, frames)

    con.executemany("""
        INSERT INTO rejected_frames(filename,relative_path,reason)
        VALUES(:filename,:relative_path,:reason)
    """, rejected)

    con.executemany("""
        INSERT INTO samples(
            sample_id, split, segment_id, start_time, end_time,
            input_rel_paths_json, target_rel_paths_json,
            input_times_json, target_times_json,
            center_code, product_code, lat_min, lat_max, lon_min, lon_max
        ) VALUES(
            :sample_id, :split, :segment_id, :start_time, :end_time,
            :input_rel_paths_json, :target_rel_paths_json,
            :input_times_json, :target_times_json,
            :center_code, :product_code, :lat_min, :lat_max, :lon_min, :lon_max
        )
    """, samples)

    con.commit()
    con.close()


def build(radar_dir: str, out_dir: str, config_json: str, fast=False):
    cfg = load_config(config_json)
    radar_root = Path(radar_dir).resolve()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    pngs = sorted(
        p for p in radar_root.rglob("*")
        if p.is_file() and p.suffix.lower() == ".png"
    )

    accepted = []
    rejected = []
    seen_times = {}

    print(f"[1/5] 扫描 PNG：{len(pngs)} 张")
    for i, p in enumerate(pngs, start=1):
        rel = str(p.relative_to(radar_root))

        meta, reason = parse_filename(p, cfg)
        if meta is None:
            rejected.append({
                "filename": p.name,
                "relative_path": rel,
                "reason": reason
            })
            continue

        if meta["timestamp"] in seen_times:
            rejected.append({
                "filename": p.name,
                "relative_path": rel,
                "reason": f"重复资料时次：{meta['timestamp_text']}；已存在 {seen_times[meta['timestamp']]}"
            })
            continue

        img_info, reason = inspect_image(p, cfg, full_verify=not fast)
        if img_info is None:
            rejected.append({
                "filename": p.name,
                "relative_path": rel,
                "reason": reason
            })
            continue

        seen_times[meta["timestamp"]] = p.name
        accepted.append({
            "filename": p.name,
            "relative_path": rel,
            "dt": meta["timestamp"],
            "timestamp": meta["timestamp_text"],
            "center_code": meta["center"],
            "product_code": meta["product"],
            "lat_min": meta.get("lat_min"),
            "lat_max": meta.get("lat_max"),
            "lon_min": meta.get("lon_min"),
            "lon_max": meta.get("lon_max"),
            "width": img_info["width"],
            "height": img_info["height"],
            "mode": img_info["mode"]
        })

        if i % 500 == 0:
            print(f"  已检查 {i}/{len(pngs)}")

    accepted.sort(key=lambda x: x["dt"])

    print("[2/5] 时间连续性检查")
    expected = int(cfg["task"]["interval_seconds"])
    tolerance = int(cfg["task"]["tolerance_seconds"])
    segment_id = 0
    breaks = []
    gap_counter = Counter()
    prev = None

    for frame_id, r in enumerate(accepted):
        if prev is not None:
            delta = int((r["dt"] - prev["dt"]).total_seconds())
            gap_counter[delta] += 1
            if abs(delta - expected) > tolerance:
                segment_id += 1
                breaks.append({
                    "previous_time": prev["timestamp"],
                    "current_time": r["timestamp"],
                    "gap_seconds": delta,
                    "gap_minutes": delta / 60.0
                })

        r["frame_id"] = frame_id
        r["segment_id"] = segment_id
        r["split"] = assign_split(r["dt"], cfg)
        prev = r

    print("[3/5] 构造 5→3 样本")
    tin = int(cfg["task"]["tin"])
    tout = int(cfg["task"]["tout"])
    total = tin + tout

    groups = defaultdict(list)
    for r in accepted:
        groups[(r["split"], r["segment_id"])].append(r)

    samples = []
    sid = 0

    for (split, seg), rows in sorted(groups.items(), key=lambda kv: kv[1][0]["dt"]):
        rows.sort(key=lambda x: x["dt"])

        for i in range(0, len(rows) - total + 1):
            window = rows[i:i + total]

            # 兜底：必须全部保持 6min ±1min
            ok = True
            for a, b in zip(window, window[1:]):
                delta = int((b["dt"] - a["dt"]).total_seconds())
                if abs(delta - expected) > tolerance:
                    ok = False
                    break
            if not ok:
                continue

            xs = window[:tin]
            ys = window[tin:]
            first = window[0]

            samples.append({
                "sample_id": f"S{sid:06d}",
                "split": split,
                "segment_id": seg,
                "start_time": window[0]["timestamp"],
                "end_time": window[-1]["timestamp"],
                "input_rel_paths_json": json.dumps([r["relative_path"] for r in xs], ensure_ascii=False),
                "target_rel_paths_json": json.dumps([r["relative_path"] for r in ys], ensure_ascii=False),
                "input_times_json": json.dumps([r["timestamp"] for r in xs], ensure_ascii=False),
                "target_times_json": json.dumps([r["timestamp"] for r in ys], ensure_ascii=False),
                "center_code": first["center_code"],
                "product_code": first["product_code"],
                "lat_min": first["lat_min"],
                "lat_max": first["lat_max"],
                "lon_min": first["lon_min"],
                "lon_max": first["lon_max"]
            })
            sid += 1

    print("[4/5] 写 CSV / SQLite")
    frame_fields = [
        "frame_id", "filename", "relative_path", "timestamp", "split",
        "segment_id", "center_code", "product_code",
        "lat_min", "lat_max", "lon_min", "lon_max",
        "width", "height", "mode"
    ]
    frames_csv_rows = [
        {k: r[k] for k in frame_fields}
        for r in accepted
    ]

    sample_fields = [
        "sample_id", "split", "segment_id", "start_time", "end_time",
        "input_rel_paths_json", "target_rel_paths_json",
        "input_times_json", "target_times_json",
        "center_code", "product_code",
        "lat_min", "lat_max", "lon_min", "lon_max"
    ]

    write_csv(out / "frames.csv", frames_csv_rows, frame_fields)
    write_csv(out / "rejected_frames.csv", rejected, ["filename", "relative_path", "reason"])
    write_csv(
        out / "continuity_breaks.csv",
        breaks,
        ["previous_time", "current_time", "gap_seconds", "gap_minutes"]
    )
    write_csv(out / "samples.csv", samples, sample_fields)

    create_database(
        out / "radar_dataset.sqlite",
        frames_csv_rows,
        rejected,
        samples,
        cfg
    )

    print("[5/5] 生成质量报告 / 数据卡")
    split_counts = Counter(s["split"] for s in samples)
    center_counts = Counter(r["center_code"] for r in accepted)
    product_counts = Counter(r["product_code"] for r in accepted)
    bbox_counts = Counter(
        (r["lat_min"], r["lat_max"], r["lon_min"], r["lon_max"])
        for r in accepted
    )

    exp_counts = cfg["expected"]["sample_counts"]
    checks = {
        "png_count_match": len(pngs) == cfg["expected"]["png_count"],
        "accepted_all_png": len(accepted) == len(pngs),
        "no_rejected_png": len(rejected) == 0,
        "single_center": len(center_counts) == 1,
        "single_product": len(product_counts) == 1,
        "single_spatial_coverage": len(bbox_counts) == 1,
        "sample_train_match": split_counts.get("train", 0) == exp_counts["train"],
        "sample_val_match": split_counts.get("val", 0) == exp_counts["val"],
        "sample_test_match": split_counts.get("test", 0) == exp_counts["test"],
        "sample_total_match": len(samples) == exp_counts["total"]
    }
    delivery_ready = all(checks.values())

    report = {
        "delivery_ready": delivery_ready,
        "checks": checks,
        "found_png": len(pngs),
        "accepted_png": len(accepted),
        "rejected_png": len(rejected),
        "time_range": [
            accepted[0]["timestamp"],
            accepted[-1]["timestamp"]
        ] if accepted else None,
        "segment_count": max((r["segment_id"] for r in accepted), default=-1) + 1,
        "continuity_break_count": len(breaks),
        "gap_seconds_counts": {
            str(k): v for k, v in sorted(gap_counter.items())
        },
        "center_counts": dict(center_counts),
        "product_counts": dict(product_counts),
        "spatial_coverage_counts": {
            str(k): v for k, v in bbox_counts.items()
        },
        "sample_split_counts": dict(split_counts),
        "sample_total": len(samples),
        "expected_sample_counts": exp_counts,
        "notes": [
            "资料时次时区在现有两份方案中未明确，因此本包保存为资料时次字符串，不擅自标UTC。",
            "当前数值为归一化灰度强度代理量，不是dBZ。",
            "bbox字段顺序按示例数值推断；如项目内部定义不同，只修改config.json的bbox_order。"
        ]
    }
    (out / "quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    data_card = {
        "data_version": cfg["data_version"],
        "raw_filename_example": "Z_RADA_C_BABJ_P_ACHN_CREF000_20260123180000_12.2_54.2_73.0_135.0.png",
        "filename_fields": {
            "center_code": "BABJ",
            "product_code": "ACHN_CREF000",
            "timestamp": "20260123180000",
            "spatial_coverage": cfg["filename"]["bbox_order"]
        },
        "model_interface": {
            "Tin": tin,
            "Tout": tout,
            "C": 1,
            "H": cfg["task"]["target_height"],
            "W": cfg["task"]["target_width"],
            "batch_inputs": "[B,5,1,352,512] float32",
            "batch_targets": "[B,3,1,352,512] float32"
        },
        "time": {
            "interval_seconds": expected,
            "tolerance_seconds": tolerance,
            "missing_frame_rule": "超过容差即断开连续片段，不跨缺帧构造样本",
            "time_standard": cfg["metadata"]["time_standard"]
        },
        "preprocess": {
            "steps": [
                "读取RGBA",
                "RGB除以255归一化到[0,1]",
                "0.299R+0.587G+0.114B转单通道灰度",
                "alpha无效显示区域置0",
                "缩放到352x512"
            ],
            "value_range": [0.0, 1.0],
            "physical_unit": cfg["metadata"]["physical_unit"],
            "normalization_status": "already_normalized"
        },
        "split": {
            "method": "chronological",
            "train_end": cfg["split"]["train_end"],
            "val_end": cfg["split"]["val_end"],
            "no_cross_split_windows": True
        },
        "dataloader": {
            "returns": ["inputs", "targets", "metadata"],
            "dataset_class": "RadarDataset"
        },
        "sharing": {
            "cloud_upload": "allowed",
            "public_git": cfg["metadata"]["public_git"]
        }
    }
    (out / "data_card.json").write_text(
        json.dumps(data_card, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    summary_text = f"""雷达数据交付摘要
================
delivery_ready: {delivery_ready}

PNG:
- found: {len(pngs)}
- accepted: {len(accepted)}
- rejected: {len(rejected)}

samples:
- train: {split_counts.get('train', 0)}
- val: {split_counts.get('val', 0)}
- test: {split_counts.get('test', 0)}
- total: {len(samples)}

model interface:
- inputs  [B,5,1,352,512] float32
- targets [B,3,1,352,512] float32

value:
- normalized grayscale [0,1]
- NOT dBZ

请优先查看：
1. quality_report.json
2. data_card.json
3. rejected_frames.csv
4. continuity_breaks.csv
"""
    (out / "交付摘要.txt").write_text(summary_text, encoding="utf-8")

    print("\n=== 完成 ===")
    print("delivery_ready:", delivery_ready)
    print("samples:", dict(split_counts), "total=", len(samples))
    print("输出目录:", out.resolve())
    return report


class RadarDataset:
    """
    PyTorch Dataset
    单样本：
      inputs  [5,1,352,512]
      targets [3,1,352,512]
      metadata dict

    DataLoader后：
      inputs  [B,5,1,352,512]
      targets [B,3,1,352,512]
    """
    def __init__(self, samples_csv, radar_root, config_json="config.json", split=None):
        try:
            import torch
        except ImportError as e:
            raise ImportError("当前Python环境没有torch，请进入项目统一PyTorch环境后再运行DataLoader。") from e

        self.torch = torch
        self.cfg = load_config(config_json)
        self.radar_root = Path(radar_root)

        with Path(samples_csv).open("r", newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))

        if split:
            rows = [r for r in rows if r["split"] == split]

        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]

        input_rel = json.loads(r["input_rel_paths_json"])
        target_rel = json.loads(r["target_rel_paths_json"])

        x = np.stack(
            [preprocess_png(self.radar_root / p, self.cfg) for p in input_rel],
            axis=0
        ).astype(np.float32)

        y = np.stack(
            [preprocess_png(self.radar_root / p, self.cfg) for p in target_rel],
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
            "lat_min": float(r["lat_min"]),
            "lat_max": float(r["lat_max"]),
            "lon_min": float(r["lon_min"]),
            "lon_max": float(r["lon_max"]),
            "data_version": self.cfg["data_version"],
            "time_standard": self.cfg["metadata"]["time_standard"]
        }

        return self.torch.from_numpy(x), self.torch.from_numpy(y), metadata


def export_handoff(samples_csv, radar_root, out_npz, config_json, split="train", n=16):
    cfg = load_config(config_json)

    with Path(samples_csv).open("r", newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split]

    rows = rows[:n]
    if not rows:
        raise RuntimeError(f"split={split} 没有可导出样本")

    xs, ys, metas = [], [], []

    for r in rows:
        input_rel = json.loads(r["input_rel_paths_json"])
        target_rel = json.loads(r["target_rel_paths_json"])

        x = np.stack(
            [preprocess_png(Path(radar_root) / p, cfg) for p in input_rel],
            axis=0
        ).astype(np.float32)

        y = np.stack(
            [preprocess_png(Path(radar_root) / p, cfg) for p in target_rel],
            axis=0
        ).astype(np.float32)

        xs.append(x)
        ys.append(y)

        metas.append(json.dumps({
            "sample_id": r["sample_id"],
            "split": r["split"],
            "segment_id": int(r["segment_id"]),
            "input_times": json.loads(r["input_times_json"]),
            "target_times": json.loads(r["target_times_json"]),
            "center_code": r["center_code"],
            "product_code": r["product_code"],
            "lat_min": float(r["lat_min"]),
            "lat_max": float(r["lat_max"]),
            "lon_min": float(r["lon_min"]),
            "lon_max": float(r["lon_max"]),
            "data_version": cfg["data_version"],
            "value_range": [0.0, 1.0],
            "physical_unit": cfg["metadata"]["physical_unit"],
            "time_standard": cfg["metadata"]["time_standard"]
        }, ensure_ascii=False))

    inputs = np.stack(xs).astype(np.float32)
    targets = np.stack(ys).astype(np.float32)

    Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        inputs=inputs,
        targets=targets,
        metadata_json=np.asarray(metas)
    )

    print("已导出:", out_npz)
    print("inputs :", inputs.shape, inputs.dtype)
    print("targets:", targets.shape, targets.dtype)


def validate_handoff(npz_path, config_json):
    cfg = load_config(config_json)
    d = np.load(npz_path, allow_pickle=False)

    required = {"inputs", "targets", "metadata_json"}
    missing = required - set(d.files)
    if missing:
        raise RuntimeError(f"缺少字段: {sorted(missing)}")

    x = d["inputs"]
    y = d["targets"]
    m = d["metadata_json"]

    expected_x = (
        cfg["task"]["tin"],
        cfg["task"]["channels"],
        cfg["task"]["target_height"],
        cfg["task"]["target_width"]
    )
    expected_y = (
        cfg["task"]["tout"],
        cfg["task"]["channels"],
        cfg["task"]["target_height"],
        cfg["task"]["target_width"]
    )

    if x.ndim != 5 or tuple(x.shape[1:]) != expected_x:
        raise RuntimeError(f"inputs维度错误: {x.shape}；预期 [N,{','.join(map(str,expected_x))}]")
    if y.ndim != 5 or tuple(y.shape[1:]) != expected_y:
        raise RuntimeError(f"targets维度错误: {y.shape}；预期 [N,{','.join(map(str,expected_y))}]")
    if x.dtype != np.float32 or y.dtype != np.float32:
        raise RuntimeError("inputs/targets 必须是 float32")
    if x.shape[0] != y.shape[0] or x.shape[0] != len(m):
        raise RuntimeError("N维度与metadata数量不一致")

    if len(m):
        json.loads(str(m[0]))

    print("通过：handoff_samples.npz 满足数据侧对接格式")
    print("inputs :", x.shape, x.dtype)
    print("targets:", y.shape, y.dtype)
    print("metadata_json:", m.shape)


def check_loader(samples_csv, radar_root, config_json):
    try:
        from torch.utils.data import DataLoader
    except ImportError as e:
        raise ImportError("请进入项目统一PyTorch环境后运行该命令。") from e

    ds = RadarDataset(
        samples_csv=samples_csv,
        radar_root=radar_root,
        config_json=config_json,
        split="train"
    )

    loader = DataLoader(
        ds,
        batch_size=2,
        shuffle=False,
        num_workers=0
    )

    x, y, metadata = next(iter(loader))
    print("DataLoader通过")
    print("inputs :", tuple(x.shape), x.dtype)
    print("targets:", tuple(y.shape), y.dtype)
    print("metadata keys:", list(metadata.keys()))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build")
    p.add_argument("--radar-dir", required=True)
    p.add_argument("--out-dir", default="delivery_output")
    p.add_argument("--config", default="config.json")
    p.add_argument("--fast", action="store_true", help="跳过PNG完整verify，仅用于首次快速测试")

    p = sub.add_parser("export")
    p.add_argument("--samples-csv", required=True)
    p.add_argument("--radar-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--config", default="config.json")
    p.add_argument("--split", default="train")
    p.add_argument("--n", type=int, default=16)

    p = sub.add_parser("validate")
    p.add_argument("--npz", required=True)
    p.add_argument("--config", default="config.json")

    p = sub.add_parser("check-loader")
    p.add_argument("--samples-csv", required=True)
    p.add_argument("--radar-dir", required=True)
    p.add_argument("--config", default="config.json")

    p = sub.add_parser("all")
    p.add_argument("--radar-dir", required=True)
    p.add_argument("--out-dir", default="delivery_output")
    p.add_argument("--config", default="config.json")
    p.add_argument("--sample-n", type=int, default=16)
    p.add_argument("--fast", action="store_true")

    args = ap.parse_args()

    if args.cmd == "build":
        build(args.radar_dir, args.out_dir, args.config, args.fast)

    elif args.cmd == "export":
        export_handoff(
            args.samples_csv,
            args.radar_dir,
            args.out,
            args.config,
            args.split,
            args.n
        )

    elif args.cmd == "validate":
        validate_handoff(args.npz, args.config)

    elif args.cmd == "check-loader":
        check_loader(
            args.samples_csv,
            args.radar_dir,
            args.config
        )

    elif args.cmd == "all":
        build(
            args.radar_dir,
            args.out_dir,
            args.config,
            args.fast
        )
        samples_csv = str(Path(args.out_dir) / "samples.csv")
        handoff_npz = str(Path(args.out_dir) / "handoff_samples.npz")

        export_handoff(
            samples_csv,
            args.radar_dir,
            handoff_npz,
            args.config,
            "train",
            args.sample_n
        )
        validate_handoff(handoff_npz, args.config)

        print("\n全部完成。")
        print("下一步：")
        print(f'python radar_delivery.py check-loader --samples-csv "{samples_csv}" --radar-dir "{args.radar_dir}" --config "{args.config}"')


if __name__ == "__main__":
    main()
