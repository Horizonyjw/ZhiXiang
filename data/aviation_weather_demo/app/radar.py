from __future__ import annotations

import csv
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.models import Dataset, RadarFrame, Sample, Segment

REFERENCE = {
    "source_type": "系统历史雷达拼图 PNG 图像序列",
    "time_start": "2026-01-23 18:00:00",
    "time_end": "2026-02-23 18:00:00",
    "frame_count": 7396,
    "expected_frame_count": 7441,
    "missing_frame_count": 45,
    "missing_rate": 0.00605,
    "segment_count": 14,
    "longest_segment_frames": 3378,
    "interval_minutes": 6,
    "interval_tolerance_minutes": 1,
    "image_format": "PNG",
    "image_mode": "RGBA",
    "original_width": 3100,
    "original_height": 2100,
    "model_height": 352,
    "model_width": 512,
    "input_frames": 5,
    "target_frames": 3,
    "dbz_inversion": False,
}

PATTERNS = [
    (re.compile(r"(?<!\d)(\d{14})(?!\d)"), "%Y%m%d%H%M%S"),
    (re.compile(r"(?<!\d)(\d{8}_\d{6})(?!\d)"), "%Y%m%d_%H%M%S"),
    (re.compile(r"(?<!\d)(\d{12})(?!\d)"), "%Y%m%d%H%M"),
    (re.compile(r"(?<!\d)(\d{8}_\d{4})(?!\d)"), "%Y%m%d_%H%M"),
]

def parse_time(name: str) -> datetime | None:
    # 上传材料没有明确说明时区；保留文件名钟表值，不做时差换算。
    for pattern, fmt in PATTERNS:
        match = pattern.search(name)
        if match:
            try:
                return datetime.strptime(match.group(1), fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                pass
    return None

def _is_continuous(previous: datetime, current: datetime) -> bool:
    gap = (current - previous).total_seconds() / 60.0
    target = REFERENCE["interval_minutes"]
    tolerance = REFERENCE["interval_tolerance_minutes"]
    return target - tolerance <= gap <= target + tolerance

def _canonical_unique_frames(frames: list[RadarFrame]) -> list[RadarFrame]:
    chosen = {}
    for frame in sorted(frames, key=lambda x: (x.observation_time, x.id or 0)):
        chosen.setdefault(frame.observation_time, frame)
    return sorted(chosen.values(), key=lambda x: x.observation_time)

def split_segments(frames: list[RadarFrame]) -> list[list[RadarFrame]]:
    frames = _canonical_unique_frames(frames)
    if not frames:
        return []
    groups = [[frames[0]]]
    for frame in frames[1:]:
        if _is_continuous(groups[-1][-1].observation_time, frame.observation_time):
            groups[-1].append(frame)
        else:
            groups.append([frame])
    return groups

def _write_scan_issues(dataset_id: int, issues: list[dict]) -> Path:
    report_dir = Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"scan_issues_dataset_{dataset_id}.csv"
    fields = ["file_name", "file_path", "issue"]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(issues)
    return path

def scan_folder(db: Session, folder: str, name: str) -> tuple[Dataset, dict]:
    project_root = Path.cwd().resolve()
    allowed_root = (project_root / "data").resolve()
    scan_root = (project_root / folder).resolve()

    if not scan_root.is_relative_to(allowed_root):
        raise ValueError("只允许扫描项目 data 文件夹中的目录。")
    if not scan_root.exists():
        raise ValueError(f"目录不存在：{folder}")

    png_paths = sorted(
        [p for p in scan_root.rglob("*") if p.is_file() and p.suffix.lower() == ".png"]
    )
    if not png_paths:
        raise ValueError("目录中没有 PNG 文件。")

    records = []
    issues = []

    for path in png_paths:
        timestamp = parse_time(path.name)
        if timestamp is None:
            issues.append({
                "file_name": path.name,
                "file_path": path.relative_to(project_root).as_posix(),
                "issue": "unparsed_timestamp",
            })
            continue
        try:
            with Image.open(path) as image:
                width, height = image.size
                mode = image.mode
        except Exception:
            issues.append({
                "file_name": path.name,
                "file_path": path.relative_to(project_root).as_posix(),
                "issue": "unreadable_image",
            })
            continue
        records.append({
            "time": timestamp,
            "path": path,
            "width": width,
            "height": height,
            "mode": mode,
        })

    if not records:
        raise ValueError("没有找到可读取且文件名中可解析时次的 PNG。")

    records.sort(key=lambda x: (x["time"], x["path"].name))
    duplicate_counts = Counter(r["time"] for r in records)
    unique_times = sorted(duplicate_counts.keys())

    total_minutes = (unique_times[-1] - unique_times[0]).total_seconds() / 60.0
    expected = int(round(total_minutes / REFERENCE["interval_minutes"])) + 1
    missing = max(expected - len(unique_times), 0)

    dataset = Dataset(
        name=name,
        description=f"真实雷达 PNG 扫描：{folder}",
        start_time=unique_times[0],
        end_time=unique_times[-1],
        frame_count=len(records),
        expected_count=expected,
        missing_count=missing,
        missing_rate=(missing / expected if expected else 0.0),
        is_demo=False,
    )
    db.add(dataset)
    db.flush()

    frame_objects = []
    previous_time = None
    expected_size = (REFERENCE["original_width"], REFERENCE["original_height"])
    expected_mode = REFERENCE["image_mode"]

    for record in records:
        flags = []
        current_time = record["time"]
        gap = None if previous_time is None else (current_time - previous_time).total_seconds() / 60.0

        if gap is not None:
            target = REFERENCE["interval_minutes"]
            tol = REFERENCE["interval_tolerance_minutes"]
            if not (target - tol <= gap <= target + tol):
                flags.append(f"time_gap_{gap:g}_minutes")
        if duplicate_counts[current_time] > 1:
            flags.append("duplicate_observation_time")
        if (record["width"], record["height"]) != expected_size:
            flags.append("unexpected_image_size")
        if record["mode"] != expected_mode:
            flags.append("unexpected_image_mode")

        frame = RadarFrame(
            dataset_id=dataset.id,
            observation_time=current_time,
            file_name=record["path"].name,
            file_path=record["path"].relative_to(project_root).as_posix(),
            width=record["width"],
            height=record["height"],
            mode=record["mode"],
            gap_minutes=gap,
            qc_status=("warning" if flags else "ok"),
            qc_flags=flags,
            physical_variable="normalized_echo_intensity",
            physical_value_available=False,
        )
        db.add(frame)
        frame_objects.append(frame)
        previous_time = current_time

    db.flush()

    segments = split_segments(frame_objects)
    for group in segments:
        db.add(Segment(
            dataset_id=dataset.id,
            start_time=group[0].observation_time,
            end_time=group[-1].observation_time,
            frame_count=len(group),
        ))

    dataset.segment_count = len(segments)
    db.commit()
    db.refresh(dataset)

    issue_path = _write_scan_issues(dataset.id, issues)

    return dataset, {
        "total_png_files": len(png_paths),
        "usable_readable_timestamped_files": len(records),
        "unique_observation_times": len(unique_times),
        "duplicate_extra_files": len(records) - len(unique_times),
        "unparsed_or_unreadable_files": len(issues),
        "expected_count_from_time_span": expected,
        "missing_count_from_time_span": missing,
        "missing_rate": dataset.missing_rate,
        "segment_count": dataset.segment_count,
        "scan_issue_csv": issue_path.as_posix(),
    }

def build_samples(db: Session, dataset_id: int) -> dict[str, int]:
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        raise ValueError("数据集不存在。")

    db.execute(delete(Sample).where(Sample.dataset_id == dataset_id))

    frames = list(db.scalars(
        select(RadarFrame)
        .where(RadarFrame.dataset_id == dataset_id)
        .order_by(RadarFrame.observation_time, RadarFrame.id)
    ))

    windows = []
    window_size = REFERENCE["input_frames"] + REFERENCE["target_frames"]

    for group in split_segments(frames):
        if len(group) < window_size:
            continue
        for i in range(len(group) - window_size + 1):
            window = group[i:i + window_size]
            windows.append((
                [x.id for x in window[:REFERENCE["input_frames"]]],
                [x.id for x in window[REFERENCE["input_frames"]:]],
                window[0].observation_time,
            ))

    total = len(windows)
    train_end = int(total * 0.70)
    val_end = int(total * 0.90)
    counts = {"train": 0, "validation": 0, "test": 0}

    for i, (inputs, targets, start) in enumerate(windows):
        split = "train" if i < train_end else "validation" if i < val_end else "test"
        db.add(Sample(
            dataset_id=dataset_id,
            split=split,
            start_time=start,
            input_ids=inputs,
            target_ids=targets,
        ))
        counts[split] += 1

    dataset.sample_count = total
    db.commit()
    return counts
