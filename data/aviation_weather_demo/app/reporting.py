from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Dataset, RadarFrame, Sample, Segment
from app.radar import REFERENCE

def _frame_map(db: Session, dataset_id: int) -> dict[int, RadarFrame]:
    frames = list(db.scalars(
        select(RadarFrame)
        .where(RadarFrame.dataset_id == dataset_id)
        .order_by(RadarFrame.id)
    ))
    return {f.id: f for f in frames}

def export_sample_manifest(db: Session, dataset_id: int, report_dir: Path) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    frame_map = _frame_map(db, dataset_id)
    samples = list(db.scalars(
        select(Sample).where(Sample.dataset_id == dataset_id).order_by(Sample.id)
    ))
    path = report_dir / f"sample_manifest_dataset_{dataset_id}.csv"
    fields = [
        "sample_id", "split", "start_time",
        "input_paths_json", "target_paths_json",
        "input_times_json", "target_times_json",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            input_frames = [frame_map[i] for i in sample.input_ids]
            target_frames = [frame_map[i] for i in sample.target_ids]
            writer.writerow({
                "sample_id": sample.id,
                "split": sample.split,
                "start_time": sample.start_time.isoformat(),
                "input_paths_json": json.dumps([x.file_path for x in input_frames], ensure_ascii=False),
                "target_paths_json": json.dumps([x.file_path for x in target_frames], ensure_ascii=False),
                "input_times_json": json.dumps([x.observation_time.isoformat() for x in input_frames], ensure_ascii=False),
                "target_times_json": json.dumps([x.observation_time.isoformat() for x in target_frames], ensure_ascii=False),
            })
    return path

def export_data_issues(db: Session, dataset_id: int, report_dir: Path) -> Path:
    frames = list(db.scalars(
        select(RadarFrame)
        .where(RadarFrame.dataset_id == dataset_id)
        .order_by(RadarFrame.observation_time, RadarFrame.id)
    ))
    path = report_dir / f"data_issues_dataset_{dataset_id}.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=["frame_id", "time", "file_name", "file_path", "qc_status", "qc_flags"]
        )
        writer.writeheader()
        for frame in frames:
            if frame.qc_status != "ok":
                writer.writerow({
                    "frame_id": frame.id,
                    "time": frame.observation_time.isoformat(),
                    "file_name": frame.file_name,
                    "file_path": frame.file_path,
                    "qc_status": frame.qc_status,
                    "qc_flags": json.dumps(frame.qc_flags, ensure_ascii=False),
                })
    return path

def export_quality_summary(db: Session, dataset_id: int, report_dir: Path) -> Path:
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        raise ValueError("数据集不存在。")
    frames = list(db.scalars(select(RadarFrame).where(RadarFrame.dataset_id == dataset_id)))
    segments = list(db.scalars(select(Segment).where(Segment.dataset_id == dataset_id)))
    samples = list(db.scalars(select(Sample).where(Sample.dataset_id == dataset_id)))

    payload = {
        "source_reference_from_uploaded_files": REFERENCE,
        "actual_scan_and_build": {
            "dataset_id": dataset.id,
            "dataset_name": dataset.name,
            "start_time": dataset.start_time.isoformat() if dataset.start_time else None,
            "end_time": dataset.end_time.isoformat() if dataset.end_time else None,
            "frame_count": dataset.frame_count,
            "expected_count": dataset.expected_count,
            "missing_count": dataset.missing_count,
            "missing_rate": dataset.missing_rate,
            "segment_count": dataset.segment_count,
            "sample_count": dataset.sample_count,
            "warning_frames": sum(1 for x in frames if x.qc_status != "ok"),
            "duplicate_time_frames": sum(1 for x in frames if "duplicate_observation_time" in (x.qc_flags or [])),
            "unexpected_size_frames": sum(1 for x in frames if "unexpected_image_size" in (x.qc_flags or [])),
            "unexpected_mode_frames": sum(1 for x in frames if "unexpected_image_mode" in (x.qc_flags or [])),
            "train_samples": sum(1 for x in samples if x.split == "train"),
            "validation_samples": sum(1 for x in samples if x.split == "validation"),
            "test_samples": sum(1 for x in samples if x.split == "test"),
            "longest_actual_segment_frames": max((x.frame_count for x in segments), default=0),
        },
        "note": "actual_scan_and_build 来自本地真实 PNG；reference 只用于和上传材料对照。",
    }
    path = report_dir / f"quality_summary_dataset_{dataset_id}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path

def export_continuous_sequence_preview(db: Session, dataset_id: int, report_dir: Path):
    sample = db.scalar(
        select(Sample).where(Sample.dataset_id == dataset_id).order_by(Sample.id).limit(1)
    )
    if not sample:
        return None

    frame_map = _frame_map(db, dataset_id)
    frames = [frame_map[i] for i in (sample.input_ids + sample.target_ids)]
    thumbs = []

    for frame in frames:
        path = Path(frame.file_path)
        if not path.exists():
            return None
        with Image.open(path) as image:
            rgba = image.convert("RGBA")
            rgba.thumbnail((320, 220))
            canvas = Image.new("RGB", (340, 270), "white")
            rgb = Image.new("RGB", rgba.size, "white")
            rgb.paste(rgba, mask=rgba.getchannel("A"))
            x = (340 - rgb.width) // 2
            y = 8 + (220 - rgb.height) // 2
            canvas.paste(rgb, (x, y))
            draw = ImageDraw.Draw(canvas)
            draw.text((10, 235), frame.observation_time.strftime("%Y-%m-%d %H:%M"), fill="black")
            thumbs.append(canvas)

    sheet = Image.new("RGB", (340 * 4, 270 * 2), "white")
    for idx, img in enumerate(thumbs):
        sheet.paste(img, ((idx % 4) * 340, (idx // 4) * 270))

    out = report_dir / f"continuous_sequence_dataset_{dataset_id}.png"
    sheet.save(out)
    return out

def export_all_reports(db: Session, dataset_id: int) -> dict:
    report_dir = Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest = export_sample_manifest(db, dataset_id, report_dir)
    issues = export_data_issues(db, dataset_id, report_dir)
    quality = export_quality_summary(db, dataset_id, report_dir)
    preview = export_continuous_sequence_preview(db, dataset_id, report_dir)
    return {
        "sample_manifest": manifest.as_posix(),
        "data_issues": issues.as_posix(),
        "quality_summary": quality.as_posix(),
        "continuous_sequence_preview": preview.as_posix() if preview else None,
    }
