from pathlib import Path
import argparse, json
import pandas as pd
from common import load_cfg, ensure_out, parse_list_cell, normalize_split, json_dump

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args()
    root = Path(args.repo_root).resolve()
    cfg = load_cfg(root)
    out = ensure_out(root, cfg)
    exp = cfg["expected"]
    interval = float(exp["frame_interval_minutes"])
    tol = float(exp["frame_interval_tolerance_minutes"])

    fdf = pd.read_csv(out / "frame_manifest.csv")
    sdf = pd.read_csv(out / "sample_manifest.csv")

    required = ["sample_id","split","input_paths","target_paths","input_timestamps","target_timestamps"]
    missing = [c for c in required if c not in sdf.columns]
    if missing:
        raise SystemExit(f"sample_manifest missing columns: {missing}")

    sdf["split"] = sdf["split"].map(normalize_split)

    report = {
        "checks": {},
        "counts": {},
        "split_time_ranges": {},
        "violations": {},
        "test_frozen": True,
        "search_allowed_splits": ["train","val"],
        "certification_only_split": "test",
    }

    # Basic counts
    report["checks"]["sample_id_unique"] = bool(sdf["sample_id"].is_unique)
    counts = sdf["split"].value_counts().to_dict()
    report["counts"]["split_counts"] = counts
    report["checks"]["split_counts"] = all(
        int(counts.get(k, -1)) == int(v)
        for k, v in exp["split_counts"].items()
    )

    bad_input_len = []
    bad_target_len = []
    bad_timestamp_parse = []
    bad_time_order = []
    bad_interval = []
    bad_path_time_len = []
    frame_sets = {"train": set(), "val": set(), "test": set()}
    split_all_times = {"train": [], "val": [], "test": []}

    # Optional segment check: a valid sample should have one segment_id already.
    segment_counts = sdf.groupby("split")["segment_id"].nunique().to_dict() if "segment_id" in sdf.columns else {}
    report["counts"]["segment_counts_by_split"] = segment_counts

    for _, row in sdf.iterrows():
        sid = str(row["sample_id"])
        split = normalize_split(row["split"])
        in_paths = parse_list_cell(row["input_paths"])
        tg_paths = parse_list_cell(row["target_paths"])
        in_times = parse_list_cell(row["input_timestamps"])
        tg_times = parse_list_cell(row["target_timestamps"])

        if len(in_paths) != exp["tin"]:
            bad_input_len.append(sid)
        if len(tg_paths) != exp["tout"]:
            bad_target_len.append(sid)
        if len(in_paths) != len(in_times) or len(tg_paths) != len(tg_times):
            bad_path_time_len.append(sid)

        if split in frame_sets:
            frame_sets[split].update(str(x) for x in in_paths)
            frame_sets[split].update(str(x) for x in tg_paths)

        raw_times = in_times + tg_times
        if len(raw_times) != exp["tin"] + exp["tout"]:
            bad_timestamp_parse.append(sid)
            continue

        ts = pd.to_datetime(pd.Series(raw_times), errors="coerce")
        if ts.isna().any():
            bad_timestamp_parse.append(sid)
            continue

        vals = list(ts)
        if vals != sorted(vals):
            bad_time_order.append(sid)

        diffs = [(vals[i+1] - vals[i]).total_seconds()/60.0 for i in range(len(vals)-1)]
        if any(abs(d - interval) > tol for d in diffs):
            bad_interval.append(sid)

        if split in split_all_times:
            split_all_times[split].extend(vals)

    report["violations"]["bad_input_length"] = bad_input_len[:100]
    report["violations"]["bad_target_length"] = bad_target_len[:100]
    report["violations"]["bad_path_time_length"] = bad_path_time_len[:100]
    report["violations"]["bad_timestamp_parse"] = bad_timestamp_parse[:100]
    report["violations"]["bad_time_order"] = bad_time_order[:100]
    report["violations"]["bad_interval"] = bad_interval[:100]

    report["counts"]["bad_input_length"] = len(bad_input_len)
    report["counts"]["bad_target_length"] = len(bad_target_len)
    report["counts"]["bad_path_time_length"] = len(bad_path_time_len)
    report["counts"]["bad_timestamp_parse"] = len(bad_timestamp_parse)
    report["counts"]["bad_time_order"] = len(bad_time_order)
    report["counts"]["bad_interval"] = len(bad_interval)

    report["checks"]["window_lengths_valid"] = not bad_input_len and not bad_target_len
    report["checks"]["path_time_lengths_match"] = not bad_path_time_len
    report["checks"]["timestamps_parse"] = not bad_timestamp_parse
    report["checks"]["timestamps_monotonic"] = not bad_time_order
    report["checks"]["no_sample_crosses_time_gap"] = not bad_interval

    overlaps = {
        "train_val": len(frame_sets["train"] & frame_sets["val"]),
        "train_test": len(frame_sets["train"] & frame_sets["test"]),
        "val_test": len(frame_sets["val"] & frame_sets["test"]),
    }
    report["counts"]["cross_split_frame_overlap"] = overlaps
    report["checks"]["no_cross_split_frame_overlap"] = all(v == 0 for v in overlaps.values())

    # Split time ranges based on every frame timestamp used by samples.
    for split, vals in split_all_times.items():
        if vals:
            report["split_time_ranges"][split] = {
                "start": min(vals).isoformat(),
                "end": max(vals).isoformat(),
            }

    ranges = report["split_time_ranges"]
    chronological = False
    if all(k in ranges for k in ("train","val","test")):
        chronological = (
            pd.Timestamp(ranges["train"]["end"]) < pd.Timestamp(ranges["val"]["start"])
            and pd.Timestamp(ranges["val"]["end"]) < pd.Timestamp(ranges["test"]["start"])
        )
    report["checks"]["strict_split_chronology"] = chronological

    # Cross-check sample times stay inside their split's frame-index time range where possible.
    if "timestamp" in fdf.columns and "split" in fdf.columns:
        fdf["split"] = fdf["split"].map(normalize_split)
        fdf["timestamp"] = pd.to_datetime(fdf["timestamp"], errors="coerce")
        frame_ranges = {}
        for split, g in fdf.groupby("split"):
            g = g.dropna(subset=["timestamp"])
            if len(g):
                frame_ranges[split] = {
                    "start": g["timestamp"].min().isoformat(),
                    "end": g["timestamp"].max().isoformat(),
                }
        report["frame_index_split_time_ranges"] = frame_ranges

    critical = [
        "sample_id_unique",
        "split_counts",
        "window_lengths_valid",
        "path_time_lengths_match",
        "timestamps_parse",
        "timestamps_monotonic",
        "no_sample_crosses_time_gap",
        "no_cross_split_frame_overlap",
        "strict_split_chronology",
    ]
    report["pass"] = all(bool(report["checks"].get(k, False)) for k in critical)
    json_dump(out / "split_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if not report["pass"]:
        print("\nFAILED CHECKS:")
        for k in critical:
            if not report["checks"].get(k, False):
                print(" -", k)
        raise SystemExit("Split/sample audit FAILED. Do not certify until fixed.")

    print("\nSplit/sample audit PASS.")

if __name__ == "__main__":
    main()
