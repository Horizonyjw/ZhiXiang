from pathlib import Path
import argparse, json
from common import load_cfg, p, ensure_out, read_csv_if_exists, find_col, normalize_split, json_dump

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args()
    root = Path(args.repo_root).resolve()
    cfg = load_cfg(root)
    out = ensure_out(root, cfg)
    exp = cfg["expected"]

    # Canonical truth = 03_index, not 02_standardized.
    canonical_frame = p(root, cfg.get("canonical_frame_index", cfg["raw_frames_csv"]))
    canonical_sample = p(root, cfg.get("canonical_sample_index", cfg["raw_samples_csv"]))
    std_frame = p(root, cfg["standardized_frames_csv"])
    std_sample = p(root, cfg["standardized_samples_csv"])
    breaks_path = p(root, cfg["continuity_breaks_csv"])

    required = [
        canonical_frame, canonical_sample, breaks_path,
        p(root, cfg["quality_report_json"]),
        p(root, cfg["data_card_json"]),
        p(root, cfg["handoff_npz"]),
    ]

    result = {
        "canonical_sources": {
            "frames": str(canonical_frame.relative_to(root)) if canonical_frame.exists() else str(canonical_frame),
            "samples": str(canonical_sample.relative_to(root)) if canonical_sample.exists() else str(canonical_sample),
        },
        "checks": {},
        "observed": {},
        "standardized_coverage_info": {},
    }

    fdf = read_csv_if_exists(canonical_frame)
    sdf = read_csv_if_exists(canonical_sample)
    breaks = read_csv_if_exists(breaks_path)
    sfdf = read_csv_if_exists(std_frame)
    ssdf = read_csv_if_exists(std_sample)

    result["checks"]["all_canonical_required_paths_exist"] = all(x.exists() for x in required)

    if fdf is None or sdf is None:
        result["pass"] = False
        json_dump(out/"preflight_report.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit("Canonical 03_index files are missing.")

    result["observed"]["canonical_frame_rows"] = len(fdf)
    result["observed"]["canonical_sample_rows"] = len(sdf)
    result["checks"]["frame_count"] = len(fdf) == exp["frame_count"]
    result["checks"]["sample_count"] = len(sdf) == exp["sample_count"]

    split_col = find_col(sdf, ["split", "dataset_split"])
    if split_col:
        counts = sdf[split_col].map(normalize_split).value_counts().to_dict()
        result["observed"]["canonical_split_counts"] = counts
        result["checks"]["split_counts"] = all(
            int(counts.get(k, -1)) == int(v)
            for k, v in exp["split_counts"].items()
        )
    else:
        result["checks"]["split_counts"] = False
        result["observed"]["sample_columns"] = list(sdf.columns)

    if breaks is not None:
        result["observed"]["continuity_break_rows"] = len(breaks)
        result["checks"]["continuity_breaks_accessible"] = True
    else:
        result["checks"]["continuity_breaks_accessible"] = False

    # Standardized CSVs are diagnostic only.
    if sfdf is not None:
        result["standardized_coverage_info"]["standardized_frame_rows"] = len(sfdf)
        sc = find_col(sfdf, ["split","dataset_split"])
        if sc:
            result["standardized_coverage_info"]["frame_split_counts"] = (
                sfdf[sc].map(normalize_split).value_counts().to_dict()
            )
        result["standardized_coverage_info"]["frame_columns"] = list(sfdf.columns)
    if ssdf is not None:
        result["standardized_coverage_info"]["standardized_sample_rows"] = len(ssdf)
        sc = find_col(ssdf, ["split","dataset_split"])
        if sc:
            result["standardized_coverage_info"]["sample_split_counts"] = (
                ssdf[sc].map(normalize_split).value_counts().to_dict()
            )
        result["standardized_coverage_info"]["sample_columns"] = list(ssdf.columns)

    critical = [
        "all_canonical_required_paths_exist",
        "frame_count",
        "sample_count",
        "split_counts",
        "continuity_breaks_accessible",
    ]
    result["pass"] = all(result["checks"].get(k, False) for k in critical)
    json_dump(out/"preflight_report.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if not result["pass"]:
        raise SystemExit(
            "Preflight FAILED using canonical 03_index files. "
            "Check 03_index counts/columns; do not use 02_standardized row counts as total dataset counts."
        )
    print("Preflight PASS using canonical 03_index indices.")

if __name__ == "__main__":
    main()
