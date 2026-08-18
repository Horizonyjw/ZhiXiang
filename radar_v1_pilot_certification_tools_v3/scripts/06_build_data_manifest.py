from pathlib import Path
import argparse, json
from common import load_cfg, ensure_out, p, sha256_file, json_dump

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args()
    root = Path(args.repo_root).resolve()
    cfg = load_cfg(root)
    out = ensure_out(root, cfg)

    candidates = [
        cfg["standardized_frames_csv"],
        cfg["standardized_samples_csv"],
        cfg["raw_frames_csv"],
        cfg["raw_samples_csv"],
        cfg["continuity_breaks_csv"],
        cfg["quality_report_json"],
        cfg["data_card_json"],
        cfg["handoff_npz"],
        cfg["output_dir"] + "/frame_manifest.csv",
        cfg["output_dir"] + "/sample_manifest.csv",
        cfg["output_dir"] + "/split_audit.json",
        cfg["output_dir"] + "/normalization.json",
        cfg["output_dir"] + "/train_intensity_statistics.json",
    ]
    records = []
    total = 0
    for rel in candidates:
        path = p(root, rel)
        if path.exists() and path.is_file():
            size = path.stat().st_size
            total += size
            records.append({
                "path": str(path.relative_to(root)),
                "size_bytes": size,
                "sha256": sha256_file(path)
            })

    result = {
        "dataset_id": cfg["dataset_id"],
        "dataset_version": cfg["dataset_version"],
        "frame_count": cfg["expected"]["frame_count"],
        "sample_count": cfg["expected"]["sample_count"],
        "split_counts": cfg["expected"]["split_counts"],
        "manifested_file_count": len(records),
        "manifested_total_size_bytes": total,
        "files": records,
        "note": "Large standardized/raw radar image payloads may live outside GitHub. This manifest hashes every certification-relevant file accessible under the repository paths listed in pilot_config.json."
    }
    json_dump(out / "data_manifest.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
