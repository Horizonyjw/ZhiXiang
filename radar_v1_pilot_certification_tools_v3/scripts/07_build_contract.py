from pathlib import Path
import argparse, json
from common import load_cfg, ensure_out, json_dump

def readj(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args()
    root = Path(args.repo_root).resolve()
    cfg = load_cfg(root)
    out = ensure_out(root, cfg)
    exp = cfg["expected"]
    audit = readj(out/"split_audit.json")

    contract = {
        "dataset_id": cfg["dataset_id"],
        "dataset_version": cfg["dataset_version"],
        "contract_status": "pending_certification",
        "frame_count": exp["frame_count"],
        "sample_count": exp["sample_count"],
        "split": {
            "strategy": "chronological",
            "counts": exp["split_counts"],
            "time_ranges": audit.get("split_time_ranges", {}),
            "test_frozen": True,
            "search_allowed_splits": ["train", "val"],
            "certification_only_split": "test"
        },
        "tensor_contract": {
            "inputs": [None, exp["tin"], exp["channels"], exp["height"], exp["width"]],
            "targets": [None, exp["tout"], exp["channels"], exp["height"], exp["width"]],
            "dtype": exp["dtype"],
            "value_range": exp["value_range"],
            "frame_interval_minutes": exp["frame_interval_minutes"]
        },
        "physical_unit": exp["physical_unit"],
        "formal_physical_claims": False,
        "normalization_file": "normalization.json",
        "frame_manifest": "frame_manifest.csv",
        "sample_manifest": "sample_manifest.csv",
        "train_statistics": "train_intensity_statistics.json",
        "data_manifest": "data_manifest.json",
        "known_limitations": [
            "Current values are normalized grayscale proxies rather than physical dBZ.",
            "PNG-to-dBZ mapping is not established in radar_v1.",
            "Formal CRS / georeferencing details remain to be independently verified.",
            "Evaluation activity/event thresholds must be frozen by the evaluation team using train-only information before formal pilot evaluation."
        ]
    }
    json_dump(out / "radar_v1_pilot_data_contract.json", contract)
    print(json.dumps(contract, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
