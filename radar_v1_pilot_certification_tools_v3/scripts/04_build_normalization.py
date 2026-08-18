from pathlib import Path
import argparse
from common import load_cfg, ensure_out, json_dump

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args()
    root = Path(args.repo_root).resolve()
    cfg = load_cfg(root)
    out = ensure_out(root, cfg)

    n = cfg["normalization"].copy()
    n.update({
        "dataset_id": cfg["dataset_id"],
        "dataset_version": cfg["dataset_version"],
        "output_dtype": cfg["expected"]["dtype"],
        "output_range": cfg["expected"]["value_range"],
        "physical_unit": cfg["expected"]["physical_unit"],
        "formal_physical_claims": False,
        "note": "This file documents the existing radar_v1 preprocessing. It does not reprocess data."
    })
    json_dump(out / "normalization.json", n)
    print(f"Wrote {out/'normalization.json'}")

if __name__ == "__main__":
    main()
