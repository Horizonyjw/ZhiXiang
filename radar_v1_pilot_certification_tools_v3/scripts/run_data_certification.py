from pathlib import Path
import argparse, subprocess, sys

STEPS = [
    "01_preflight.py",
    "02_build_manifests.py",
    "03_audit_samples_and_splits.py",
    "04_build_normalization.py",
    "05_train_intensity_statistics.py",
    "06_build_data_manifest.py",
    "07_build_contract.py",
    "08_certify.py",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args()
    here = Path(__file__).resolve().parent
    for script in STEPS:
        cmd = [sys.executable, str(here/script), "--repo-root", args.repo_root]
        print("\n==>", " ".join(cmd))
        subprocess.run(cmd, check=True)
    print("\nAll data certification steps finished.")

if __name__ == "__main__":
    main()
