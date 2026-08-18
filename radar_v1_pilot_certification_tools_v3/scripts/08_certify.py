from pathlib import Path
import argparse, json
from common import load_cfg, ensure_out, json_dump

def readj(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args()
    root = Path(args.repo_root).resolve()
    cfg = load_cfg(root)
    out = ensure_out(root, cfg)

    needed = [
        "preflight_report.json",
        "frame_manifest.csv",
        "sample_manifest.csv",
        "split_audit.json",
        "normalization.json",
        "train_intensity_statistics.json",
        "data_manifest.json",
        "radar_v1_pilot_data_contract.json"
    ]
    exists = {x: (out/x).exists() for x in needed}

    pre = readj(out/"preflight_report.json") or {}
    audit = readj(out/"split_audit.json") or {}
    stats = readj(out/"train_intensity_statistics.json") or {}
    contract = readj(out/"radar_v1_pilot_data_contract.json") or {}

    checks = {
        "all_artifacts_exist": all(exists.values()),
        "preflight_pass": bool(pre.get("pass")),
        "split_audit_pass": bool(audit.get("pass")),
        "train_stats_train_only": stats.get("source_split") == "train" and stats.get("val_used") is False and stats.get("test_used") is False,
        "test_frozen": contract.get("split",{}).get("test_frozen") is True,
        "formal_physical_claims_false": contract.get("formal_physical_claims") is False,
        "physical_unit_proxy_not_dbz": contract.get("physical_unit") == "normalized_grayscale_proxy_NOT_dBZ"
    }
    status = "certified_for_pilot" if all(checks.values()) else "not_certified"
    report = {"contract_status": status, "checks": checks, "artifacts": exists}
    json_dump(out/"certification_report.json", report)

    if contract:
        contract["contract_status"] = status
        json_dump(out/"radar_v1_pilot_data_contract.json", contract)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if status != "certified_for_pilot":
        raise SystemExit("NOT CERTIFIED. Fix failed checks first.")
    print("CERTIFIED FOR PILOT.")

if __name__ == "__main__":
    main()
