from pathlib import Path
import argparse, json
import numpy as np
from common import load_cfg, p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args()
    root = Path(args.repo_root).resolve()
    cfg = load_cfg(root)
    path = p(root, cfg["handoff_npz"])
    z = np.load(path, allow_pickle=True)
    keys = list(z.files)
    required = ["inputs","targets","metadata_json"]
    missing = [k for k in required if k not in keys]
    if missing:
        raise SystemExit(f"Missing handoff fields: {missing}; got {keys}")

    x = z["inputs"]; y = z["targets"]
    exp = cfg["expected"]
    ok_x = x.ndim == 5 and list(x.shape[1:]) == [exp["tin"],exp["channels"],exp["height"],exp["width"]]
    ok_y = y.ndim == 5 and list(y.shape[1:]) == [exp["tout"],exp["channels"],exp["height"],exp["width"]]
    report = {
        "path": str(path),
        "keys": keys,
        "inputs_shape": list(x.shape),
        "targets_shape": list(y.shape),
        "inputs_dtype": str(x.dtype),
        "targets_dtype": str(y.dtype),
        "inputs_range": [float(np.nanmin(x)), float(np.nanmax(x))],
        "targets_range": [float(np.nanmin(y)), float(np.nanmax(y))],
        "shape_ok": bool(ok_x and ok_y),
        "dtype_ok": str(x.dtype) == "float32" and str(y.dtype) == "float32"
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not (report["shape_ok"] and report["dtype_ok"]):
        raise SystemExit("Handoff validation failed.")

if __name__ == "__main__":
    main()
