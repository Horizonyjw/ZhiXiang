from pathlib import Path
import argparse, json
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    args = ap.parse_args()
    path = Path(args.predictions).resolve()
    z = np.load(path, allow_pickle=True)
    req = ["inputs","targets","predictions","metadata_json"]
    missing = [k for k in req if k not in z.files]
    if missing:
        raise SystemExit(f"Missing fields: {missing}; got {list(z.files)}")

    x,y,p = z["inputs"],z["targets"],z["predictions"]
    checks = {
        "inputs_shape": list(x.shape),
        "targets_shape": list(y.shape),
        "predictions_shape": list(p.shape),
        "N_aligned": x.shape[0] == y.shape[0] == p.shape[0],
        "target_prediction_shape_aligned": y.shape == p.shape,
        "prediction_rank_5": p.ndim == 5,
        "prediction_core_shape": list(p.shape[1:]) == [3,1,352,512],
        "prediction_dtype": str(p.dtype),
        "finite_predictions": bool(np.isfinite(p).all())
    }
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    critical = [
        checks["N_aligned"], checks["target_prediction_shape_aligned"],
        checks["prediction_rank_5"], checks["prediction_core_shape"],
        checks["finite_predictions"]
    ]
    if not all(critical):
        raise SystemExit("predictions.npz contract validation FAILED.")
    print("predictions.npz contract PASS.")

if __name__ == "__main__":
    main()
