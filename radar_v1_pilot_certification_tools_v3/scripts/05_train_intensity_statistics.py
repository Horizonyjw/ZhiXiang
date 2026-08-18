from pathlib import Path
import argparse, json
import numpy as np
from PIL import Image
from common import load_cfg, p, ensure_out, json_dump

def iter_files(folder):
    exts = {".png",".jpg",".jpeg",".npy",".npz"}
    for x in sorted(folder.rglob("*")):
        if x.is_file() and x.suffix.lower() in exts:
            yield x

def load_frame(path: Path):
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    elif path.suffix.lower() == ".npz":
        z = np.load(path)
        if len(z.files) != 1:
            raise ValueError(f"Cannot infer frame array from multi-field npz: {path}")
        arr = z[z.files[0]]
    else:
        arr = np.asarray(Image.open(path))
    arr = arr.astype(np.float64)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.max(initial=0) > 1.0:
        arr = arr / 255.0
    return arr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()
    root = Path(args.repo_root).resolve()
    cfg = load_cfg(root)
    out = ensure_out(root, cfg)

    train_dir = p(root, cfg.get("standardized_train_dir", "radar_v1/02_standardized/frames/train"))
    if not train_dir.exists():
        raise SystemExit(f"Standardized train directory does not exist: {train_dir}")

    paths = list(iter_files(train_dir))
    if args.max_frames > 0:
        paths = paths[:args.max_frames]
    if not paths:
        raise SystemExit(f"No standardized frame files found under {train_dir}")

    count = 0
    s1 = 0.0
    s2 = 0.0
    nonzero = 0
    mn, mx = float("inf"), float("-inf")
    hist_bins = np.linspace(0, 1, 21)
    hist_counts = np.zeros(len(hist_bins)-1, dtype=np.int64)
    samples = []

    for path in paths:
        arr = load_frame(path)
        flat = arr.reshape(-1)
        count += flat.size
        s1 += float(flat.sum())
        s2 += float(np.square(flat).sum())
        nonzero += int(np.count_nonzero(flat))
        mn = min(mn, float(flat.min()))
        mx = max(mx, float(flat.max()))
        hist_counts += np.histogram(flat, bins=hist_bins)[0]
        step = max(1, flat.size // 4000)
        samples.append(flat[::step][:4000])

    mean = s1 / count
    var = max(0.0, s2 / count - mean*mean)
    sample = np.concatenate(samples)
    qs = {f"P{q}": float(np.percentile(sample, q)) for q in [50,75,90,95,99]}

    result = {
        "dataset_version": cfg["dataset_version"],
        "source_split": "train",
        "source_directory": str(train_dir),
        "unique_train_frame_files_used": len(paths),
        "total_pixels": int(count),
        "min": mn, "max": mx,
        "mean": mean, "std": var**0.5,
        "nonzero_ratio": nonzero/count,
        "percentiles_approx_from_deterministic_pixel_sample": qs,
        "histogram": {"bin_edges": hist_bins.tolist(), "counts": hist_counts.tolist()},
        "val_used": False,
        "test_used": False,
        "physical_unit": cfg["expected"]["physical_unit"]
    }
    json_dump(out/"train_intensity_statistics.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
