from pathlib import Path
import argparse
import pandas as pd
from common import load_cfg, p, ensure_out, find_col, normalize_split

FRAME_CANDIDATES = {
    "frame_id": ["frame_id", "id"],
    "timestamp": ["timestamp", "time", "datetime", "valid_time", "frame_time"],
    "path": ["relative_path", "path", "frame_path", "filepath", "file_path", "raw_path"],
    "split": ["split", "dataset_split"],
    "segment_id": ["segment_id", "segment"],
    "filename": ["filename", "file_name", "name"],
    "center_code": ["center_code"],
    "product_code": ["product_code"],
}
SAMPLE_CANDIDATES = {
    "sample_id": ["sample_id", "id"],
    "split": ["split", "dataset_split"],
    "segment_id": ["segment_id", "segment"],
    "start_time": ["start_time"],
    "end_time": ["end_time"],
    "input_paths": ["input_rel_paths_json", "input_paths", "inputs", "input_files", "x_paths"],
    "target_paths": ["target_rel_paths_json", "target_paths", "targets", "target_files", "y_paths"],
    "input_timestamps": ["input_times_json", "input_timestamps", "input_times", "x_times"],
    "target_timestamps": ["target_times_json", "target_timestamps", "target_times", "y_times"],
    "center_code": ["center_code"],
    "product_code": ["product_code"],
}

def canonicalize(df, candidates):
    out = pd.DataFrame(index=df.index)
    for dst, cands in candidates.items():
        col = find_col(df, cands)
        if col is not None:
            out[dst] = df[col]
    return out

def add_standardized_path(frame_manifest, std_df):
    if std_df is None or len(std_df) == 0:
        return frame_manifest
    std_path_col = find_col(std_df, ["standardized_path", "relative_path", "path", "frame_path", "filepath", "file_path"])
    if std_path_col is None:
        return frame_manifest
    for key, cands in [
        ("frame_id", ["frame_id","id"]),
        ("timestamp", ["timestamp","time","datetime","valid_time","frame_time"]),
        ("filename", ["filename","file_name","name"]),
    ]:
        if key not in frame_manifest.columns:
            continue
        sk = find_col(std_df, cands)
        if sk is None:
            continue
        temp = std_df[[sk, std_path_col]].copy()
        temp.columns = [key, "standardized_path"]
        temp = temp.drop_duplicates(key)
        merged = frame_manifest.merge(temp, on=key, how="left")
        if merged["standardized_path"].notna().any():
            return merged
    return frame_manifest

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args()
    root = Path(args.repo_root).resolve()
    cfg = load_cfg(root)
    out = ensure_out(root, cfg)

    fsrc = p(root, cfg.get("canonical_frame_index", cfg["raw_frames_csv"]))
    ssrc = p(root, cfg.get("canonical_sample_index", cfg["raw_samples_csv"]))
    fdf0 = pd.read_csv(fsrc)
    sdf0 = pd.read_csv(ssrc)

    fdf = canonicalize(fdf0, FRAME_CANDIDATES)
    sdf = canonicalize(sdf0, SAMPLE_CANDIDATES)

    if "frame_id" not in fdf:
        fdf["frame_id"] = [f"F{i:06d}" for i in range(len(fdf))]
    if "sample_id" not in sdf:
        sdf["sample_id"] = [f"S{i:06d}" for i in range(len(sdf))]

    if "split" in fdf:
        fdf["split"] = fdf["split"].map(normalize_split)
    if "split" in sdf:
        sdf["split"] = sdf["split"].map(normalize_split)

    std_path = p(root, cfg["standardized_frames_csv"])
    std_df = pd.read_csv(std_path) if std_path.exists() else None
    fdf = add_standardized_path(fdf, std_df)

    fdf.to_csv(out/"frame_manifest.csv", index=False)
    sdf.to_csv(out/"sample_manifest.csv", index=False)

    print("frame_manifest columns:", list(fdf.columns))
    print("sample_manifest columns:", list(sdf.columns))
    print(f"Canonical frame manifest: {len(fdf)} rows -> {out/'frame_manifest.csv'}")
    print(f"Canonical sample manifest: {len(sdf)} rows -> {out/'sample_manifest.csv'}")

    if len(fdf) != cfg["expected"]["frame_count"] or len(sdf) != cfg["expected"]["sample_count"]:
        raise SystemExit("Canonical manifest counts do not match expected radar_v1 totals.")

    required_sample_cols = ["sample_id","split","input_paths","target_paths","input_timestamps","target_timestamps"]
    missing = [c for c in required_sample_cols if c not in sdf.columns]
    if missing:
        raise SystemExit(f"Missing canonical sample columns after mapping: {missing}")

    print("Manifest build PASS.")

if __name__ == "__main__":
    main()
