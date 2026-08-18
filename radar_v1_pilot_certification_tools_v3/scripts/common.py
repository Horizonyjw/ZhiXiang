from __future__ import annotations
from pathlib import Path
import json, ast, hashlib
import pandas as pd
import numpy as np

def load_cfg(repo_root: Path):
    cfg_path = Path(__file__).resolve().parents[1] / "pilot_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["_repo_root"] = str(repo_root.resolve())
    return cfg

def p(repo_root: Path, rel: str) -> Path:
    return (repo_root / rel).resolve()

def ensure_out(repo_root: Path, cfg: dict) -> Path:
    out = p(repo_root, cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    return out

def read_csv_if_exists(path: Path):
    if not path.exists():
        return None
    return pd.read_csv(path)

def parse_list_cell(v):
    if isinstance(v, (list, tuple)):
        return list(v)
    if pd.isna(v):
        return []
    s = str(v).strip()
    if not s:
        return []
    for fn in (json.loads, ast.literal_eval):
        try:
            x = fn(s)
            if isinstance(x, (list, tuple)):
                return list(x)
        except Exception:
            pass
    # Fallback separators
    for sep in ("|", ";", ","):
        if sep in s:
            return [x.strip() for x in s.split(sep) if x.strip()]
    return [s]

def find_col(df, candidates, required=False):
    lower = {str(c).lower(): c for c in df.columns}
    for x in candidates:
        if x.lower() in lower:
            return lower[x.lower()]
    for c in df.columns:
        lc = str(c).lower()
        for x in candidates:
            if x.lower() in lc:
                return c
    if required:
        raise KeyError(f"Cannot find any of columns {candidates}. Available={list(df.columns)}")
    return None

def normalize_split(v):
    s = str(v).strip().lower()
    if s in ("validation", "valid", "dev"):
        return "val"
    return s

def sha256_file(path: Path, chunk=1024*1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def json_dump(path: Path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

def to_dt_series(s):
    return pd.to_datetime(s, errors="coerce", utc=True)
