from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

MODEL_HEIGHT = 352
MODEL_WIDTH = 512

class RadarSequenceDataset(Dataset):
    def __init__(self, manifest_path: str | Path, split: str | None = None):
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"manifest 不存在：{self.manifest_path}")
        self.project_root = Path.cwd().resolve()
        rows = []
        with self.manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if split is None or row["split"] == split:
                    rows.append(row)
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def _load_frame(self, relative_path: str) -> torch.Tensor:
        path = (self.project_root / relative_path).resolve()
        with Image.open(path) as image:
            rgba = image.convert("RGBA")
            rgba = rgba.resize((MODEL_WIDTH, MODEL_HEIGHT), Image.Resampling.BILINEAR)
            arr = np.asarray(rgba, dtype=np.float32)

        rgb = arr[..., :3] / 255.0
        alpha = arr[..., 3] / 255.0
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        gray = gray * (alpha > 0.0)
        return torch.from_numpy(gray.astype(np.float32)).unsqueeze(0)

    def __getitem__(self, index: int):
        row = self.rows[index]
        input_paths = json.loads(row["input_paths_json"])
        target_paths = json.loads(row["target_paths_json"])
        history = torch.stack([self._load_frame(p) for p in input_paths], dim=0)
        future = torch.stack([self._load_frame(p) for p in target_paths], dim=0)
        metadata = {
            "sample_id": int(row["sample_id"]),
            "split": row["split"],
            "start_time": row["start_time"],
            "input_times": json.loads(row["input_times_json"]),
            "target_times": json.loads(row["target_times_json"]),
        }
        return history, future, metadata
