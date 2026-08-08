from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = repo_root() / cfg_path
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"配置文件格式错误：{cfg_path}")
    return cfg


def resolve_device(name: str) -> str:
    import torch

    name = (name or "auto").lower()
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name in {"cuda", "cpu"}:
        if name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("配置要求 CUDA，但当前环境不可用")
        return name
    raise ValueError(f"不支持的 device：{name}")
