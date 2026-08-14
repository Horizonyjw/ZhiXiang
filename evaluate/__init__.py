"""评测模块：官方入口为 v0.2（灰度 MAE/MSE；CSI/POD/FAR 暂不计算）。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_METRICS_PATH = Path(__file__).resolve().parent / "metrics_v0.2.py"
_spec = importlib.util.spec_from_file_location("evaluate_metrics_v0_2", _METRICS_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"无法加载评测模块：{_METRICS_PATH}")
_metrics_v02 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_metrics_v02)

evaluate = _metrics_v02.evaluate
save_metrics = _metrics_v02.save_metrics
evaluate_file = getattr(_metrics_v02, "evaluate_file", None)

__all__ = ["evaluate", "save_metrics", "evaluate_file"]
