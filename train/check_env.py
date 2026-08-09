"""
统一运行环境最小检查。

用法（仓库根目录）:
  python -m train.check_env
  .\\.venv\\Scripts\\python.exe -m train.check_env
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    print("=== ZhiXiang 环境检查 ===")
    print("python:", sys.version.replace("\n", " "))
    print("platform:", platform.platform())
    print("cwd:", Path.cwd())
    print("repo:", ROOT)

    errors: list[str] = []

    try:
        import numpy as np

        print("numpy:", np.__version__)
    except Exception as e:  # noqa: BLE001
        errors.append(f"numpy 导入失败: {e}")

    try:
        import yaml

        print("PyYAML:", getattr(yaml, "__version__", "unknown"))
    except Exception as e:  # noqa: BLE001
        errors.append(f"PyYAML 导入失败: {e}")

    try:
        import torch

        print("torch:", torch.__version__)
        print("torch.cuda:", torch.version.cuda)
        print("cuda_available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("gpu_count:", torch.cuda.device_count())
            print("gpu_name:", torch.cuda.get_device_name(0))
            props = torch.cuda.get_device_properties(0)
            total_gb = props.total_memory / (1024**3)
            print(f"gpu_memory_gb: {total_gb:.2f}")
            x = torch.randn(4, 4, device="cuda", dtype=torch.float32)
            y = x @ x.T
            assert torch.isfinite(y).all()
            print("cuda_matmul: OK")
        else:
            x = torch.randn(4, 4, dtype=torch.float32)
            y = x @ x.T
            assert torch.isfinite(y).all()
            print("cpu_matmul: OK (CUDA 不可用)")
    except Exception as e:  # noqa: BLE001
        errors.append(f"PyTorch 检查失败: {e}")

    print("---")
    if errors:
        for err in errors:
            print("ERROR:", err)
        print("环境检查失败")
        return 1

    print("环境检查通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
