import argparse
import json
import time
from pathlib import Path

import psutil
import torch

from dataset.dataloader import create_dataloader

EXPECTED_INPUT = (5, 1, 352, 512)
EXPECTED_TARGET = (3, 1, 352, 512)

def main():
    parser = argparse.ArgumentParser(description="B1 数据接口交接自检")
    parser.add_argument("manifest")
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    contract_path = Path("handoff/data_contract.json")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))

    loader = create_dataloader(
        args.manifest,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=0,
    )

    process = psutil.Process()
    before = process.memory_info().rss / 1024 / 1024
    start = time.perf_counter()
    inputs, targets, metadata = next(iter(loader))
    elapsed = time.perf_counter() - start
    after = process.memory_info().rss / 1024 / 1024

    errors = []

    if tuple(inputs.shape[1:]) != EXPECTED_INPUT:
        errors.append(f"inputs shape 错误: {tuple(inputs.shape)}")
    if tuple(targets.shape[1:]) != EXPECTED_TARGET:
        errors.append(f"targets shape 错误: {tuple(targets.shape)}")
    if inputs.dtype != torch.float32 or targets.dtype != torch.float32:
        errors.append(f"dtype 错误: {inputs.dtype}, {targets.dtype}")
    if float(inputs.min()) < 0 or float(inputs.max()) > 1:
        errors.append("inputs 不在 [0,1]")
    if float(targets.min()) < 0 or float(targets.max()) > 1:
        errors.append("targets 不在 [0,1]")

    print("=== B1 DATA HANDOFF CHECK ===")
    print("manifest:", args.manifest)
    print("split:", args.split)
    print("inputs shape :", tuple(inputs.shape))
    print("targets shape:", tuple(targets.shape))
    print("inputs range :", float(inputs.min()), float(inputs.max()))
    print("targets range:", float(targets.min()), float(targets.max()))
    print("metadata keys:", list(metadata.keys()))
    print("first batch seconds:", round(elapsed, 3))
    print("RSS MB:", round(before, 1), "->", round(after, 1))
    print("contract:", contract_path)

    if errors:
        print("\nFAILED")
        for item in errors:
            print("-", item)
        raise SystemExit(1)

    print("\nPASS: DataLoader 满足当前 B1 接口契约。")

if __name__ == "__main__":
    main()
