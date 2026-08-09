import argparse
import time
import psutil
from dataset.dataloader import create_dataloader

def main():
    parser = argparse.ArgumentParser(description="检查 DataLoader 输出、速度和内存")
    parser.add_argument("manifest")
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    loader = create_dataloader(
        args.manifest,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    process = psutil.Process()
    before_mb = process.memory_info().rss / 1024 / 1024
    start = time.perf_counter()
    history, future, metadata = next(iter(loader))
    elapsed = time.perf_counter() - start
    after_mb = process.memory_info().rss / 1024 / 1024

    print("history shape:", tuple(history.shape))
    print("future shape :", tuple(future.shape))
    print("batch size   :", history.shape[0])
    print("elapsed sec  :", round(elapsed, 3))
    print("RSS before MB:", round(before_mb, 1))
    print("RSS after MB :", round(after_mb, 1))
    print("first sample :", metadata["sample_id"][0])

if __name__ == "__main__":
    main()
