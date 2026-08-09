import argparse
import shutil
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="导出模型侧 B1 交接目录")
    parser.add_argument("--manifest", required=True, help="reports/sample_manifest_dataset_*.csv")
    parser.add_argument("--output", default="handoff_output")
    args = parser.parse_args()

    manifest = Path(args.manifest)
    if not manifest.exists():
        raise FileNotFoundError(f"manifest 不存在: {manifest}")

    out = Path(args.output)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    shutil.copy2("B1_HANDOFF.md", out / "B1_HANDOFF.md")
    shutil.copy2("handoff/data_contract.json", out / "data_contract.json")
    shutil.copy2("handoff/model_side_example.py", out / "model_side_example.py")
    shutil.copy2(manifest, out / "sample_manifest.csv")

    print("B1 交接目录已生成:", out.resolve())
    print("注意：没有复制真实雷达 PNG。模型侧需要共享/挂载同一数据目录，或由数据提供方单独传输真实数据。")

if __name__ == "__main__":
    main()
