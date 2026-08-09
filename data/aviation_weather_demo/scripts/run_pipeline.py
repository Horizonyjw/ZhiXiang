import argparse
from app.database import Base, SessionLocal, engine
from app.radar import build_samples, scan_folder
from app.reporting import export_all_reports

def main():
    parser = argparse.ArgumentParser(description="真实雷达数据：扫描 -> 全量样本 -> 报告")
    parser.add_argument("--data-dir", default="data/radar_raw")
    parser.add_argument("--name", default="系统历史雷达拼图_全量")
    args = parser.parse_args()

    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        dataset, scan_summary = scan_folder(db, args.data_dir, args.name)
        split_counts = build_samples(db, dataset.id)
        report_paths = export_all_reports(db, dataset.id)

        print("=== 扫描完成 ===")
        for k, v in scan_summary.items():
            print(f"{k}: {v}")

        print("\n=== 全量 5->3 样本 ===")
        print(f"dataset_id: {dataset.id}")
        print(f"sample_count: {sum(split_counts.values())}")
        print(f"train: {split_counts['train']}")
        print(f"validation: {split_counts['validation']}")
        print(f"test: {split_counts['test']}")

        print("\n=== 输出文件 ===")
        for k, v in report_paths.items():
            print(f"{k}: {v}")

if __name__ == "__main__":
    main()
