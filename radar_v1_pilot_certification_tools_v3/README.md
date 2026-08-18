# radar_v1 Pilot 冻结与认证工具

这套工具**不重新清洗 radar_v1，不改变 Tin/Tout、分辨率、归一化或 split**。
它的目标是在现有 `radar_v1` 基础上补齐：

1. v1 完整性预检；
2. frame/sample manifest；
3. split / 时间 / 跨断点 / 跨 split 审计；
4. 归一化说明；
5. 仅 train 的强度统计；
6. 文件 SHA-256 清单；
7. pilot data contract；
8. 最终 `certified_for_pilot` 验收；
9. handoff 样本接口检查；
10. `predictions.npz` 接口检查。

## 推荐放置位置

把整个目录复制到 ZhiXiang 仓库根目录，例如：

```text
ZhiXiang/
├── radar_v1/
├── models/
├── evaluate/
├── ...
└── radar_v1_pilot_certification_tools/
```

## 环境

```bash
python -m pip install -r radar_v1_pilot_certification_tools/requirements.txt
```

## 最短执行流程

先根据实际仓库检查并修改：

```text
radar_v1_pilot_certification_tools/pilot_config.json
```

然后从仓库根目录执行：

```bash
python radar_v1_pilot_certification_tools/scripts/01_preflight.py --repo-root .
python radar_v1_pilot_certification_tools/scripts/02_build_manifests.py --repo-root .
python radar_v1_pilot_certification_tools/scripts/03_audit_samples_and_splits.py --repo-root .
python radar_v1_pilot_certification_tools/scripts/04_build_normalization.py --repo-root .
python radar_v1_pilot_certification_tools/scripts/05_train_intensity_statistics.py --repo-root .
python radar_v1_pilot_certification_tools/scripts/06_build_data_manifest.py --repo-root .
python radar_v1_pilot_certification_tools/scripts/07_build_contract.py --repo-root .
python radar_v1_pilot_certification_tools/scripts/08_certify.py --repo-root .
python radar_v1_pilot_certification_tools/scripts/09_validate_handoff.py --repo-root .
```

模型侧生成 `predictions.npz` 后：

```bash
python radar_v1_pilot_certification_tools/scripts/10_validate_predictions.py \
  --predictions path/to/predictions.npz
```

也可以一次性执行数据侧 01–08：

```bash
python radar_v1_pilot_certification_tools/scripts/run_data_certification.py --repo-root .
```

## 输出目录

默认输出：

```text
radar_v1/07_pilot_contract/
├── frame_manifest.csv
├── sample_manifest.csv
├── split_audit.json
├── normalization.json
├── train_intensity_statistics.json
├── data_manifest.json
├── radar_v1_pilot_data_contract.json
└── certification_report.json
```

## 重要原则

- 不覆盖 `radar_v1` 已有索引和标准化数据；
- 不重新划分 train/val/test；
- 统计阈值只从 train 获得；
- validation 用于搜索和筛选；
- test 从 pilot 启动后冻结，不参与 AutoResearch search；
- 当前数值只称为 `normalized_grayscale_proxy_NOT_dBZ`；
- 如果改变数据规则，应建立 `radar_v2`，不要覆盖 `radar_v1`。
