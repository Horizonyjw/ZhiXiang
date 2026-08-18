# v2 修复说明

如果旧版 `01_preflight.py` 输出类似：

```text
frame_rows = 4964
sample_rows = 4884
split_counts = {"train": 4884}
```

这通常说明旧脚本错误地把：

```text
02_standardized/standardized_frames.csv
02_standardized/samples_standardized.csv
```

当成了**整个 radar_v1 的 canonical index**。

但 `radar_v1` 的完整数据规模和 split 应以：

```text
03_index/frames.csv
03_index/samples.csv
```

为准。

v2 已修复：

1. `01_preflight.py`
   - 总帧数、总样本数、split 数量改为从 `03_index` 检查；
   - `02_standardized` 只作为“标准化覆盖情况”报告，不再要求它必须有 7396/7302 行。

2. `02_build_manifests.py`
   - `frame_manifest.csv` / `sample_manifest.csv` 以 `03_index` 为主索引；
   - 尝试把标准化 CSV 中的路径信息合并进去。

3. `05_train_intensity_statistics.py`
   - 默认直接扫描 `02_standardized/frames/train/`；
   - 因此不会因为 `standardized_frames.csv` 行数较少而漏算或误用 val/test。

## 你现在怎么做

用 v2 覆盖旧的工具目录，或者直接把 v2 解压到仓库根目录，然后：

```bash
python radar_v1_pilot_certification_tools_v2/scripts/01_preflight.py --repo-root .
```

先看 canonical 结果是否为：

```text
7396 frames
7302 samples
train/val/test = 5097/1472/733
```

如果是，再继续：

```bash
python radar_v1_pilot_certification_tools_v2/scripts/02_build_manifests.py --repo-root .
python radar_v1_pilot_certification_tools_v2/scripts/03_audit_samples_and_splits.py --repo-root .
```

如果 `03_audit...` 因列名无法识别失败，请把命令行输出和
`03_index/frames.csv`、`03_index/samples.csv` 的表头（第一行）发回来即可，不需要发完整数据。
