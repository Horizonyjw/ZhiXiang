# 航空气象 B1 演示交付版

这个版本用于**原始雷达 PNG 尚未进入当前工程时的接口演示和团队联调**。

## 你可以直接交付的内容

```text
DEMO_HANDOFF.md                 演示数据说明
B1_HANDOFF.md                   B1 数据侧逐条对接说明
handoff/data_contract.json      接口契约
demo/demo_20_samples.npz        20 组演示接口样本
demo/demo_manifest.csv          20 组时间信息
scripts/demo_check.py           一键接口检查
dataset/                        正式 DataLoader
app/                            正式真实数据处理后端
```

## 演示数据接口

```text
inputs : [20, 5, 1, 352, 512] float32
targets: [20, 3, 1, 352, 512] float32
dt     : 6 min
```

20 组满足当前阶段“10～20 组联调样本”的最低数量要求。

## 一键检查

Docker Desktop 启动后：

```bash
docker compose up --build
```

另开终端：

```bash
docker compose exec api python scripts/demo_check.py
```

正常输出：

```text
sample count : 20
inputs shape : (20, 5, 1, 352, 512)
targets shape: (20, 3, 1, 352, 512)
dtype        : float32
PASS
```

## 模型侧直接读取

```python
import json
import numpy as np

data = np.load("demo/demo_20_samples.npz", allow_pickle=False)

inputs = data["inputs"]
targets = data["targets"]
metadata = json.loads(str(data["metadata_json"]))
```

## 边界说明

当前工程中没有收到原始 7396 张雷达 PNG。

因此：

- 20 组数据用于接口/页面/代码演示；
- 数组值没有实测雷达物理含义；
- 不用于训练结论；
- 不用于 CSI / POD / FAR 等正式评测；
- 不用于论文或项目报告中的实验结果。

正式雷达数据到达后：

```text
data/radar_raw/
→ scripts/run_pipeline.py
→ reports/sample_manifest_dataset_*.csv
→ dataset/dataloader.py
```

正式 DataLoader 仍保持：

```text
inputs  [B, 5, 1, 352, 512]
targets [B, 3, 1, 352, 512]
```

因此模型侧无需更换输入输出接口。
