# eval_v1.1 输入输出位置

## 第一层：雷达预测评测

评测程序：`evaluate/metrics_v1.1.py`

### 输入

| 输入文件 | 位置 |
| --- | --- |
| 模型预测 | `results/<experiment_id>/predictions.npz` |
| 原始雷达图片 | 模型侧电脑中的 `气象大模型数据（中南）-20260224.zip` |
| 冻结样本清单 | `radar_v1/07_pilot_contract/sample_manifest.csv` |
| 评测配置 | `configs/evaluation_v1.1.yaml` |

模型侧只需在 `metrics_v1.1.py` 开头填写：

```python
RADAR_DATA_ZIP_PATH = r"本机实际数据路径"
```

### 输出

输出目录：`results/<experiment_id>/evaluation_v1.1/`

| 输出文件 | 内容 |
| --- | --- |
| `metrics.json` | MAE、MSE、SSIM、CSI、POD、FAR |
| `evaluation_manifest.json` | 本次评测使用的数据、代码、配置及哈希 |

## 第二层：自主研究评测

评测程序：`evaluate/auto_research_metrics_v1.1.py`

| 类型 | 文件位置 |
| --- | --- |
| 输入 | `results/auto_research/<batch_id>/experiment_registry.jsonl` |
| 输出 | `results/auto_research/<batch_id>/auto_research_metrics_v1.1.json` |

第二层输出：VCR、ACR、Error Recovery、Gain/GPU-hour、Gain/100K Tokens、Validation-Test Gap。
