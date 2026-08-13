# evaluate｜统一评测

**状态：已接入评测 v0.2（灰度 MAE/MSE；CSI/POD/FAR 暂不计算）。**

## 文件

| 文件 | 说明 |
| --- | --- |
| `metrics_v0.2.py` | **官方实现**（文件名含点，由 `__init__.py` 加载）：`[0,1]` 灰度 MAE/MSE；CSI/POD/FAR 固定为 `None` |
| `metrics.py` | v0.1 历史实现（20 dBZ 阈值），已弃用 |
| `run_from_npz.py` | 读取 `predictions.npz` → 写 `metrics.json/.csv`，可选出图 |
| `__init__.py` | 导出 v0.2 的 `evaluate`、`save_metrics` |

口径说明见 `docs/评测指标与阈值说明v0.2.md`。

## 约定

- 输入预测/真值形状：`[B, Tout, C, H, W]`
- 有限值必须在 `[0,1]`（归一化灰度，**不是 dBZ**）
- CSI / POD / FAR 当前版本不计算，字段保留为 `null`，不能按 0 解读
- 模型线性头若略超出 `[0,1]`，`run_from_npz` 默认 clip 后再评测
- 模型侧导出字段见 `train/export_predictions.py`：`inputs` / `targets` / `predictions`

## 命令

```powershell
cd E:\ZhiXiang\code
.\.venv\Scripts\python.exe -m evaluate.run_from_npz `
  --npz results/<experiment_id>/predictions.npz `
  --output-dir results/<experiment_id> `
  --plot
```

结果图默认不覆盖已有文件；重跑时加 `--overwrite`。
