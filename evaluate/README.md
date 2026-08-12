# evaluate｜统一评测

**状态：已接入 zjr 分支评测实现（第一阶段指标）。**

## 文件

| 文件 | 说明 |
| --- | --- |
| `metrics.py` | MAE / MSE / CSI / POD / FAR（来源：`origin/zjr`） |
| `run_from_npz.py` | 读取 `predictions.npz` → 写 `metrics.json/.csv`，可选出图 |
| `__init__.py` | 导出 `evaluate`、`save_metrics` |

## 约定

- 输入预测/真值形状：`[B, Tout, C, H, W]`
- 第一阶段回波阈值：**20.0 dBZ**（见 `docs/评测指标与阈值说明v0.1.md`）
- 模型侧导出字段见 `train/export_predictions.py`：`inputs` / `targets` / `predictions`

> 若数据尚未转到 dBZ，分类指标（CSI/POD/FAR）需等单位确认后再解读。

## 命令

```powershell
cd E:\ZhiXiang\code
.\.venv\Scripts\python.exe -m evaluate.run_from_npz `
  --npz results/<experiment_id>/predictions.npz `
  --output-dir results/<experiment_id> `
  --threshold 20.0 `
  --plot
```
