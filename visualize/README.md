# visualize｜结果可视化

**状态：已接入评测 v0.2 结果图模板（需 `experiment_id`）。**

## 文件

| 文件 | 说明 |
| --- | --- |
| `plot_results.py` | 序列对比图 + 逐时效 MAE/MSE 图（来源：`origin/zjr` v0.2） |

依赖：`numpy`、`Pillow`（当前环境随 torchvision 已具备）。

`make_result_figures` 要求实验编号符合 `YYYYMMDD-模型-数据版本-序号`，例如 `20260812-unet-radar_v1-01`。未计算的 CSI/POD/FAR 不画空白面板。默认不覆盖已有图片。

通常由评测入口调用：

```powershell
.\.venv\Scripts\python.exe -m evaluate.run_from_npz --npz ... --plot
```

输出默认写入 `results/<experiment_id>/figures/{id}_sequence-comparison.png` 与 `{id}_metric-changes.png`。
