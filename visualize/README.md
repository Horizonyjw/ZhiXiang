# visualize｜结果可视化

**状态：已接入 zjr 分支结果图模板。**

## 文件

| 文件 | 说明 |
| --- | --- |
| `plot_results.py` | 序列对比图 + 逐时效指标图（来源：`origin/zjr`） |

依赖：`numpy`、`Pillow`（当前环境随 torchvision 已具备）。

通常由评测入口调用：

```powershell
.\.venv\Scripts\python.exe -m evaluate.run_from_npz --npz ... --plot
```

输出默认写入 `results/<experiment_id>/figures/`。
