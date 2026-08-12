# train｜训练与最小测试

**状态：已提供 U-Net 前向测试与人工数据过拟合脚手架；真实数据训练入口待 DataLoader 到位后接入。**

## 脚本

| 脚本 | 用途 |
| --- | --- |
| `test_unet_forward.py` | 随机张量前向（占位配置） |
| `test_handoff_forward.py` | radar_v1 handoff 真实前向 |
| `overfit_unet_toy.py` | 人工数据过拟合脚手架 |
| `overfit_unet_handoff.py` | radar_v1 handoff 小样本过拟合 |
| `export_predictions.py` | 导出 `predictions.npz` 供评测读取（对接 `evaluate.run_from_npz`） |
| `config_utils.py` | 配置读取与 device 解析 |

## 命令

在仓库根目录执行：

```powershell
# 随机张量前向
.\.venv\Scripts\python.exe -m train.test_unet_forward

# 人工数据过拟合（结果写入 results/<experiment_id>/）
.\.venv\Scripts\python.exe -m train.overfit_unet_toy
```

默认配置：`configs/unet_baseline.yaml`。

## 注意

- 人工数据结果**不能**作为正式模型对比结论；
- 真实样本前向 / 正式训练需等待数据接口确认；
- 大权重与预测文件默认不提交 Git。
