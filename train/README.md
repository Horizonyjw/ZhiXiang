# train｜训练与最小测试

**状态：已提供环境检查、Persistence / FNO / U-Net 前向测试；真实数据训练入口待 DataLoader 到位后接入。**

## 脚本

| 脚本 | 用途 |
| --- | --- |
| `check_env.py` | Python / PyTorch / CUDA / 张量运算最小检查 |
| `test_persistence_forward.py` | Persistence 随机张量前向 |
| `run_persistence.py` | Persistence 推理并保存 `predictions.npz` |
| `test_fno_forward.py` | FNO 随机张量前向 |
| `test_unet_forward.py` | U-Net 随机张量前向，检查输出维度 |
| `overfit_unet_toy.py` | 固定人工样本过拟合脚手架 |
| `export_predictions.py` | 导出 `predictions.npz` 供评测读取 |
| `config_utils.py` | 配置读取与 device 解析 |

## 命令

在仓库根目录执行：

```powershell
python -m train.check_env
python -m train.test_persistence_forward
python -m train.run_persistence
python -m train.test_fno_forward
python -m train.test_unet_forward

# 真实样本到位后（npz 含 inputs/targets）
python -m train.run_persistence --npz path/to/real_batch.npz

# 人工数据过拟合（结果写入 results/<experiment_id>/）
python -m train.overfit_unet_toy
```

默认配置：

- Persistence：`configs/persistence_baseline.yaml`
- FNO：`configs/fno_baseline.yaml`
- U-Net：`configs/unet_baseline.yaml`

## 注意

- 人工 / 随机张量结果**不能**作为正式模型对比结论；
- 真实样本前向 / 正式训练需等待数据接口确认；
- 大权重与预测文件默认不提交 Git。
