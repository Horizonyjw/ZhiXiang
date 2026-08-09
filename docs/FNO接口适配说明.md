# FNO 接口适配说明（Week1.1）

> **日期**：2026-08-09  
> **范围**：本阶段不做正式训练，只完成入口定位、维度检查、随机张量前向与适配记录。

## 1. 现有代码排查结论

在当前仓库（及本地上级共享目录检索）中：

| 目标 | 结果 |
| --- | --- |
| PDE 神经算子独立工程 | **未找到** |
| 现成 FNO 训练仓库入口 | **未找到** |
| autoresearch 代码 | **未找到** |
| 队友 U-Net 模块 | 已存在：`models/unet.py` |

因此本阶段在仓库内新增自包含轻量实现：

- 入口类：`models.fno.RadarFNO`
- 底层：`FNO2d` / `SpectralConv2d`
- 前向测试：`python -m train.test_fno_forward`
- 配置：`configs/fno_baseline.yaml`

## 2. 输入输出维度

统一公共接口：

```text
输入：[B, Tin, C, H, W]
输出：[B, Tout, C, H, W]
dtype：float32
```

`RadarFNO` 内部转换：

```text
[B, Tin, C, H, W]
  -> reshape [B, Tin*C, H, W]
  -> FNO2d
  -> reshape [B, Tout, C, H, W]
```

与 `RadarUNet` 的时间维拼通道策略一致，DataLoader / 评测侧无需为 FNO 单独改公共格式。

## 3. 随机张量前向测试

命令：

```powershell
python -m train.test_fno_forward
```

检查项：

- 输出形状等于 `(B, Tout, C, H, W)`
- 无 NaN / Inf
- CUDA 可用时在 GPU 上完成一次前向

占位默认：`Tin=Tout=5, C=1, H=W=128, fno_width=32, modes=12, n_layers=4`（待组内确认后替换）。  
配置字段 `fno_width` 表示隐藏通道数，勿与空间尺寸字段 `width` 混淆。

## 4. 接口适配情况

| 项目 | 状态 |
| --- | --- |
| 统一 5D 接口包装 | 已完成（`RadarFNO`） |
| 与 Persistence / U-Net 同配置字段风格 | 已完成 |
| 随机张量前向脚本 | 已完成 |
| 真实样本前向 | **未完成**（等待 DataLoader / 真实样例） |
| 训练入口 / 损失 / 优化器 | **未做**（本阶段不做正式训练） |
| 与评测脚本联通 | **未做** |

## 5. 第二阶段 FNO 待修改事项

1. **数据接入**：用真实 DataLoader 替换随机张量；确认 Tin/Tout、归一化范围、缺失值掩码是否进入损失。  
2. **空间尺寸与 modes**：按真实 `H,W` 重设 `modes1/modes2`（通常不超过 `H/2`、`W/2+1`）；检查奇数尺寸与 padding。  
3. **训练脚手架**：补充 `train/` 下 FNO 小样本过拟合与正式训练入口，复用 `export_predictions`。  
4. **超参锁定**：`fno_width / n_layers / lr / batch_size` 需与显存和基线对比协议一起确认，不得沿用占位值当正式结论。  
5. **权重与复数值**：确认跨设备保存/加载（`cfloat` 参数）及混合精度策略。  
6. **若后续提供外部官方 FNO 仓库**：评估是替换 backbone 还是保留当前包装器；替换时必须保持 5D 公共接口不变。  
7. **autoresearch**：外部自动实验代码到位后，再把配置读写、指标回传接到统一 `results/<experiment_id>/` 规范。

## 6. 阻塞与协作

- 真实雷达连续样例、单位与缺失值定义仍见 `docs/当前问题清单.md`（W1-01～W1-04）。  
- 在数据未确认前，FNO 只报告接口级前向结果，不报告任何气象技巧评分。
