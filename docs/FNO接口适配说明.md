# FNO 接口适配说明（Week1.1）

> **日期**：2026-08-09 初稿 / **2026-08-14 真实数据训练更新**  
> **范围**：接口对齐 + handoff 真实样本过拟合；全量 train 集正式训练仍属下一阶段。  
> **数据**：统一使用仓库内 `radar_v1`（handoff / 索引）；原始 PNG 由数据侧另行共享，不写入仓库路径配置。

## 1. 现有代码排查结论

| 目标 | 结果 |
| --- | --- |
| PDE 神经算子独立工程 | **未找到** |
| 现成 FNO 训练仓库入口 | **未找到** |
| autoresearch 代码 | **未找到** |
| 队友 U-Net 模块 | 已存在：`models/unet.py` |

因此新增项目内自包含轻量实现：

- 入口类：`models.fno.RadarFNO`
- 底层：`FNO2d` / `SpectralConv2d`
- 随机前向：`python -m train.test_fno_forward`
- 真实前向：`python -m train.test_fno_handoff`
- 真实过拟合：`python -m train.overfit_fno_handoff`
- 配置：`configs/fno_baseline.yaml`、`configs/20260814-fno-radar_v1-overfit-01.yaml`
- 联调数据：`radar_v1/04_handoff/handoff_samples.npz`

## 2. 输入输出维度（已对齐 radar_v1）

```text
输入：[B, 5, 1, 352, 512]
输出：[B, 3, 1, 352, 512]
dtype：float32
数值：[0,1] 归一化灰度代理（非 dBZ）
```

`RadarFNO` 内部：

```text
[B, Tin, C, H, W]
  -> reshape [B, Tin*C, H, W]
  -> FNO2d
  -> reshape [B, Tout, C, H, W]
```

与 Persistence / U-Net 公共格式一致。

## 3. 前向测试记录（2026-08-14 实测）

### 3.1 随机张量

```powershell
python -m train.test_fno_forward
```

```text
device: cuda
params: 2102723
input : (2, 5, 1, 352, 512)
output: (2, 3, 1, 352, 512)
forward test OK
```

### 3.2 真实 handoff 未训练前向（接口验证）

```powershell
python -m train.test_fno_handoff
```

```text
inputs : (16, 5, 1, 352, 512) float32
pred   : (2, 3, 1, 352, 512) float32
fno handoff forward OK
```

### 3.3 真实 handoff 小样本过拟合（已训练）

数据来源：`radar_v1/04_handoff/handoff_samples.npz`（N=16，标准化真实样本）。

```powershell
python -m train.overfit_fno_handoff
```

| 项 | 结果（2026-08-14） |
| --- | --- |
| 训练损失 | epoch1 MSE≈6.79e-03 → epoch40 MSE≈2.89e-05（下降） |
| overall MAE / MSE | ≈3.11e-04 / 2.87e-05 |
| 产物 | `results/20260814-fno-radar_v1-overfit-01/` |

> 这是 handoff **过拟合**，用于证明可在真实灰度数据上训练；**不是**全量 train/val/test 正式对比。

结构默认：`fno_width=32, modes1=16, modes2=16, n_layers=4`。  
字段 `fno_width` 是隐藏通道数，勿与空间 `width=512` 混淆。

## 4. 接口适配情况

| 项目 | 状态 |
| --- | --- |
| 统一 5D 接口包装 | 已完成（`RadarFNO`） |
| 与 Persistence / U-Net 同配置风格 | 已完成 |
| 随机张量前向 | 已完成 |
| 真实样本前向 | **已完成**（handoff） |
| handoff 过拟合训练 | **已完成**（`overfit_fno_handoff`） |
| 全量 DataLoader 训练 | 未做（原始 PNG 路径已接入，待下一阶段） |
| 与评测脚本联通 | 已导出训练后 `predictions.npz` + `metrics.json` |

## 5. 第二阶段 FNO 待修改事项

1. **数据接入**：用全量 DataLoader（train/val/test）替换 handoff 小样本。  
2. **modes / 显存**：按 352×512 与 batch 再调 `modes1/modes2`、`fno_width`。  
3. **训练脚手架**：补充过拟合与正式训练入口，复用 `export_predictions`。  
4. **超参锁定**：与 Persistence / U-Net 统一对比协议后再定 lr、epochs 等。  
5. **权重保存**：确认 `cfloat` 参数跨设备加载与混合精度策略。  
6. **外部官方 FNO**：若后续接入，保持 5D 公共接口不变。  
7. **autoresearch**：外部自动实验代码到位后再接 `results/<experiment_id>/`。

## 6. 协作说明

- 联调与训练入口统一使用仓库内 `radar_v1`（handoff / 索引 / DataLoader 代码）。  
- 原始 PNG 不提交 Git；全量训练时由成员自行配置本机数据根目录（勿提交含本机盘符的配置）。  
- 评测 v0.2：灰度 MAE/MSE；CSI/POD/FAR 暂不计算。  
- Persistence 真实结果见 `docs/Persistence联调结果.md`。  
- FNO 过拟合结果：`results/20260814-fno-radar_v1-overfit-01/`。
