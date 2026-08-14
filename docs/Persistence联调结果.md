# Persistence 联调结果

> **状态：radar_v1 handoff 真实前向已跑通（2026-08-14）。**  
> 对齐队友 [wjh](https://github.com/Horizonyjw/ZhiXiang/tree/wjh) U-Net handoff 联调路径。

## 1. 联调前置条件

- [x] 已获得 `radar_v1/04_handoff/handoff_samples.npz`
- [x] 数值范围 `[0,1]` 归一化灰度代理（非 dBZ）
- [x] Tin=5, Tout=3, C=1, H=352, W=512
- [x] Persistence 统一 5D 接口通过随机张量与真实样本检查
- [x] 已保存 `predictions.npz`；本地计算灰度 MAE/MSE（评测 v0.2 口径）
- [ ] CSI / POD / FAR（评测 v0.2 暂不计算）

## 2. 实验信息

| 项目 | 实际值 |
| --- | --- |
| 实验编号 | `20260814-persistence-radar_v1-01` |
| 日期 | 2026-08-14 |
| 数据版本 | `radar_v1` |
| 模型 | Persistence（复制输入最后一帧到 Tout 个未来时刻） |
| 输入 / 预测 | Tin=5 → Tout=3，C=1，H=352，W=512 |
| handoff | `radar_v1/04_handoff/handoff_samples.npz`（N=16） |
| 配置 | `configs/20260814-persistence-radar_v1-01.yaml` |
| 设备 | CUDA（RTX 5060 Laptop） |
| 结果目录 | `results/20260814-persistence-radar_v1-01/` |

## 3. 运行证据

随机张量前向：

```text
input : (2, 5, 1, 352, 512)
output: (2, 3, 1, 352, 512)
forward test OK
```

真实 handoff 前向：

```text
inputs : (16, 5, 1, 352, 512) float32
targets: (16, 3, 1, 352, 512) float32
pred   : (16, 3, 1, 352, 512) float32
persistence handoff forward OK
```

## 4. handoff 灰度误差（评测 v0.2 口径，非 dBZ）

| 范围 | MAE | MSE |
| --- | ---: | ---: |
| overall | 1.446e-04 | 3.388e-05 |
| T+6 min（step 1） | 1.095e-04 | 2.384e-05 |
| T+12 min（step 2） | 1.500e-04 | 3.480e-05 |
| T+18 min（step 3） | 1.745e-04 | 4.300e-05 |

说明：指标在 `[0,1]` 灰度上计算；CSI/POD/FAR 未计算。完整数值见 `results/20260814-persistence-radar_v1-01/metrics.json`。

## 5. 命令

```powershell
# 随机张量
python -m train.test_persistence_forward

# 真实 handoff（推荐）
python -m train.test_persistence_handoff

# 等价
python -m train.run_persistence --config configs/20260814-persistence-radar_v1-01.yaml
```

## 6. 结果目录

```text
results/20260814-persistence-radar_v1-01/
├── config.yaml
├── predictions.npz      # inputs / targets / predictions / metadata_json
├── metrics.json         # MAE / MSE（overall + per lead time）
└── summary.json
```
