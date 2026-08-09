# models｜模型模块

**状态：U-Net / Persistence / FNO（接口测试版）已实现。**

## 公共接口

所有模型必须遵守：

```text
输入：[B, Tin, C, H, W]
输出：[B, Tout, C, H, W]
默认 dtype：float32
```

## 当前实现

| 文件 | 说明 |
| --- | --- |
| `persistence.py` | Persistence：复制输入最后一帧到全部未来时刻 |
| `unet.py` | 经典 2D U-Net + `RadarUNet` 包装器（时间维拼通道） |
| `fno.py` | 轻量 2D FNO + `RadarFNO` 包装器（时间维拼通道） |
| `__init__.py` | 导出 `Persistence`、`RadarUNet`、`RadarFNO` 等 |

### Persistence

- 无参数基线；
- `y[:, t] = x[:, -1]`，对所有 `t < Tout`；
- 用于验证数据—模型—保存链路。

### RadarUNet

- 外部保持 5D 接口；
- 内部将 `[B,Tin,C,H,W]` reshape 为 `[B,Tin*C,H,W]`；
- 输出再还原为 `[B,Tout,C,H,W]`；
- 默认 `depth=4` 时，`H`、`W` 需能被 16 整除。

### RadarFNO

- 外部保持 5D 接口；
- 内部同样时间维拼通道后走频谱卷积；
- 仓库内原无外部 FNO 工程，本实现用于 Week1.1 随机张量前向；
- 正式训练前修改项见 `docs/FNO接口适配说明.md`。

### 参数占位

`Tin/Tout/H/W` 等以配置文件为准，当前为开发占位，待组内确认后替换。

## 最小测试

```powershell
# 在仓库根目录
python -m train.test_persistence_forward
python -m train.test_fno_forward
python -m train.test_unet_forward
```
