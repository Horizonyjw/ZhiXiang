# models｜模型模块

**状态：U-Net 基线骨架已实现；Persistence / FNO 等尚未实现。**

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
| `unet.py` | 经典 2D U-Net + `RadarUNet` 包装器（时间维拼通道） |
| `__init__.py` | 导出 `RadarUNet`、`ClassicUNet2D` |

### RadarUNet

- 外部保持 5D 接口；
- 内部将 `[B,Tin,C,H,W]` reshape 为 `[B,Tin*C,H,W]`；
- 输出再还原为 `[B,Tout,C,H,W]`；
- 回归线性头，无 softmax/sigmoid；
- 结构为项目内自包含实现，参考经典 U-Net，**非**直接依赖第三方 U-Net 库。

### 空间尺寸

默认 `depth=4` 时，`H`、`W` 需能被 16 整除（或由数据侧先 pad）。

### 参数占位

`Tin/Tout/H/W` 等以 `configs/unet_baseline.yaml` 为准，当前为开发占位，待组内确认后替换。

## 最小测试

```powershell
cd E:\ZhiXiang\code
.\.venv\Scripts\python.exe -m train.test_unet_forward
```
