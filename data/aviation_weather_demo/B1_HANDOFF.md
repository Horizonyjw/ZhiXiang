# 王梓涵 → 模型侧：B1 数据对接说明

> 交接目标：回答《09-数据与评测对接清单.md》里数据侧的 7 个问题。  
> 本交接包只使用此前上传材料中已明确的信息；材料未明确的地方会标为“待确认”，不会自行补造。

## 1. 样例 / 全量样本在哪？如何读成张量？

原始真实雷达 PNG 放在：

```text
data/radar_raw/
```

运行全流程：

```bash
docker compose exec api python scripts/run_pipeline.py
```

之后会生成全量样本索引：

```text
reports/sample_manifest_dataset_<dataset_id>.csv
```

模型侧读取：

```python
from dataset.dataloader import create_dataloader

loader = create_dataloader(
    "reports/sample_manifest_dataset_1.csv",
    split="train",
    batch_size=4,
)

inputs, targets, metadata = next(iter(loader))

print(inputs.shape)   # [B, 5, 1, 352, 512]
print(targets.shape)  # [B, 3, 1, 352, 512]
print(metadata.keys())
```

如果只想检查接口：

```bash
docker compose exec api python scripts/check_b1_handoff.py \
  reports/sample_manifest_dataset_1.csv --batch-size 4
```

---

## 2. Tin / Tout / C / H / W / 时间间隔

当前接口固定为：

```text
Tin  = 5
Tout = 3
C    = 1
H    = 352
W    = 512
时间间隔 = 6 min
dtype = float32
layout = [B, T, C, H, W]
```

即利用过去 5 帧雷达图预测未来 3 帧。

---

## 3. 数值单位、范围、缺失值、无回波

### 当前张量

```text
表示：归一化灰度图像强度
范围：[0, 1]
物理单位：无
```

当前材料明确说明“不进行颜色到物理 dBZ 的反演”，所以**不能把这里的数值称为 dBZ**。

预处理：

```text
RGBA PNG
→ RGB / 255
→ 灰度
→ alpha 无效区域置 0
→ resize 352×512
→ float32
```

### 时间缺失

当前不插值补雷达帧：

```text
出现时间缺口
→ 切断连续片段
→ 样本不能跨缺口
```

### 无回波

此前上传材料没有明确给出“无回波”的专门编码规则。

因此当前实现：

- 不自行创造特殊 sentinel 值；
- PNG 灰度是多少就按归一化结果读取；
- 只有 alpha 标记为无效的区域明确置 0。

---

## 4. 是否归一化？训练 / 评测尺度

是。

当前 DataLoader 输出统一为：

```text
float32, [0, 1]
```

训练和当前图像空间评测应使用同一尺度。

由于当前没有 PNG → dBZ 的物理反演映射，所以不存在可靠的“反归一化回 dBZ”操作。  
CSI / POD / FAR 的物理阈值如何定义属于评测侧仍需统一的事项，不能在数据侧自行假设。

---

## 5. DataLoader 返回什么？

可以直接：

```python
inputs, targets, metadata = next(iter(loader))
```

其中：

```text
inputs  : [B, 5, 1, 352, 512] float32
targets : [B, 3, 1, 352, 512] float32
metadata:
    sample_id
    split
    start_time
    input_times
    target_times
```

---

## 6. train / val / test 如何划分？是否按天气过程？

当前交接版本的实现约定：

```text
生成全部合法连续 5→3 滑动窗口
→ 按样本开始时间排序
→ 70% train
→ 20% validation
→ 10% test
```

**当前不是按天气过程划分。**

原因不是材料明确要求这样做，而是上传材料并未给出“天气过程”的识别字段或正式划分规则，所以本版本先采用可复现的时间顺序划分。

如果后续团队确定“按天气过程划分”的事件 ID / 起止时间，应统一替换此规则后再进行最终模型比较。

---

## 7. 授权 / 引用 / Git

当前能明确的只有：

- 代码可以做版本管理；
- 原始雷达数据的公开授权与引用要求，上传任务材料要求“核实”，但现有材料没有给出最终授权结论；
- 因此在导师 / 数据提供方确认之前，不把真实雷达 PNG 上传到公开 Git；
- `data/radar_raw/*` 和 `reports/*` 已默认加入 `.gitignore`；
- 数据来源、发布机构、正式引用格式仍需由导师 / 数据提供方确认。

---

# 模型侧最短接入方式

```python
from dataset.dataloader import create_dataloader

train_loader = create_dataloader(
    "reports/sample_manifest_dataset_1.csv",
    split="train",
    batch_size=4,
)

for inputs, targets, metadata in train_loader:
    # inputs:  [B, 5, 1, 352, 512]
    # targets: [B, 3, 1, 352, 512]
    predictions = model(inputs)
    loss = criterion(predictions, targets)
    break
```

更详细的机器可读接口见：

```text
handoff/data_contract.json
```
