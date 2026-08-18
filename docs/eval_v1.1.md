# eval_v1.1：模型—评测器双层接口规范

> 状态：Pilot 冻结草案。本文只规定模型、第一层评测器与第二层自主研究评测器之间的接口

## 1. 总览

```text
模型训练脚本：只输出预测文件。
第一层评测器：按 sample_id 从冻结数据读取真实答案，计算模型成绩。
实验调度器：记录实验结果、资源消耗和失败信息。
第二层评测器：汇总多次实验记录，计算 Auto Research 的工作成绩。
```

- 第一层评“这次雷达预测准不准”。
- 第二层评“自动研究系统是否有效、省资源、少人工”。

## 2. 模型提交与评测时机

### 2.1 两种正式评测

```text
日常模型迭代：train 训练 → 模型提交完整 val 的预测 → 评测器评 val → Agent 决定下一次改动。
最终独立认证：方案冻结、不再根据结果改模型 → 模型提交完整 test 的预测一次 → 评测器评 test。
```

模型搜索阶段只能指定 `split=val`；方案冻结后才允许指定 `split=test`。

### 2.2 `predictions.npz`

```text
pred:        float32 [N, 3, 1, 352, 512]
sample_id:   string [N]
horizon_min: int [3]，固定为 [6, 12, 18]
```

真实 target 不能由模型组提交。评测器按唯一 `sample_id` 从外部冻结真值读取；ID 缺失、重复或预测不完整均为 `invalid`。

## 3. 第一层：雷达预测评测

### 3.1 评测范围

```text
mask_definition = all_pixels_valid_no_mask
masked MAE      = not_computed
masked SSIM     = not_computed
```

所有像素参与 MAE、MSE、SSIM，不设置 mask。

原因：alpha 非零范围会随每张图片变化，表示当前图片的显示内容，不能证明某个地图位置是永久无效区域。若用 `alpha>0` 排除评分像素，可能把真实无回波区域误当成“不需要评分”，因此当前按整图评分。

只要预测或真值含 NaN/Inf，整次评测返回 `evaluation_status=invalid`。

### 3.2 预测时效与指标

```text
T+6  = 最后输入图之后 6 分钟的预测图
T+12 = 最后输入图之后 12 分钟的预测图
T+18 = 最后输入图之后 18 分钟的预测图
```

每个时效单独评分；`overall` 为三个时效的总分。

| 指标 | 含义 | 每个时效怎样算 | overall 怎样算 |
| --- | --- | --- | --- |
| MAE | 平均像素误差，越小越好 | 把每张预测图逐像素对照真实图，再把所有像素误差取平均。 | 把三个时效的所有像素误差取平均。 |
| MSE | 大误差惩罚更重，越小越好 | 把每张预测图逐像素对照真实图；误差先平方，再把所有像素取平均。 | 把三个时效的所有像素平方误差取平均。 |
| SSIM | 图形结构和纹理像不像，越接近 1 越好 | 每张预测图与真实图算一次 SSIM，再把所有图的 SSIM 取平均。 | 把三个时效的所有单图 SSIM 取平均。 |

每次报告：`T+6`、`T+12`、`T+18`、`overall`。

### 3.3 SSIM 实现

```text
implementation = skimage.metrics.structural_similarity
library        = scikit-image==0.24.0
data_range     = 1.0
```

### 3.4 阈值与 CSI/POD/FAR

本阶段根据 train 强度统计冻结活动灰度代理阈值：

```text
proxy_activity_threshold = 0.05
threshold_source = train_only
```

train 中非零像素占 0.5723%，灰度值 `≥0.05` 的像素占 0.5476%，即保留约 95.7% 的非零像素，同时过滤 0 附近的少量残留值。统计使用 4,964 张唯一 train 帧、约 8.95 亿像素，未使用 val/test。禁止查看 val/test 结果后调整阈值。

事件定义与公式为：

```text
target_event = target >= τ
pred_event   = pred >= τ

CSI = TP / (TP + FP + FN)
POD = TP / (TP + FN)
FAR = FP / (TP + FP)
```

其中：`TP` 是预测和真实都为活动，`FP` 是预测活动但真实不活动，`FN` 是真实活动但预测不活动。

三项指标的含义和趋势：

| 指标 | 含义 | 取值与趋势 |
| --- | --- | --- |
| CSI | 综合衡量活动区域是否预测准确，同时考虑漏报和误报 | 取值为 0～1，**越大越好**；1 表示没有漏报和误报，0 表示活动区域完全没有预测对 |
| POD | 真实活动像素中，有多少被模型成功预测出来，主要反映漏报情况 | 取值为 0～1，**越大越好**；越大说明漏报越少，1 表示所有真实活动都被预测到 |
| FAR | 模型预测为活动的像素中，有多少实际上不是活动，反映误报情况 | 取值为 0～1，**越小越好**；越小说明误报越少，0 表示没有误报 |

POD 和 FAR 必须结合观察：模型把大量位置都预测为活动，可能让 POD 升高，但也会让 FAR 变差；因此不能只看 POD，CSI 用来综合判断漏报和误报后的整体效果。

汇总规则：

```text
零分母：metric_value = null；metric_status = undefined_zero_denominator。
每个时效：micro 聚合；先合并该时效全部样本的 TP/FP/FN，再计算 CSI/POD/FAR。
overall：先合并 T+6、T+12、T+18 的全部 TP/FP/FN，再计算 CSI/POD/FAR。
```

### 3.5 候选方案决策规则

```text
主指标：验证集（val）overall MAE
辅助指标：验证集（val）overall MSE、验证集（val）overall SSIM
分时效报告：T+6、T+12、T+18 各自报告 MAE、MSE、SSIM
分类指标：T+6、T+12、T+18 和 overall 均报告 CSI、POD、FAR
```

候选方案进入下一轮的条件：相对 Persistence 的验证集（val）MAE 改善至少 0.5%，SSIM 不低于 Persistence 减 0.002，且三个时效均没有 `invalid/failed`。MAE 改善不足 0.5% 时标记 `inconclusive`，不宣称改进。

## 4. 第一层输出与边界测试

评测器自动生成 `metrics.json` 和 `evaluation_manifest.json`。

`metrics.json` 必含：

```text
experiment_id、spec_digest、dataset_version、evaluator_version、split、
per_horizon、overall、threshold、mask_definition、valid_count、
decision_inputs、evaluation_status
```

`evaluation_manifest.json` 记录预测文件哈希、外部真值版本标识/哈希、代码版本、运行时间、依赖版本，作为复跑回执。

边界测试用人工小数据检查裁判不会算错：

| 测试 | 预期 |
| --- | --- |
| 完美预测 | MAE=0、MSE=0、SSIM=1（允许浮点微小误差） |
| 全零预测 | 正常给分，不能自动判错 |
| NaN/Inf、shape 错误 | `invalid`，不给指标 |
| sample_id 缺失/重复/乱序 | `invalid`；乱序但 ID 完整唯一时应能正确对齐 |
| 预测样本不完整 | `invalid`，列出缺失 ID |
| 无活动像素、全活动像素、零分母 | 按当前 CSI/POD/FAR 规则测试；出现零分母时返回预定义的 `null` 状态 |

## 5. 第二层：自主研究指标

第二层读取 `experiment_registry.jsonl`，不直接读取雷达图片。调度器每次实验结束自动写一条记录：

```text
experiment_id、candidate_id、父实验、代码版本、数据版本、split、开始/结束时间、
GPU_seconds、token_count、是否运行完成、是否人工介入、是否自动修复、失败类型、
val 的 MAE/MSE/SSIM、test 是否被访问。
```

| 指标 | 固定定义 |
| --- | --- |
| VCR | 有效且可复现实验数 / 已提交候选实验数 |
| ACR | 无人工介入且有效实验数 / 已提交候选实验数 |
| Error Recovery | 自动恢复成功的可恢复失败数 / 可恢复失败总数 |
| Gain/GPU-hour | 验证集（val）里的relative-MAE gain / (GPU_seconds / 3600) |
| Gain/100K Tokens | 验证集（val）里的relative-MAE gain / (token_count / 100000) |
| Validation-Test Gap | 验证集（val）里的relative-MAE gain − 测试集（test）里的relative-MAE gain |

```text
relative-MAE gain = (MAE_persistence − MAE_candidate) / MAE_persistence
```

GPU 时间必须包含失败和重试。模型搜索期间禁止读 test，此时 `Validation-Test Gap=null`；方案冻结且合法测过 test 后才计算该指标。

第二层指标需要汇总多次实验才有意义，所以在一批实验结束后再汇总报告。

## 6. 运行顺序

```text
调度器指定本次 split 与本规范版本
  → Agent 提出候选模型方案
  → 训练/推理生成 predictions.npz
  → 第一层在完整 val 计算第一层指标 MAE/MSE/SSIM
  → 调度器写实验记录
  → Agent 决定下一次尝试
  → 重复上述步骤，直到本批实验结束
  → 第二层汇总本批实验的第二层指标 VCR/ACR/Gain 等
  → 方案冻结后，第一层只在完整 test 认证一次
  → 第二层生成本轮最终 Auto Research 审计结果
```
