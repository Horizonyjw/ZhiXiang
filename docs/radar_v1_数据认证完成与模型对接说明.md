# radar_v1 数据认证完成与模型对接说明

## 1. 当前状态

当前雷达数据版本统一为：

```text
data_version = radar_v1
```

在现有 `radar_v1` 基础上，数据侧已经完成 Pilot 级冻结与认证，不再继续修改 v1 的数据规则。

最终认证状态：

```text
contract_status = certified_for_pilot
```

本轮认证通过的关键检查包括：

```text
all_artifacts_exist = true
preflight_pass = true
split_audit_pass = true
train_stats_train_only = true
test_frozen = true
formal_physical_claims_false = true
physical_unit_proxy_not_dbz = true
```

这意味着 `radar_v1` 已经具备进入模型联调和 AutoResearch Pilot 的数据条件。

---

## 2. 已完成的数据工作

### 2.1 已有 radar_v1 基础工作

`radar_v1` 原有工作已经包括：

- 原始雷达 PNG 检查；
- UTC 资料时次解析；
- 时间连续性检查；
- 缺测断点记录；
- 5→3 监督样本构建；
- Train / Validation / Test 时间顺序划分；
- 352×512 单通道标准化；
- 数据索引与 SQLite 数据库；
- DataLoader 接口；
- handoff 联调样本；
- GitHub 协作版本。

当前固定任务参数：

```text
Tin  = 5
Tout = 3
C    = 1
H    = 352
W    = 512
frame_interval = 6 min
dtype = float32
value_range = [0,1]
physical_unit = normalized_grayscale_proxy_NOT_dBZ
```

模型侧统一张量接口：

```text
inputs  [B,5,1,352,512]
targets [B,3,1,352,512]
```

---

### 2.2 数据版本与主索引冻结

已经确认完整数据主索引以：

```text
radar_v1/03_index/frames.csv
radar_v1/03_index/samples.csv
```

为准。

当前完整数据规模：

```text
frame_count  = 7396
sample_count = 7302

train = 5097
val   = 1472
test  = 733
```

`02_standardized` 中的 CSV 仅用于标准化覆盖与路径信息，不再作为整个数据集总规模的唯一依据。

---

### 2.3 Frame / Sample Manifest

已经生成 Pilot 认证层的固定索引：

```text
radar_v1/07_pilot_contract/frame_manifest.csv
radar_v1/07_pilot_contract/sample_manifest.csv
```

其中：

- `frame_manifest.csv` 用于逐帧追踪；
- `sample_manifest.csv` 用于逐样本追踪；
- 每个 `sample_id` 可以追踪到输入 5 帧、目标 3 帧、时间、split 和 segment。

当前真实 `samples.csv` 字段已经完成映射：

```text
input_rel_paths_json  -> input_paths
target_rel_paths_json -> target_paths
input_times_json      -> input_timestamps
target_times_json     -> target_timestamps
```

---

### 2.4 Split 与数据泄漏审计

已经完成以下检查：

- `sample_id` 唯一；
- 输入帧数固定为 5；
- 目标帧数固定为 3；
- 输入和目标时间顺序正确；
- 样本不跨缺测断点；
- Train / Val / Test 按时间顺序划分；
- 滑动窗口不跨 split 边界；
- 不同 split 不共享同一帧；
- Test 已冻结，不参与搜索。

对应输出：

```text
radar_v1/07_pilot_contract/split_audit.json
```

后续 AutoResearch 阶段严格遵循：

```text
Train      -> 模型训练
Validation -> 搜索、筛选、诊断
Test       -> 最终独立认证
```

禁止在搜索阶段使用 Test 进行：

- 超参数选择；
- Data Action 选择；
- 阈值选择；
- checkpoint 选择；
- 候选方案比较。

---

### 2.5 Normalization 合同

现有标准化流程已经正式记录为：

```text
RGBA
-> RGB / 255
-> 单通道灰度
-> alpha 无效显示区域置 0
-> resize 到 352×512
-> float32
```

当前数据表示仅定义为：

```text
normalized_grayscale_proxy_NOT_dBZ
```

不进行正式 dBZ 物理解释。

对应输出：

```text
radar_v1/07_pilot_contract/normalization.json
```

当前已明确：

```text
formal_physical_claims = false
```

PNG→dBZ 映射、CRS / georeferencing 等继续作为后续问题调查，但不阻塞本次 Pilot。

---

### 2.6 Train-only 强度统计

已经仅使用 Train 标准化帧完成灰度强度统计。

统计包括：

```text
min / max
mean / std
non-zero ratio
P50 / P75 / P90 / P95 / P99
histogram
```

对应输出：

```text
radar_v1/07_pilot_contract/train_intensity_statistics.json
```

并已经确认：

```text
val_used  = false
test_used = false
```

该统计用于给评测侧选择代理 activity threshold / event threshold 提供依据。

数据侧不根据 Test 选择阈值。

---

### 2.7 数据 Manifest 与 SHA-256

已经生成认证相关文件的：

- 文件路径；
- 文件大小；
- SHA-256；

对应输出：

```text
radar_v1/07_pilot_contract/data_manifest.json
```

用于后续确认不同实验使用的是同一版 `radar_v1` 数据与认证文件。

---

### 2.8 Pilot Data Contract

已经生成：

```text
radar_v1/07_pilot_contract/radar_v1_pilot_data_contract.json
```

并最终通过：

```text
radar_v1/07_pilot_contract/certification_report.json
```

认证状态为：

```text
certified_for_pilot
```

---

## 3. 模型 handoff 已完成的检查

当前联调文件：

```text
radar_v1/04_handoff/handoff_samples.npz
```

已经完成接口验证。

文件字段：

```text
inputs
targets
metadata_json
```

实际验证结果：

```text
inputs_shape  = [16,5,1,352,512]
targets_shape = [16,3,1,352,512]

inputs_dtype  = float32
targets_dtype = float32

shape_ok = true
dtype_ok = true
```

本批 handoff 样本实际数值范围约为：

```text
inputs_range  = [0.0, 0.4901106655597687]
targets_range = [0.0, 0.4901106655597687]
```

注意：这只是当前 16 个 handoff 样本的实际范围，不改变数据合同中的合法总体范围 `[0,1]`。

---

# 4. 对模型负责人的下一步对接需求

## 4.1 第一优先级：真实模型 Forward 联调

请模型侧直接读取：

```text
radar_v1/04_handoff/handoff_samples.npz
```

完成真实模型前向传播：

```text
inputs
[B,5,1,352,512]
        ↓
      model
        ↓
predictions
[B,3,1,352,512]
```

要求：

1. 模型能够正常读取 `float32` 输入；
2. 不修改数据侧统一输入接口；
3. 模型输出与 target 在时间、通道和空间维度上完全一致；
4. 若模型内部需要不同张量布局，由模型侧 Adapter 转换，不修改 `radar_v1`。

例如模型内部需要：

```text
[B,5,352,512]
```

则应采用：

```text
radar_v1 标准接口
        ↓
Model Adapter
        ↓
模型内部格式
```

而不是重新生成另一套数据。

---

## 4.2 统一正式预测输出

模型侧 forward 跑通后，所有模型统一输出：

```text
predictions.npz
```

推荐字段：

```text
inputs
targets
predictions
metadata_json
```

统一 shape：

```text
inputs       [N,5,1,352,512] float32
targets      [N,3,1,352,512] float32
predictions  [N,3,1,352,512] float32
```

`metadata_json` 至少保留：

```text
sample_id
split
input_times
target_times
```

不要出现：

```text
U-Net 一套输出格式
FNO 一套输出格式
AutoResearch 又一套输出格式
```

所有模型必须走相同结果接口。

---

## 4.3 Prediction Contract 检查

模型侧输出 `predictions.npz` 后，数据侧使用：

```bash
python radar_v1_pilot_certification_tools_v3/scripts/10_validate_predictions.py --predictions <path_to_predictions.npz>
```

应通过：

```text
N_aligned
target_prediction_shape_aligned
prediction_rank_5
prediction_core_shape
finite_predictions
```

最终目标：

```text
predictions.npz contract PASS.
```

---

## 4.4 模型侧不得自行修改的数据规则

模型负责人原则上不要自行修改：

```text
data_version = radar_v1

Tin  = 5
Tout = 3

C = 1
H = 352
W = 512

normalization = [0,1]
split = fixed chronological split
```

禁止：

- 随机重新划分 Train / Val / Test；
- 重新定义 sample_id；
- 跨 continuity break 重新构造样本；
- 修改 normalization 后仍称为 `radar_v1`；
- 修改 Tin / Tout 后仍复用 `radar_v1` 名称。

如确实需要改变基础数据规则，应建立：

```text
radar_v2
```

而不是覆盖 v1。

---

# 5. 对评测侧的下一步需求

当前数据侧已经提供：

```text
train_intensity_statistics.json
```

下一步需要评测负责人冻结统一 Evaluation Policy。

至少明确：

```text
overall MAE
overall SSIM

masked MAE
masked SSIM

CSI
POD
FAR

T+6 min
T+12 min
T+18 min
```

并统一：

```text
activity_threshold
event_thresholds
```

由于当前数据不是物理 dBZ：

```text
不能直接把 20/30/40 dBZ 等物理阈值套到当前 [0,1] 灰度代理量。
```

阈值来源必须注明：

```text
threshold_source = train_only
```

Test 不参与阈值选择。

---

# 6. 从现在开始的团队流程

当前已经完成：

```text
Radar v1 数据治理
        ↓
Pilot 数据认证
        ↓
Handoff Contract PASS
```

接下来：

```text
① 模型真实 Forward
        ↓
② 统一 predictions.npz
        ↓
③ Prediction Contract PASS
        ↓
④ Evaluation Policy 冻结
        ↓
⑤ Persistence Baseline
        ↓
⑥ 学习型 Baseline
        ↓
⑦ AutoResearch Train + Validation 搜索
        ↓
⑧ Freeze Candidate
        ↓
⑨ Held-out Test
        ↓
⑩ Certification
```

整个 AutoResearch 搜索过程中：

```text
Train + Validation 可访问
Test 不可访问
```

最终候选方案冻结之后，再进入独立 Test 认证。

---

# 7. 当前数据侧里程碑总结

截至目前，数据侧已经从：

```text
数据清洗 / 样本构造
```

推进到：

```text
可冻结
+
可追踪
+
可复验
+
可用于独立认证
```

当前可以向团队同步为：

> Radar v1 已完成 Pilot 级数据认证，数据划分、样本索引、归一化规则、Train-only 强度统计和数据合同均已冻结，Test 已隔离；handoff 输入输出接口验证通过。下一阶段数据侧主要负责模型接口兼容、prediction contract 检查和统一评测规则对接，不再重新清洗 radar_v1。

---

# 8. GitHub 更新建议

## 8.1 不要把大量本地标准化帧提交 GitHub

当前仓库协作原则是不提交真实大数据、模型权重和大量预测结果。

因此不要直接执行：

```bash
git add .
```

因为当前本地存在大量：

```text
radar_v1/02_standardized/frames/train/*.npy
```

这些文件现在显示为 untracked，如果执行 `git add .` 会把大量数据一起加入暂存区。

本次建议只提交：

```text
radar_v1/07_pilot_contract/
radar_v1_pilot_certification_tools_v3/
本说明文档
```

---

## 8.2 建议忽略本地标准化帧

打开：

```text
radar_v1/.gitignore
```

增加：

```gitignore
# Local standardized radar frame payloads
02_standardized/frames/
```

如果希望保留目录结构，可以改成只忽略 `.npy`：

```gitignore
02_standardized/frames/**/*.npy
```

之后执行：

```bash
git status
```

大量 `.npy` 应不再显示为 untracked。

---

## 8.3 清理旧认证工具目录

当前已经确认 v3 可用，因此建议不要同时把 v1/v2/v3 三套工具提交。

如果旧工具不再使用，可以本地删除：

Windows CMD：

```bat
rmdir /s /q radar_v1_pilot_certification_tools
rmdir /s /q radar_v1_pilot_certification_tools_v2
```

保留：

```text
radar_v1_pilot_certification_tools_v3/
```

---

## 8.4 正确的 Git 提交流程

先检查：

```bash
git status
```

然后只 add 需要提交的轻量内容：

```bash
git add radar_v1/07_pilot_contract
git add radar_v1_pilot_certification_tools_v3
git add radar_v1/.gitignore
git add docs/radar_v1_数据认证完成与模型对接说明.md
```

再次检查：

```bash
git status
```

此时应该在：

```text
Changes to be committed
```

下面看到待提交文件。

然后：

```bash
git commit -m "certify radar_v1 and add model handoff contract"
```

最后：

```bash
git push origin <当前分支名>
```

如果当前就在 `main`：

```bash
git push origin main
```

如果当前在认证分支，例如：

```text
data/radar-v1-pilot-certification
```

则：

```bash
git push origin data/radar-v1-pilot-certification
```

---

## 8.5 当前 “no changes added to commit” 的含义

如果终端显示：

```text
no changes added to commit
(use "git add" and/or "git commit -a")
```

它并不是数据认证失败，也不是 GitHub 报错。

它的含义只是：

> 当前有新文件或修改文件，但还没有执行 `git add` 把它们放进 Git 暂存区。

正确顺序必须是：

```text
文件变化
   ↓
git add
   ↓
staged
   ↓
git commit
   ↓
git push
```

所以不要重复直接 `git commit`，先执行针对性的 `git add`。

---

# 9. 推荐本次 GitHub 最终新增结构

```text
ZhiXiang/
├── docs/
│   └── radar_v1_数据认证完成与模型对接说明.md
│
├── radar_v1/
│   ├── 02_standardized/
│   ├── 03_index/
│   ├── 04_handoff/
│   ├── 05_metadata/
│   ├── 06_code/
│   └── 07_pilot_contract/
│       ├── preflight_report.json
│       ├── frame_manifest.csv
│       ├── sample_manifest.csv
│       ├── split_audit.json
│       ├── normalization.json
│       ├── train_intensity_statistics.json
│       ├── data_manifest.json
│       ├── radar_v1_pilot_data_contract.json
│       └── certification_report.json
│
└── radar_v1_pilot_certification_tools_v3/
```

这样既能保留可复验的认证证据，又不会把大量雷达数据直接塞进 Git 仓库。
