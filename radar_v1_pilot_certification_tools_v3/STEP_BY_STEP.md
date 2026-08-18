# 从 radar_v1 到 `certified_for_pilot`：逐步操作

## Step 0：冻结工作方式，不改 v1 规则

确认本轮不修改：

- Tin=5 / Tout=3；
- 单通道 352×512；
- `[0,1]` 灰度代理量；
- 当前 train/val/test；
- 缺测断点规则；
- sample_id / 时间索引规则。

如要修改以上任一项，停止本流程并建立 `radar_v2`。

---

## Step 1：拉取仓库并建立工作分支

```bash
git clone https://github.com/Horizonyjw/ZhiXiang.git
cd ZhiXiang
git checkout -b data/radar-v1-pilot-certification
```

将本工具目录放到仓库根目录。

---

## Step 2：预检 v1 已有交付

```bash
python radar_v1_pilot_certification_tools/scripts/01_preflight.py --repo-root .
```

检查：

- 已有 CSV/JSON/NPZ 是否存在；
- frame_count 是否为 7396；
- sample_count 是否为 7302；
- train/val/test 是否为 5097/1472/733；
- 13 个 continuity breaks 是否可访问。

如果失败，先修复“索引/路径/配置”问题，不要继续认证。

---

## Step 3：生成冻结 manifest

```bash
python radar_v1_pilot_certification_tools/scripts/02_build_manifests.py --repo-root .
```

得到：

- `frame_manifest.csv`
- `sample_manifest.csv`

这两个文件是后续实验追溯的稳定入口。

---

## Step 4：审计 sample 与 split

```bash
python radar_v1_pilot_certification_tools/scripts/03_audit_samples_and_splits.py --repo-root .
```

重点关注：

- `sample_id` 是否唯一；
- 输入和目标时间是否单调；
- 5+3 时间窗口是否连续；
- 是否跨 continuity break；
- 是否跨 train/val/test；
- 同一帧是否出现在不同 split；
- test 是否可单独识别并冻结。

输出：

`split_audit.json`

只有关键泄漏检查为 PASS，才继续。

---

## Step 5：固化 normalization 说明

```bash
python radar_v1_pilot_certification_tools/scripts/04_build_normalization.py --repo-root .
```

它不会重新处理图像，只会把现有 v1 的标准化规则写成机器可读合同：

`normalization.json`

---

## Step 6：只使用 train 计算强度统计

```bash
python radar_v1_pilot_certification_tools/scripts/05_train_intensity_statistics.py --repo-root .
```

脚本优先从 `frame_manifest.csv` 中只选择 `split=train` 的**唯一标准化帧**，计算：

- min / max；
- mean / std；
- non-zero ratio；
- P50/P75/P90/P95/P99；
- 直方图。

输出：

`train_intensity_statistics.json`

**不要用 val/test 计算活动区域阈值。**

---

## Step 7：生成 SHA-256 数据清单

```bash
python radar_v1_pilot_certification_tools/scripts/06_build_data_manifest.py --repo-root .
```

输出：

`data_manifest.json`

默认只 hash GitHub 中当前可访问的 v1 索引、元数据、标准化索引和 handoff 文件。
如果标准化帧实际存在本地，也会把可访问帧纳入 hash。

---

## Step 8：生成 pilot data contract

```bash
python radar_v1_pilot_certification_tools/scripts/07_build_contract.py --repo-root .
```

输出：

`radar_v1_pilot_data_contract.json`

合同会明确：

- 数据版本；
- tensor 形状；
- split 数量；
- 当前单位不是 dBZ；
- test_frozen；
- known limitations。

---

## Step 9：最终认证

```bash
python radar_v1_pilot_certification_tools/scripts/08_certify.py --repo-root .
```

输出：

`certification_report.json`

当关键检查均通过时：

```text
contract_status = certified_for_pilot
```

否则：

```text
contract_status = not_certified
```

不要手工把失败状态改成 certified。

---

## Step 10：模型侧 handoff 联调

```bash
python radar_v1_pilot_certification_tools/scripts/09_validate_handoff.py --repo-root .
```

模型负责人应继续做真实 batch forward：

```text
[B,5,1,352,512]
      ↓ model
[B,3,1,352,512]
```

数据侧只负责接口一致性，不替模型负责人修改模型。

---

## Step 11：统一 predictions.npz

模型侧正式评测输出统一为：

```text
inputs
targets
predictions
metadata_json
```

检查：

```bash
python radar_v1_pilot_certification_tools/scripts/10_validate_predictions.py \
  --predictions results/<exp_id>/predictions.npz
```

---

## Step 12：等待评测负责人冻结阈值

将：

`templates/evaluation_policy.template.json`

复制到项目公共配置目录，并由评测负责人填写：

- activity threshold；
- CSI/POD/FAR thresholds；
- 各 lead time 是否共用阈值。

数据侧只提供 train-only 强度统计，不自行选最终阈值。

---

## Step 13：Git 冻结

通过认证后：

```bash
git add radar_v1/07_pilot_contract
git add radar_v1_pilot_certification_tools
git commit -m "certify radar_v1 for AutoResearch pilot"
git push origin data/radar-v1-pilot-certification
```

合并主分支后：

```bash
git tag -a radar-v1-pilot-certified -m "radar_v1 certified for AutoResearch pilot"
git push origin radar-v1-pilot-certified
```

---

## Step 14：AutoResearch 搜索阶段

只允许使用：

- train：训练；
- val：候选方案筛选；
- train-only statistics：阈值/数据先验的来源。

禁止：

- 用 test 选超参数；
- 用 test 选 Data Action；
- 用 test 选择活动区域阈值；
- 用 test 反复比较候选。

候选冻结后，才进入独立 Certification。

