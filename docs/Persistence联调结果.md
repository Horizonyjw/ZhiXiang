# Persistence 联调结果

> **状态：实现与随机张量接口验证已完成；真实数据联调未执行。**
>
> Persistence 代码、保存脚本与随机张量前向已就绪。因尚未获得获授权真实雷达连续样例，本文件不登记虚构指标或“联调成功”结论。

## 1. 联调前置条件

- [ ] 已获得获授权的真实连续雷达样例；
- [ ] 已确认数据单位、数值范围、缺失值与无回波的定义；
- [ ] 已确认 Tin、Tout、时间间隔、空间处理与回波阈值；
- [x] 已实现 Persistence，并以统一输入输出接口通过随机张量检查；
- [x] 已实现预测保存（`train/run_persistence.py` → `predictions.npz`）；评测读取与可视化仍待对应模块；
- [ ] 已保存真实数据联调的配置、日志和代码版本。

## 2. 待登记实验信息

| 项目 | 实际值 |
| --- | --- |
| 实验编号 | 【待真实数据运行后填写】 |
| 日期与负责人 | 2026-08-09 / Persistence 负责人（代码已就绪） |
| 数据版本与划分 | 【待实际运行后填写】 |
| 模型 | Persistence（`models/persistence.py`） |
| 输入 / 预测设置 | 占位 Tin=5, Tout=5, C=1, H=W=128（待确认） |
| 回波阈值与单位 | 【待实际运行后填写】 |
| 运行环境 | 见 `docs/环境安装与启动说明.md` |
| 推理时长 | 【待真实数据运行后填写】 |
| MAE / MSE / CSI / POD / FAR | 【待真实数据 + 评测模块后填写】 |
| 配置、预测、日志、图像位置 | 代码：`configs/persistence_baseline.yaml`；真实结果目录待生成 |
| 结论与问题 | 真实样例未到位，联调未闭环 |

## 3. 预期结果目录

实际联调完成后，结果应按以下结构保存：

~~~text
results/<experiment_id>/
├── config.yaml
├── predictions.npz
├── metrics.json 或 metrics.csv
├── run.log
└── figures/
~~~

真实样本到位后执行：

~~~powershell
python -m train.run_persistence --npz path/to/real_batch.npz
~~~

> 未获得真实数据前不创建虚构预测文件、指标数值、图像或“成功”结论。
