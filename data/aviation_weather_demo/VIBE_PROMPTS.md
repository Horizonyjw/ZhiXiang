# 给 ChatGPT / Cursor / VS Code Agent 的提示词

## 修复报错

我是一名技术初学者。下面是完整终端报错：
【粘贴报错】

请先解释根因，只做最小修改，不重构技术栈，不删除数据库，不执行 `docker compose down -v`，最后告诉我重新运行哪条命令。

## 适配真实雷达文件名

我的三个真实雷达文件名是：
【粘贴三个文件名】

请只修改 `app/radar.py` 的 `parse_time()`，让三个文件名时次都能正确解析。
不要改数据库结构、样本规则、网页或 DataLoader。

## 检查全量样本

请读取 `reports/quality_summary_dataset_*.json` 和 `sample_manifest_dataset_*.csv`，告诉我：
1. 实际有效帧数；
2. 缺失帧数和缺测率；
3. 连续片段数；
4. 全量 5→3 样本数；
5. train / validation / test 数量；
6. 与材料基准 7396 / 7441 / 45 / 0.605% / 14 是否一致。

不要用材料中的数字替代程序实际扫描结果。
