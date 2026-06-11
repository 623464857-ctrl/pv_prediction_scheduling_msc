# 实验脚本索引

## 原则

- **一实验一脚本**：每个子步骤对应独立可执行脚本
- **预测 / 调度分轨**：当前在 `prediction/` 下推进

## 预测方向（prediction）

| 编号 | 目录 | 脚本 | 状态 |
|------|------|------|------|
| P01 | `prediction/step1_data_cleaning_alignment/` | `run_exp_p01_preprocessing.py` | 已完成 |
| P02 | `prediction/step2_baseline_models/` | 见下方分阶段脚本 | 已完成 |

### EXP-P02 分阶段脚本

| 阶段 | 脚本 | 状态 |
|------|------|------|
| 1 | `run_exp_p02_prepare_samples.py` | 已完成 |
| 2a~2e | `run_exp_p02_train_*.py` | 待执行 |
| 3 | `run_exp_p02_summarize_results.py` | 待执行 |

## 调度方向（scheduling）

预留，待预测基础数据就绪后扩展。
