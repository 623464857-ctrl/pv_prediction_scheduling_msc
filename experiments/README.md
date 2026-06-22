# 实验脚本索引

## 原则

- **一实验一脚本**：每个子步骤对应独立可执行脚本
- **预测 / 调度分轨**：当前在 `prediction/` 下推进

## 预测方向（prediction）

| 编号 | 目录 | 脚本 | 状态 |
|------|------|------|------|
| P01 | `prediction/step1_data_cleaning_alignment/` | `run_exp_p01_preprocessing.py` | ✅ 已完成 |
| P02 | `prediction/step2_baseline_models/` | 见下方分阶段脚本 | ✅ 已完成 |
| P03 | `prediction/step3_hybrid_models/` | 多脚本（见 step3 README） | ✅ 已完成 |
| P04 | `prediction/step4_optuna_hybrid/` | `run_exp_p04_*.py` | ✅ 已完成 |

### EXP-P04 分阶段脚本

| 阶段 | 脚本 | 功能 |
|------|------|------|
| 1 | `run_exp_p04_prepare_samples.py` | 构造 h1/h4/h16 滑动窗口样本 + 特征工程 |
| 2 | `run_exp_p04_optuna.py` | Optuna 超参搜索（8 trials/模型） |
| 3 | `run_exp_p04_final_train.py` | 最优参数最终训练 |
| 4 | `run_exp_p04_reproduce.py` | 多 seed（42/43/44）复现，汇总均值/标准差 |
| 5 | `run_exp_p04_report.py` | 生成 Markdown 报告 + 图表 |

**运行示例：**
```powershell
python -m experiments.prediction.step4_optuna_hybrid.run_exp_p04_prepare_samples --horizon 1
python -m experiments.prediction.step4_optuna_hybrid.run_exp_p04_optuna --horizon 1
python -m experiments.prediction.step4_optuna_hybrid.run_exp_p04_final_train --horizon 1
python -m experiments.prediction.step4_optuna_hybrid.run_exp_p04_reproduce --horizon 1
python -m experiments.prediction.step4_optuna_hybrid.run_exp_p04_report --horizon 1
```
`--horizon` 支持 `1` (15min) / `4` (1h) / `16` (4h)

## 调度方向（scheduling）

预留，待预测基础数据就绪后扩展。
