# 实验脚本索引

## 原则

- **一实验一脚本**：每个子步骤对应独立可执行脚本
- **预测 / 调度分轨**：当前在 `prediction/` 下推进
- **当前主线**：P01 → P04（Optuna + CNN-BiLSTM 残差）→ P05（历史产出）

## 预测方向（prediction）

| 编号 | 目录 | 脚本 | 状态 |
|------|------|------|------|
| P01 | `prediction/step1_data_cleaning_alignment/` | `run_exp_p01_preprocessing.py` | ✅ 已完成 |
| P04 | `prediction/step4_optuna_hybrid/` | `run_exp_p04_*.py` | 🔄 已重构，待重跑 |
| P05 | `prediction/step5_new_experiments/` | （脚本已移除，产出保留） | ✅ 历史产出 |

### EXP-P01 辅助脚本

| 脚本 | 功能 |
|------|------|
| `repair_preprocessed_power.py` | 修复已生成 CSV 中的辐照-功率不一致 |
| `check_site4_integrity.py` | Site_4 数据完整性检查 |

### EXP-P04 分阶段脚本

| 阶段 | 脚本 | 功能 |
|------|------|------|
| 1 | `run_exp_p04_prepare_samples.py` | 构造 `h{H}_lb{LB}/` 滑动窗口样本（13 维 step1 特征，残差目标） |
| 2 | `run_exp_p04_optuna.py` | Optuna-AFSA 混合消融搜索（S2-S6，无 S1 Random） |
| 3 | `run_exp_p04_train_final.py` | 最优参数最终训练 |
| 4 | `run_exp_p04_reproduce.py` | 多 seed（42/43/44/45/46）复现，汇总均值/标准差 |
| 5 | `run_exp_p04_report.py` | 生成 Markdown 报告 + 图表 |
| 6 | `run_exp_p04_check_pipeline.py` | 读取各步审计记录与日志，逐步 PASS/FAIL 排查 |
| — | `exp_p04_step_audit.py` | 审计模块（各步脚本自动调用，无需手动运行） |

**审计产出（每步运行后自动写入）：**
- `data/prediction/step4_optuna_hybrid/audit/h{H}/{step}.json` — 单步结构化结果
- `data/prediction/step4_optuna_hybrid/audit/h{H}/pipeline_manifest.json` — 流水线总览
- `logs/prediction/step4_optuna_hybrid/PIPELINE_AUDIT_h{H}.log` — 可读的逐步追加日志

**运行示例：**
```powershell
python -m experiments.prediction.step4_optuna_hybrid.run_exp_p04_prepare_samples --horizon 1
python -m experiments.prediction.step4_optuna_hybrid.run_exp_p04_optuna --horizon 1
python -m experiments.prediction.step4_optuna_hybrid.run_exp_p04_train_final --horizon 1
python -m experiments.prediction.step4_optuna_hybrid.run_exp_p04_reproduce --horizon 1
python -m experiments.prediction.step4_optuna_hybrid.run_exp_p04_report --horizon 1
python -m experiments.prediction.step4_optuna_hybrid.run_exp_p04_check_pipeline --horizon 1
```
`--horizon` 支持 `1` (15min) / `4` (1h) / `16` (4h)

## 调度方向（scheduling）

预留，待预测基础数据就绪后扩展。
