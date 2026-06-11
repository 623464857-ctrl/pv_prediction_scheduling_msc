# EXP-P02：五类基础预测模型对比

## 阶段划分（均已完成）

| 阶段 | 脚本 | 说明 |
|------|------|------|
| 1 | `run_exp_p02_prepare_samples.py` | 样本构造 / 划分 / 标准化 |
| 2a | `run_exp_p02_train_bp.py` | BP 神经网络 |
| 2b | `run_exp_p02_train_svr.py` | SVR |
| 2c | `run_exp_p02_train_rf.py` | Random Forest |
| 2d | `run_exp_p02_train_lstm.py` | LSTM |
| 2e | `run_exp_p02_train_bilstm.py` | BiLSTM |
| 3 | `run_exp_p02_summarize_results.py` | 指标汇总 / 作图 / 结论 |

## 运行命令

```powershell
python experiments/prediction/step2_baseline_models/run_exp_p02_prepare_samples.py
python experiments/prediction/step2_baseline_models/run_exp_p02_train_bp.py
python experiments/prediction/step2_baseline_models/run_exp_p02_train_svr.py
python experiments/prediction/step2_baseline_models/run_exp_p02_train_rf.py
python experiments/prediction/step2_baseline_models/run_exp_p02_train_lstm.py
python experiments/prediction/step2_baseline_models/run_exp_p02_train_bilstm.py
python experiments/prediction/step2_baseline_models/run_exp_p02_summarize_results.py
```

## 关键结果（Site_1 测试集）

| 模型 | MAE | RMSE | R² |
|------|-----|------|-----|
| BiLSTM | **0.0219** | **0.0465** | **0.971** |
| LSTM | 0.0231 | 0.0492 | 0.968 |
| BP | 0.0255 | 0.0545 | 0.961 |
| Random Forest | 0.0283 | 0.0652 | 0.943 |
| SVR | 0.0317 | 0.0677 | 0.939 |

**推荐后续改进基线**：BiLSTM

## 产出路径

- 数据/模型/图表：`data/prediction/step2_baseline_models/`
- 初步结论：`data/prediction/step2_baseline_models/reports/EXP-P02_preliminary_conclusion.md`
- 日志：`logs/prediction/step2_baseline_models/EXP-P02_*.log`
