# EXP-P02 产出文件说明

## 阶段 1：样本与划分

见 `samples/` 目录（`*.npy`, `site1_window_meta.json`, `scaler_params.json`）。

## 阶段 2：模型训练

| 路径 | 说明 |
|------|------|
| `models/bp.pt` | BP 神经网络权重 |
| `models/svr.joblib` | SVR 模型（RBF，C=1, ε=0.01） |
| `models/rf.joblib` | Random Forest 模型 |
| `models/lstm.pt` / `models/bilstm.pt` | LSTM / BiLSTM 权重 |
| `metrics/*_train_history.csv` | BP/LSTM/BiLSTM 训练历史（epoch, train_loss, val_loss） |
| `predictions/*_test.csv` | 测试集预测（timestamp, y_true, y_pred, model_name） |

> SVR 在 10000 训练子样本上拟合（固定种子），验证/测试使用全量样本。

## 阶段 3：结果汇总

| 路径 | 说明 |
|------|------|
| `metrics/baseline_comparison_metrics.csv` | 五模型 MAE/RMSE/MAPE/R² 对比 |
| `figures/loss_bp.png` / `loss_lstm.png` / `loss_bilstm.png` | 官方损失曲线（与 `metrics/*_train_history.csv` 一致） |
| `figures/pred_*.png` | 各模型测试集预测曲线（5 天窗口） |
| `figures/pred_all_models_overlay.png` | 五模型 + 真值同图对比 |
| `figures/metrics_bar_comparison.png` | 指标柱状图 |
| `reports/EXP-P02_preliminary_conclusion.md` | 初步结论 |
