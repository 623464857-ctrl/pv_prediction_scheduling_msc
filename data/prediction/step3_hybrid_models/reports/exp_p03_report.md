# EXP-P03 混合深度学习模型对比 — 实验报告

## 1. 实验概述

- 实验编号: EXP-P03
- 站点: Site_1
- 任务: 光伏功率归一化值预测（power_pu）
- 划分: 时序 70% 训练 / 14% 验证 / 30% 测试
- 输入窗口: lookback=16, horizon=1
- 特征数: 13

## 2. 模型汇总

### LSTM

- MAE: 0.026526
- RMSE: 0.055880
- MAPE: 20.84%
- R²: 0.957510

### BiLSTM

- MAE: 0.026995
- RMSE: 0.052554
- MAPE: 20.94%
- R²: 0.962417

### CNN-LSTM

- MAE: 0.024367
- RMSE: 0.047724
- MAPE: 21.08%
- R²: 0.969009

### CNN-BiLSTM

- MAE: 0.025944
- RMSE: 0.049592
- MAPE: 21.44%
- R²: 0.966535

## 3. 结论

- 测试集 RMSE 最低模型: **CNN-LSTM** (RMSE=0.047724)
- 测试集 R² 最高模型: **CNN-LSTM** (R²=0.969009)

## 4. 产出文件

- `data/prediction/step3_hybrid_models/metrics/*.csv`
- `data/prediction/step3_hybrid_models/predictions/*.csv`
- `data/prediction/step3_hybrid_models/figures/*.png`
- `data/prediction/step3_hybrid_models/models/*.pt`
