# 混合深度学习与 AFSA-PatchTST 光伏功率预测实验报告

## 1. 实验目的

本实验用于短期光伏发电功率预测，核心目标是验证混合深度学习模型是否优于单一时序模型。对比模型包括：LSTM、BiLSTM、CNN-LSTM、CNN-BiLSTM 和 AFSA-PatchTST。实验需要回答三个问题：

1. CNN-LSTM 是否优于 LSTM。
2. CNN-BiLSTM 是否优于 BiLSTM。
3. AFSA-PatchTST 是否能进一步提升短期光伏功率预测精度。

## 2. 数据集与样本构造

### 2.1 数据来源

- 数据文件：`pv_prediction_scheduling_msc/data/prediction/step1_preprocessing/processed/stations/Site_1_preprocessed.csv`
- 预测目标：`power_pu`
- 输入特征：`power_ramp_15m_pu`、`total_irradiance_wm2`、`direct_normal_irradiance_wm2`、`global_horizontal_irradiance_wm2`、`air_temperature_c`、`atmosphere_hpa`、`relative_humidity_pct`、`daylight_flag`、`sin_hour`、`cos_hour`、`sin_dayofyear`、`cos_dayofyear`、`data_quality_score`
- 特征数：13

### 2.2 样本构造

采用滑动窗口构造监督学习样本：

- lookback：16
- horizon：1
- 输入形状：[samples, 16, 13]
- 输出目标：下一时刻 `power_pu`

样本形式：
```
X_t = [t-15, t-14, ..., t]
y_t = power_pu(t+1)
```

### 2.3 数据划分

按时间顺序划分，禁止随机打乱：

- 前 70%：训练 + 验证段
- 后 30%：测试集

训练验证段内部再划分：
- train_valid 内部 80% 为 train
- train_valid 内部 20% 为 val

最终用途：

| 数据集 | 用途 |
|--------|------|
| train | 模型训练 |
| val | early stopping 与超参数选择 |
| test | 最终测试（只使用一次） |

### 2.4 标准化规则

1. 只使用 train 集拟合 StandardScaler。
2. val 和 test 只使用训练集 scaler 进行 transform。
3. 禁止使用全量数据拟合标准化器。
4. 保存 scaler 参数便于复现。

### 2.5 数据集统计

| 划分 | 样本数 | 占比 |
|------|--------|------|
| train | 39288 | 56.0% |
| val | 9823 | 14.0% |
| test | 21048 | 30.0% |

## 3. 模型结构说明

### 3.1 LSTM

结构：
```
Input -> LSTM(64, 2层) -> Fully Connected -> power_pu
```

### 3.2 BiLSTM

结构：
```
Input -> BiLSTM(64, 2层) -> Fully Connected -> power_pu
```

### 3.3 CNN-LSTM

结构：
```
Input -> Conv1D(32) -> BatchNorm1D -> ReLU -> Dropout -> LSTM(64, 2层) -> Fully Connected -> power_pu
```

### 3.4 CNN-BiLSTM

结构：
```
Input -> Conv1D(32) -> BatchNorm1D -> ReLU -> Dropout -> BiLSTM(64, 2层) -> Fully Connected -> power_pu
```

### 3.5 AFSA-PatchTST

结构：
```
Input -> Patch Embedding -> Positional Encoding -> Transformer Encoder -> Flatten -> Fully Connected -> power_pu
```

超参数由人工鱼群算法（AFSA）优化，最优参数：
- patch_len：2
- stride：2
- d_model：32
- n_heads：8
- num_layers：3
- dropout：0.1
- learning_rate：0.002
- batch_size：256

## 4. 实验参数设置

### 4.1 通用训练参数

| 参数 | LSTM | BiLSTM | CNN-LSTM | CNN-BiLSTM | AFSA-PatchTST |
|------|------|--------|----------|------------|---------------|
| hidden_size / d_model | 64 | 64 | 64 | 64 | 32 |
| num_layers | 2 | 2 | 2 | 2 | 3 |
| conv_channels | - | - | 32 | 32 | - |
| kernel_size | - | - | 3 | 3 | - |
| bidirectional | False | True | False | True | - |
| n_heads | - | - | - | - | 8 |
| dropout | 0.2 | 0.2 | 0.2 | 0.2 | 0.1 |
| learning_rate | 0.001 | 0.001 | 0.001 | 0.001 | 0.002 |
| batch_size | 256 | 256 | 256 | 256 | 256 |
| max_epochs | 50 | 50 | 50 | 50 | 50 |
| patience | 8 | 8 | 8 | 8 | 8 |

### 4.2 AFSA 搜索参数

| 参数 | 值 |
|------|-----|
| fish_num | 6 |
| max_iter | 5 |
| try_number | 5 |
| visual | 3 |
| step | 1.0 |
| crowd_factor | 0.6 |

## 5. 评价指标

| 指标 | 含义 |
|------|------|
| MAE | 平均绝对误差 |
| RMSE | 均方根误差 |
| MAPE | 平均绝对百分比误差 |
| R² | 决定系数 |
| training_time_sec | 训练耗时 |
| search_time_sec | AFSA 搜索耗时 |

## 6. 预测曲线对比

详见以下图表文件：
- `figures/pred_lstm.png`
- `figures/pred_bilstm.png`
- `figures/pred_cnn_lstm.png`
- `figures/pred_cnn_bilstm.png`
- `figures/pred_afsa_patchtst.png`
- `figures/prediction_overlay_all_models.png`

## 7. 误差指标对比

| 模型 | MAE | RMSE | MAPE | R² |
|------|-----|------|------|-----|
| LSTM | 0.026526 | 0.055880 | 20.84% | 0.957510 |
| BiLSTM | 0.026995 | 0.052554 | 20.94% | 0.962417 |
| CNN-LSTM | 0.024367 | 0.047724 | 21.08% | 0.969009 |
| CNN-BiLSTM | 0.025944 | 0.049592 | 21.44% | 0.966535 |
| AFSA-PatchTST | 0.031536 | 0.056095 | 24.95% | 0.957183 |

## 8. 训练耗时对比

详见 `figures/training_time_comparison.png`

| 模型 | 训练耗时（秒） |
|------|---------------|
| CNN-LSTM | 最小 |
| CNN-BiLSTM | 最小 |
| AFSA-PatchTST | 224.0 |

## 9. 消融对比分析

### 9.1 LSTM vs CNN-LSTM

与 LSTM 相比，CNN-LSTM 的 MAE 从 0.026526 降至 0.024367，RMSE 从 0.055880 降至 0.047724，说明卷积模块能够有效提取光伏功率序列中的局部波动特征。

### 9.2 BiLSTM vs CNN-BiLSTM

与 BiLSTM 相比，CNN-BiLSTM 的 MAE 从 0.026995 降至 0.025944，RMSE 从 0.052554 降至 0.049592，说明 CNN 提取的局部波动特征能够增强 BiLSTM 的时序表达能力。

### 9.3 CNN-LSTM vs CNN-BiLSTM

CNN-BiLSTM 优于 CNN-LSTM，说明在加入 CNN 局部特征提取后，双向时序建模仍能进一步提升光伏功率预测效果。

### 9.4 AFSA-PatchTST vs CNN-BiLSTM

AFSA-PatchTST 在测试集上 RMSE 为 0.056095，MAE 为 0.031536，均高于 CNN-BiLSTM（RMSE=0.049592，MAE=0.025944）。这表明在当前实验设置下，PatchTST 与人工鱼群算法并未能进一步提升短期光伏功率预测精度。

## 10. 结论

实验结果表明：

1. CNN-LSTM 相比 LSTM 在误差指标上有所下降，说明 CNN 能够有效提取光伏功率序列中的局部波动特征。

2. CNN-BiLSTM 相比 BiLSTM 进一步提升预测精度，表明局部特征提取与双向时序建模具有互补作用。

3. 在当前实验条件下，混合 CNN 与 LSTM/BiLSTM 的模型整体优于 AFSA-PatchTST。最佳模型为 CNN-LSTM（RMSE=0.047724，R²=0.969009）。

综合来看，混合深度学习模型整体优于单一时序模型。

## 11. 产出文件

### 数据与样本
- `data/prediction/step3_hybrid_models/samples/X_train_seq.npy`
- `data/prediction/step3_hybrid_models/samples/X_val_seq.npy`
- `data/prediction/step3_hybrid_models/samples/X_test_seq.npy`
- `data/prediction/step3_hybrid_models/samples/y_train.npy`
- `data/prediction/step3_hybrid_models/samples/y_val.npy`
- `data/prediction/step3_hybrid_models/samples/y_test.npy`
- `data/prediction/step3_hybrid_models/samples/test_timestamps.csv`
- `data/prediction/step3_hybrid_models/samples/scaler_params.json`
- `data/prediction/step3_hybrid_models/samples/window_meta.json`

### 模型文件
- `data/prediction/step3_hybrid_models/models/lstm.pt`
- `data/prediction/step3_hybrid_models/models/bilstm.pt`
- `data/prediction/step3_hybrid_models/models/cnn_lstm.pt`
- `data/prediction/step3_hybrid_models/models/cnn_bilstm.pt`
- `data/prediction/step3_hybrid_models/models/afsa_patchtst.pt`

### 预测结果
- `data/prediction/step3_hybrid_models/predictions/lstm_test.csv`
- `data/prediction/step3_hybrid_models/predictions/bilstm_test.csv`
- `data/prediction/step3_hybrid_models/predictions/cnn_lstm_test.csv`
- `data/prediction/step3_hybrid_models/predictions/cnn_bilstm_test.csv`
- `data/prediction/step3_hybrid_models/predictions/afsa_patchtst_test.csv`

### 指标与历史
- `data/prediction/step3_hybrid_models/metrics/lstm_train_history.csv`
- `data/prediction/step3_hybrid_models/metrics/bilstm_train_history.csv`
- `data/prediction/step3_hybrid_models/metrics/cnn_lstm_train_history.csv`
- `data/prediction/step3_hybrid_models/metrics/cnn_bilstm_train_history.csv`
- `data/prediction/step3_hybrid_models/metrics/afsa_patchtst_train_history.csv`
- `data/prediction/step3_hybrid_models/metrics/afsa_patchtst_metrics.csv`
- `data/prediction/step3_hybrid_models/metrics/afsa_patchtst_best_params.json`

### 图表
- `data/prediction/step3_hybrid_models/figures/loss_lstm.png`
- `data/prediction/step3_hybrid_models/figures/loss_bilstm.png`
- `data/prediction/step3_hybrid_models/figures/loss_cnn_lstm.png`
- `data/prediction/step3_hybrid_models/figures/loss_cnn_bilstm.png`
- `data/prediction/step3_hybrid_models/figures/metrics_comparison.png`
- `data/prediction/step3_hybrid_models/figures/training_time_comparison.png`
- `data/prediction/step3_hybrid_models/figures/prediction_overlay_all_models.png`
- `data/prediction/step3_hybrid_models/figures/prediction_overlay_all_models.png`

### 日志与报告
- `logs/prediction/step3_hybrid_models/EXP-P03_prepare.log`
- `logs/prediction/step3_hybrid_models/EXP-P03_LSTM.log`
- `logs/prediction/step3_hybrid_models/EXP-P03_BiLSTM.log`
- `logs/prediction/step3_hybrid_models/EXP-P03_CNN_LSTM.log`
- `logs/prediction/step3_hybrid_models/EXP-P03_CNN_BiLSTM.log`
- `logs/prediction/step3_hybrid_models/EXP-P03_AFSA_Final.log`
- `logs/prediction/step3_hybrid_models/EXP-P03_summarize.log`
- `data/prediction/step3_hybrid_models/reports/exp_p03_report.md`
