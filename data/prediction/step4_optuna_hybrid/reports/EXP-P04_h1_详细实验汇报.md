# EXP-P04 Optuna 调参详细实验汇报

**Horizon: 15min (1 步)**

**生成时间: 2026-06-22 11:30:09**


---


## 1. 实验配置


| 配置项 | 值 |
|---|---|

| Lookback | 16 |

| Horizon | 1 |

| 特征数量 | 13 |

| 训练样本数 | 49,111 |

| 验证样本数 | 10,524 |

| 测试样本数 | 10,524 |

| Rolling Folds | 3 |

| Optuna Trials/模型 | 8 |

| 最终 Max Epochs | 50 |

| 最终 Patience | 8 |

| Reproduce Seeds | [42, 43, 44] |

| 特征列 | `['power_ramp_15m_pu', 'total_irradiance_wm2', 'direct_normal_irradiance_wm2', 'global_horizontal_irradiance_wm2', 'air_temperature_c', 'atmosphere_hpa', 'relative_humidity_pct', 'daylight_flag', 'sin_hour', 'cos_hour', 'sin_dayofyear', 'cos_dayofyear', 'data_quality_score']` |



## 2. 特征工程


使用特征列表：


**气象/辐照**：`total_irradiance_wm2`, `direct_normal_irradiance_wm2`, `global_horizontal_irradiance_wm2`, `air_temperature_c`, `atmosphere_hpa`, `relative_humidity_pct`, `daylight_flag`


**时间周期**：`sin_hour`, `cos_hour`, `sin_dayofyear`, `cos_dayofyear`


**多尺度 ramp**：`power_ramp_15m_pu`


**其他**：`data_quality_score`


## 3. Optuna 最优参数


| Model        | Best Params                                                                                                | Val Loss |
| ------------ | ---------------------------------------------------------------------------------------------------------- | -------- |
| LSTM         | hidden=64, layers=2, dropout=0.2, lr=0.001, batch_size=64                                                  | 0.123710 |
| BiLSTM       | hidden=64, layers=2, dropout=0.2, lr=0.001, batch_size=64                                                  | 0.130875 |
| CNN-LSTM     | conv_channels=16, kernel_size=5, lstm_hidden=64, lstm_layers=1, dropout=0.1, lr=0.001, batch_size=32       | 0.029149 |
| CNN-BiLSTM   | conv_channels=16, kernel_size=3, bilstm_hidden=128, bilstm_layers=1, dropout=0.2, lr=0.0001, batch_size=64 | 0.147415 |
| MiniPatchTST | patch_len=8, stride=2, d_model=32, n_heads=2, num_layers=2, dropout=0.2, lr=0.001, batch_size=64           | 0.167077 |


## 4. 多 Seed 复现结果（Mean ± Std）


| Model        | Rank | MAE             | RMSE            | MAPE(%)      | R²              | Time(s) |
| ------------ | ---- | --------------- | --------------- | ------------ | --------------- | ------- |
| CNN-LSTM     | 1    | 0.0190 ± 0.0009 | 0.0413 ± 0.0022 | 21.77 ± 1.83 | 0.9773 ± 0.0025 | 138.0   |
| CNN-BiLSTM   | 2    | 0.0200 ± 0.0011 | 0.0424 ± 0.0004 | 20.69 ± 1.22 | 0.9761 ± 0.0004 | 294.7   |
| LSTM         | 3    | 0.0189 ± 0.0011 | 0.0438 ± 0.0012 | 19.67 ± 0.81 | 0.9745 ± 0.0014 | 151.9   |
| BiLSTM       | 4    | 0.0188 ± 0.0000 | 0.0440 ± 0.0022 | 19.26 ± 0.50 | 0.9743 ± 0.0025 | 241.2   |
| MiniPatchTST | 5    | 0.0251 ± 0.0027 | 0.0453 ± 0.0035 | 22.49 ± 0.19 | 0.9726 ± 0.0043 | 93.6    |



## 5. 与旧 AFSA-PatchTST 对比


| 指标 | 旧 AFSA-PatchTST |
|---|---|

| MAE | 0.0315 |

| RMSE | 0.0561 |

| MAPE(%) | 24.95 |

| R² | 0.9572 |


*注：旧 AFSA-PatchTST 为 horizon=1 的结果。*


## 6. 关键发现


- **最优模型**: CNN-LSTM (RMSE=0.0413, MAE=0.0190 ± 0.0009, R²=0.9773)

- **最差模型**: MiniPatchTST (RMSE=0.0453)

- **相对旧 AFSA 提升**: RMSE 降低 26.5%



## 7. 可视化


### 7.1 指标对比


![metrics_mae_bar](data\prediction\step4_optuna_hybrid\figures\h1\metrics_mae_bar.png)


![metrics_rmse_bar](data\prediction\step4_optuna_hybrid\figures\h1\metrics_rmse_bar.png)


![metrics_r2_bar](data\prediction\step4_optuna_hybrid\figures\h1\metrics_r2_bar.png)


![training_time](data\prediction\step4_optuna_hybrid\figures\h1\training_time.png)


![predictions_overlay](data\prediction\step4_optuna_hybrid\figures\h1\predictions_overlay.png)


---


*报告由 run_exp_p04_report.py 自动生成*
