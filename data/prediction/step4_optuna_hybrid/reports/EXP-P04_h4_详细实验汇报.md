# EXP-P04 Optuna 调参详细实验汇报

**Horizon: 1h (4 步)**

**生成时间: 2026-06-22 17:38:43**


---


## 1. 实验配置


| 配置项 | 值 |
|---|---|

| Lookback | 16 |

| Horizon | 4 |

| 特征数量 | 13 |

| 训练样本数 | 49,108 |

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


| Model        | Best Params                                                                                               | Val Loss |
| ------------ | --------------------------------------------------------------------------------------------------------- | -------- |
| LSTM         | hidden=64, layers=2, dropout=0.2, lr=0.001, batch_size=64                                                 | 0.157481 |
| BiLSTM       | hidden=64, layers=2, dropout=0.2, lr=0.001, batch_size=64                                                 | 0.150190 |
| CNN-LSTM     | conv_channels=16, kernel_size=3, lstm_hidden=128, lstm_layers=1, dropout=0.2, lr=0.0001, batch_size=64    | 0.174068 |
| CNN-BiLSTM   | conv_channels=16, kernel_size=5, bilstm_hidden=128, bilstm_layers=2, dropout=0.2, lr=0.001, batch_size=64 | 0.114755 |
| MiniPatchTST | patch_len=4, stride=4, d_model=32, n_heads=2, num_layers=2, dropout=0.1, lr=0.001, batch_size=64          | 0.158660 |


## 4. 多 Seed 复现结果（Mean ± Std）


| Model        | Rank | MAE             | RMSE            | MAPE(%)      | R²              | Time(s) |
| ------------ | ---- | --------------- | --------------- | ------------ | --------------- | ------- |
| CNN-LSTM     | 1    | 0.0254 ± 0.0004 | 0.0561 ± 0.0011 | 28.95 ± 0.11 | 0.9582 ± 0.0016 | 150.9   |
| CNN-BiLSTM   | 2    | 0.0263 ± 0.0002 | 0.0571 ± 0.0019 | 26.87 ± 0.93 | 0.9566 ± 0.0029 | 339.3   |
| BiLSTM       | 3    | 0.0249 ± 0.0016 | 0.0574 ± 0.0025 | 28.01 ± 1.61 | 0.9561 ± 0.0037 | 188.9   |
| LSTM         | 4    | 0.0260 ± 0.0021 | 0.0577 ± 0.0012 | 27.65 ± 2.79 | 0.9557 ± 0.0018 | 103.3   |
| MiniPatchTST | 5    | 0.0275 ± 0.0014 | 0.0590 ± 0.0014 | 28.98 ± 1.86 | 0.9537 ± 0.0022 | 84.4    |



## 5. 与旧 AFSA-PatchTST 对比


| 指标 | 旧 AFSA-PatchTST |
|---|---|

| MAE | 0.0315 |

| RMSE | 0.0561 |

| MAPE(%) | 24.95 |

| R² | 0.9572 |


*注：旧 AFSA-PatchTST 为 horizon=1 的结果。*


## 6. 关键发现


- **最优模型**: CNN-LSTM (RMSE=0.0561, MAE=0.0254 ± 0.0004, R²=0.9582)

- **最差模型**: MiniPatchTST (RMSE=0.0590)

- **相对旧 AFSA 提升**: RMSE 降低 0.1%



## 7. 可视化


### 7.1 指标对比


![h4_metrics_comparison](data\prediction\step4_optuna_hybrid\figures\h4\h4_metrics_comparison.png)


![training_time](data\prediction\step4_optuna_hybrid\figures\h4\training_time.png)


![predictions_overlay](data\prediction\step4_optuna_hybrid\figures\h4\predictions_overlay.png)


---


*报告由 run_exp_p04_report.py 自动生成*
