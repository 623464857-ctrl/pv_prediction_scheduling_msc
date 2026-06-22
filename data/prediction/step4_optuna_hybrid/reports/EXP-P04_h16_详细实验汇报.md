# EXP-P04 Optuna 调参详细实验汇报

**Horizon: 4h (16 步)**

**生成时间: 2026-06-22 17:38:41**


---


## 1. 实验配置


| 配置项 | 值 |
|---|---|

| Lookback | 16 |

| Horizon | 16 |

| 特征数量 | 13 |

| 训练样本数 | 49,100 |

| 验证样本数 | 10,522 |

| 测试样本数 | 10,522 |

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
| LSTM         | hidden=64, layers=2, dropout=0.2, lr=0.001, batch_size=64                                                  | 0.186893 |
| BiLSTM       | hidden=64, layers=2, dropout=0.2, lr=0.001, batch_size=64                                                  | 0.179968 |
| CNN-LSTM     | conv_channels=16, kernel_size=5, lstm_hidden=128, lstm_layers=2, dropout=0.2, lr=0.001, batch_size=64      | 0.214649 |
| CNN-BiLSTM   | conv_channels=16, kernel_size=3, bilstm_hidden=128, bilstm_layers=1, dropout=0.2, lr=0.0001, batch_size=64 | 0.229339 |
| MiniPatchTST | patch_len=4, stride=2, d_model=32, n_heads=4, num_layers=2, dropout=0.2, lr=0.0001, batch_size=32          | 0.249876 |


## 4. 多 Seed 复现结果（Mean ± Std）


| Model        | Rank | MAE             | RMSE            | MAPE(%)      | R²              | Time(s) |
| ------------ | ---- | --------------- | --------------- | ------------ | --------------- | ------- |
| CNN-BiLSTM   | 1    | 0.0387 ± 0.0002 | 0.0821 ± 0.0005 | 47.49 ± 1.63 | 0.9103 ± 0.0010 | 230.2   |
| LSTM         | 2    | 0.0384 ± 0.0004 | 0.0833 ± 0.0029 | 41.69 ± 1.70 | 0.9076 ± 0.0064 | 67.5    |
| CNN-LSTM     | 3    | 0.0391 ± 0.0017 | 0.0843 ± 0.0016 | 46.17 ± 1.77 | 0.9055 ± 0.0035 | 168.1   |
| MiniPatchTST | 4    | 0.0393 ± 0.0006 | 0.0843 ± 0.0008 | 46.82 ± 3.53 | 0.9054 ± 0.0017 | 223.6   |
| BiLSTM       | 5    | 0.0422 ± 0.0052 | 0.0887 ± 0.0080 | 43.00 ± 1.19 | 0.8949 ± 0.0192 | 132.2   |



## 5. 与旧 AFSA-PatchTST 对比


| 指标 | 旧 AFSA-PatchTST |
|---|---|

| MAE | 0.0315 |

| RMSE | 0.0561 |

| MAPE(%) | 24.95 |

| R² | 0.9572 |


*注：旧 AFSA-PatchTST 为 horizon=1 的结果。*


## 6. 关键发现


- **最优模型**: CNN-BiLSTM (RMSE=0.0821, MAE=0.0387 ± 0.0002, R²=0.9103)

- **最差模型**: BiLSTM (RMSE=0.0887)

- **相对旧 AFSA**: RMSE 差 46.4%



## 7. 可视化


### 7.1 指标对比


![h16_metrics_comparison](data\prediction\step4_optuna_hybrid\figures\h16\h16_metrics_comparison.png)


![training_time](data\prediction\step4_optuna_hybrid\figures\h16\training_time.png)


![predictions_overlay](data\prediction\step4_optuna_hybrid\figures\h16\predictions_overlay.png)


---


*报告由 run_exp_p04_report.py 自动生成*
