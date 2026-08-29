# EXP-P04 Optuna 调参详细实验汇报

**Horizon: 15min (1 步)**

**生成时间: 2026-08-30 00:55:37**


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


| Model      | Best Params                                                                                               | Val Loss |
| ---------- | --------------------------------------------------------------------------------------------------------- | -------- |
| CNN-BiLSTM | conv_channels=16, kernel_size=5, bilstm_hidden=128, bilstm_layers=2, dropout=0.2, batch_size=64, lr=0.001 | 0.693188 |


## 4. 多 Seed 复现结果（Mean ± Std）


*无复现结果*


## 5. 关键发现


*等待实验完成*


## 6. 可视化


### 6.1 跨 Horizon 预测曲线（H1/H4/H16）


![跨Horizon预测曲线](data\prediction\step4_optuna_hybrid\figures\predictions_h1_h4_h16_combined.png)


*真实值：黑色实线；预测值：彩色虚线（H1 蓝 / H4 橙 / H16 绿）。三行子图共用同一时间窗口，x 轴标注起止时间与采样点数。*


### 6.2 指标对比


![predictions_overlay](data\prediction\step4_optuna_hybrid\figures\h1\predictions_overlay.png)


---


*报告由 run_exp_p04_report.py 自动生成*
