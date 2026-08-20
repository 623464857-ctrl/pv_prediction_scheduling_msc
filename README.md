# 光伏预测与调度协同研究工程

## 项目概述

本项目用于短期光伏发电功率预测研究，通过对比单一时序模型（LSTM、BiLSTM）与混合深度学习模型（CNN-LSTM、CNN-BiLSTM、PatchTST），验证混合模型在光伏功率预测任务中的优越性，并系统探索残差预测、强基线对比、多 Horizon 评估与数据审计。项目经历 7 个实验阶段（EXP-P01 ~ EXP-P10），覆盖数据清洗、基线建模、混合深度学习、Optuna 超参优化、残差预测建模、混合搜索与数据完整性审计。

## 当前进度

| 实验 | 名称 | 状态 |
|------|------|------|
| EXP-P01 | 数据清洗与时间对齐 | ✅ 已完成 |
| EXP-P02 | 五类基础模型对比 | ✅ 已完成 |
| EXP-P03 | 混合深度学习模型对比 | ✅ 已完成 |
| EXP-P04 | Optuna 超参优化与多 Horizon 预测 | ✅ 已完成 |
| EXP-P05 | 残差预测建模与混合超参搜索 | ✅ 已完成 |
| EXP-P10 | 数据划分一致性审计与 Oracle 根因实验 | ✅ 已完成 |

## 快速开始

### 环境要求

- Python 3.13+
- PyTorch
- NumPy, Pandas, Scikit-learn
- XGBoost, LightGBM
- Matplotlib, Optuna

### 运行实验

```powershell
# Step 1: 数据预处理
python experiments/prediction/step1_data_cleaning_alignment/run_exp_p01_preprocessing.py

# Step 2: 基础模型训练
python experiments/prediction/step2_baseline_models/run_exp_p02_train_lstm.py
python experiments/prediction/step2_baseline_models/run_exp_p02_train_bilstm.py
python experiments/prediction/step2_baseline_models/run_exp_p02_train_cnn_lstm.py
python experiments/prediction/step2_baseline_models/run_exp_p02_train_cnn_bilstm.py
python experiments/prediction/step2_baseline_models/run_exp_p02_train_bp.py
python experiments/prediction/step2_baseline_models/run_exp_p02_train_svr.py
python experiments/prediction/step2_baseline_models/run_exp_p02_train_rf.py
python experiments/prediction/step2_baseline_models/run_exp_p02_summarize_results.py

# Step 4: Optuna 超参优化 + 多 Horizon 预测
python -m experiments.prediction.step4_optuna_hybrid.run_exp_p04_prepare_samples --horizon 1
python -m experiments.prediction.step4_optuna_hybrid.run_exp_p04_optuna --horizon 1
python -m experiments.prediction.step4_optuna_hybrid.run_exp_p04_final_train --horizon 1
python -m experiments.prediction.step4_optuna_hybrid.run_exp_p04_reproduce --horizon 1
python -m experiments.prediction.step4_optuna_hybrid.run_exp_p04_report --horizon 1
# (--horizon 可选 1/4/16，分别对应 15min/1h/4h 预测)

# Step 5: 残差预测建模与混合搜索
python experiments/prediction/step5_new_experiments/run_exp_p05_main.py --horizon 1
# (--horizon 可选 1/4/16)

# Step 5: 生成报告图表（在 report 之后运行）
python experiments/prediction/step5_new_experiments/run_exp_p05_figures.py --horizon 1
# 生成跨 horizon 综合对比图（需在 h1/h4/h16 全部运行后执行）
python experiments/prediction/step5_new_experiments/run_exp_p05_figures.py --horizon 1 --cross

# Step 10: 数据审计报告
# 详细报告位于: experiments/prediction/step10_data_audit/
```

## 目录结构

```
pv_prediction_scheduling_msc/
├── README.md                                    # 项目总览与使用说明
├── docs/
│   └── PROJECT_STRUCTURE.md                    # 工程结构说明
│
├── data/                                        # 数据目录
│   ├── raw/                                    # 原始数据（只读）
│   │   ├── Solar station site 1 (Nominal capacity-50MW).csv
│   │   ├── Solar station site 2 (Nominal capacity-130MW).csv
│   │   ├── Solar station site 3 (Nominal capacity-130MW).csv
│   │   ├── Solar station site 4 (Nominal capacity-130MW).csv
│   │   ├── Solar station site 5 (Nominal capacity-110MW).csv
│   │   ├── Solar station site 6 (Nominal capacity-35MW).csv
│   │   ├── Solar station site 7 (Nominal capacity-30MW).csv
│   │   └── Solar station site 8 (Nominal capacity-30MW).csv
│   │
│   └── prediction/                              # 预测任务数据
│       ├── step1_preprocessing/                 # EXP-P01 数据预处理
│       │   ├── OUTPUT_FILE_COMMENTS.md          # 输出字段说明
│       │   └── processed/
│       │       ├── solar_dispatch_panel_common_window.csv  # 多站调度面板共同窗口
│       │       ├── solar_feature_scaling_reference.csv      # 特征缩放参考
│       │       ├── solar_site_quality_summary.csv           # 站点质量汇总
│       │       └── stations/                              # 各站点预处理数据
│       │
│       ├── step2_baseline_models/               # EXP-P02 基础模型
│       │   ├── config/
│       │   │   └── exp_p02_config.json          # 实验配置
│       │   ├── scalers/                         # 标准化器
│       │   ├── samples/                         # 训练样本
│       │   ├── models/                          # 训练好的模型
│       │   ├── predictions/                     # 预测结果
│       │   ├── metrics/                         # 训练指标与历史
│       │   └── reports/
│       │
│       ├── step3_hybrid_models/                 # EXP-P03 混合模型
│       │   ├── config/
│       │   │   └── exp_p03_config.json          # 实验配置
│       │   ├── samples/                         # 滑动窗口样本
│       │   ├── models/                          # LSTM/BiLSTM/CNN-LSTM/CNN-BiLSTM/AFSA-PatchTST
│       │   ├── predictions/                     # 测试集预测结果
│       │   ├── metrics/                         # 训练历史与指标
│       │   ├── figures/                         # 可视化图表
│       │   └── reports/
│       │
│       ├── step4_optuna_hybrid/                 # EXP-P04 Optuna 超参优化 + 多 Horizon
│       │   ├── config/
│       │   │   ├── exp_p04_base.json           # 基础配置（lookback/epochs/Optuna 参数等）
│       │   │   ├── exp_p04_h1.json             # Horizon=1 配置（15min）
│       │   │   ├── exp_p04_h4.json             # Horizon=4 配置（1h）
│       │   │   └── exp_p04_h16.json            # Horizon=16 配置（4h）
│       │   ├── samples/                         # 各 horizon 滑动窗口样本
│       │   │   ├── h1/                         # Horizon=1 样本
│       │   │   ├── h4/                         # Horizon=4 样本
│       │   │   └── h16/                        # Horizon=16 样本
│       │   ├── models/                          # 训练好的最优模型
│       │   ├── predictions/                     # 测试集预测结果
│       │   ├── metrics/                         # 训练指标与历史
│       │   │   └── h{1,4,16}/
│       │   │       ├── lstm_optuna.json        # Optuna 最优参数
│       │   │       ├── lstm_final_train_history.csv
│       │   │       ├── lstm_test_metrics.json
│       │   │       └── lstm_reproduce.json      # 多 seed 复现均值/标准差
│       │   ├── figures/                         # 可视化图表
│       │   │   ├── h1/
│       │   │   ├── h4/
│       │   │   ├── h16/
│       │   │   └── comparison_all_horizons.png   # 三 horizon 跨模型对比
│       │   └── reports/
│       │       ├── EXP-P04_h1_详细实验汇报.md
│       │       ├── EXP-P04_h4_详细实验汇报.md
│       │       ├── EXP-P04_h16_详细实验汇报.md
│       │       └── FINAL_综合实验报告.md         # EXP-P03 & EXP-P04 综合报告
│       │
│       └── step5_new_experiments/             # EXP-P05 残差预测建模与混合搜索
│           ├── config/
│           │   └── exp_p05_base.json           # XGB/LGBM 参数、混合搜索权重
│           ├── samples/                         # 增强特征样本
│           ├── models/                          # 残差深度学习模型
│           ├── predictions/                     # 测试集预测
│           ├── metrics/                         # 分段评价指标
│           ├── figures/                         # 综合可视化
│           │   ├── h1/
│           │   ├── h4/
│           │   ├── h16/
│           │   ├── comparison_summary.png
│           │   ├── comparison_best_model.png
│           │   └── inference_benchmark_all_horizons.png
│           └── reports/
│               └── FINAL_综合实验报告.md         # EXP-P05 综合报告
│
├── experiments/                                 # 实验脚本目录
│   ├── prediction/                              # 预测任务实验
│   │   ├── step1_data_cleaning_alignment/       # EXP-P01 数据预处理
│   │   │   ├── README.md
│   │   │   └── run_exp_p01_preprocessing.py
│   │   │
│   │   ├── step2_baseline_models/               # EXP-P02 基础模型
│   │   │   ├── README.md
│   │   │   ├── exp_p02_common.py
│   │   │   ├── exp_p02_torch_utils.py
│   │   │   ├── run_exp_p02_prepare_samples.py
│   │   │   ├── run_exp_p02_train_*.py           # LSTM/BiLSTM/CNN-LSTM/CNN-BiLSTM/BP/SVR/RF
│   │   │   └── run_exp_p02_summarize_results.py
│   │   │
│   │   ├── step3_hybrid_models/                 # EXP-P03 混合模型
│   │   │   ├── exp_p03_common.py
│   │   │   ├── exp_p03_models.py               # LSTM/BiLSTM/CNN/PatchTST
│   │   │   ├── exp_p03_torch_utils.py
│   │   │   ├── exp_p03_afsa.py                 # AFSA 人工鱼群算法
│   │   │   ├── run_exp_p03_*.py                 # 各模型训练入口
│   │   │   └── run_exp_p03_summarize.py
│   │   │
│   │   ├── step4_optuna_hybrid/                 # EXP-P04 Optuna + 多 Horizon
│   │   │   ├── exp_p04_common.py
│   │   │   ├── exp_p04_models.py               # 含 MiniPatchTST
│   │   │   ├── exp_p04_torch_utils.py
│   │   │   ├── exp_p04_cv.py                   # Rolling Window CV
│   │   │   ├── exp_p04_features.py             # lag/ramp/rolling 特征
│   │   │   ├── run_exp_p04_prepare_samples.py
│   │   │   ├── run_exp_p04_optuna.py
│   │   │   ├── run_exp_p04_final_train.py
│   │   │   ├── run_exp_p04_reproduce.py        # 多 seed 复现
│   │   │   ├── run_exp_p04_validation.py
│   │   │   ├── run_exp_p04_report.py
│   │   │   └── run_exp_p04_analysis.py
│   │   │
│   │   ├── step5_new_experiments/               # EXP-P05 残差预测与混合搜索
│   │   │   ├── exp_p05_common.py
│   │   │   ├── baselines.py                    # Persistence/Ridge/XGB/LGBM
│   │   │   ├── exp_p05_models.py              # 残差版 LSTM/BiLSTM/CNN-LSTM/CNN-BiLSTM/PatchTST
│   │   │   ├── exp_p05_features.py            # lag/rolling/ramp/daylight
│   │   │   ├── exp_p05_residual.py            # 残差建模工具
│   │   │   ├── run_exp_p05_main.py            # 主入口
│   │   │   ├── run_exp_p05_baselines.py
│   │   │   ├── run_exp_p05_residual_train.py
│   │   │   ├── run_exp_p05_evaluation.py     # 分段评价 + 推理计时
│   │   │   ├── run_exp_p05_hybrid_search.py    # S1-S6 混合搜索消融
│   │   │   └── run_exp_p05_report.py
│   │   │
│   │   └── step10_data_audit/                  # EXP-P10 数据审计
│   │       ├── STEP10_MAIN_LOG.md              # 主日志（7 阶段）
│   │       ├── PHASE1_DATA_AUDIT_LOG.md         # 数据划分审计
│   │       ├── PHASE2_SCALER_AUDIT_LOG.md       # Scaler 链路审计
│   │       ├── PHASE3_CLIPPING_AUDIT_LOG.md     # 物理后处理裁剪审计
│   │       ├── PHASE4_PEAK_DISTRIBUTION_AUDIT_LOG.md  # 峰值分布审计
│   │       ├── PHASE5_H16_DECAY_DIAGNOSIS_LOG.md      # H16 信息衰减诊断
│   │       ├── PHASE6_WEATHER_TRANSITION_LOG.md       # 天气切换诊断
│   │       ├── PHASE7_ORACLE_EXPERIMENT_LOG.md        # Oracle 根因实验
│   │       ├── SITE1_GEOLOCATION_REPORT.md      # 站点地理定位报告
│   │       └── figures/
│   │           ├── p4/
│   │           ├── p5/
│   │           ├── p6/
│   │           ├── p7/
│   │           └── geolocation/
│   │
│   └── scheduling/                              # 预留调度任务目录
│       └── README.md
│
└── logs/                                        # 运行日志目录
    ├── prediction/
    │   ├── step1_data_cleaning_alignment/
    │   ├── step2_baseline_models/
    │   ├── step3_hybrid_models/
    │   ├── step4_optuna_hybrid/
    │   │   └── EXP-P04_h{1,4,16}_{prepare,optuna|final|reproduce|report}.log
    │   └── step5_new_experiments/
    └── wrf_feature_engineering.log
```

## 核心实验结果

### EXP-P04: Optuna 超参优化与多 Horizon 预测

**实验设计：**
- 三个预测步长：h1=15min, h4=1h, h16=4h
- 特征工程：辐照/气象 + 时间周期编码 + 15min ramp 特征
- Optuna 8 trials/模型/horizon，Rolling 3-fold 时序交叉验证
- 最终训练 50 epochs（patience=8），多 seed（42/43/44）复现

**关键结果（多 Seed 复现 Mean ± Std）：**

| Horizon | 最优模型 | RMSE | MAE | R² |
|---------|---------|------|-----|-----|
| **h1 (15min)** | **CNN-LSTM** | **0.0413 ± 0.0022** | 0.0190 ± 0.0009 | 0.9773 |
| **h4 (1h)** | **CNN-LSTM** | **0.0561 ± 0.0011** | 0.0254 ± 0.0004 | 0.9582 |
| **h16 (4h)** | **CNN-BiLSTM** | **0.0821 ± 0.0005** | 0.0387 ± 0.0002 | 0.9103 |

> 排名标准：RMSE（主要）> MAE（次要）> R²，越小越优

### EXP-P05: 残差预测建模与混合超参搜索

**实验设计：**
- 核心创新：**残差预测范式** `Δy = y(t+H) - y(t)`，最终输出 `y_hat = y_last + Δy_hat`
- 强基线体系：Persistence、Moving Average、Ridge、XGBoost、LightGBM
- 深度学习残差模型：LSTM、BiLSTM、CNN-LSTM、CNN-BiLSTM、PatchTST
- 白天/夜间分段评价（主表采用 Daytime-only RMSE）
- 标准化推理效率测量（ms/sample、参数量）
- Optuna-AFSA 混合搜索（S1-S6 消融实验）

**关键发现：XGBoost / LightGBM 在所有 Horizon 上全面领先深度学习模型**

| Horizon | 最优模型 | RMSE | MAE | R² |
|---------|---------|------|-----|-----|
| **h1 (15min)** | **XGBoost / LightGBM** | **0.0300** | 0.0108 | 0.9880 |
| **h4 (1h)** | **XGBoost** | **0.0474** | 0.0190 | 0.9701 |
| **h16 (4h)** | **XGBoost** | **0.0843** | 0.0382 | 0.9056 |

**残差深度学习模型内部排序（Daytime-only RMSE）：**

| Horizon | 最佳残差模型 | RMSE |
|---------|------------|------|
| h1 (15min) | LSTM (Residual) | 0.0458 |
| h4 (1h) | CNN-BiLSTM (Residual) | 0.0715 |
| h16 (4h) | PatchTST (Residual) | 0.1158 |

**跨 Horizon 误差增长（PatchTST 最缓慢）：**

| 模型 | H1 RMSE | H4 RMSE | H16 RMSE | H1→H16 增幅 |
|------|---------|---------|---------|-------------|
| PatchTST (Residual) | 0.0473 | 0.0729 | 0.1158 | **2.45×** |
| CNN-LSTM (Residual) | 0.0468 | 0.0735 | 0.1166 | 2.49× |
| LSTM (Residual) | 0.0458 | 0.0723 | 0.1159 | 2.53× |
| XGBoost | 0.0300 | 0.0474 | 0.0843 | 2.81× |

### EXP-P10: 数据划分一致性审计与 Oracle 根因实验

**7 阶段审计结果摘要：**

| 阶段 | 名称 | 核心发现 |
|------|------|----------|
| Phase 1 | 数据划分审计 | ⚠️ Step3 使用 56/14/30 比例，其他 Step 使用 70/15/15，测试集完全不重叠 |
| Phase 2 | Scaler 链路审计 | ✅ 全部通过：Scaler 仅 fit train，Horizon 独立，Inverse transform 正确 |
| Phase 3 | 物理裁剪审计 | ⚠️ 代码库无裁剪逻辑；深度学习模型有 4.5-55.8% 样本预测为负 |
| Phase 4 | 峰值分布审计 | ✅ 测试集峰值完全被训练集覆盖，峰值误差非数据覆盖问题 |
| Phase 5 | H16 信息衰减诊断 | ⚠️ **H16 自相关为负值 (-0.19)**，辐照特征衰减 28-35% |
| Phase 6 | 天气切换诊断 | ⚠️ 下午 (12-18h) RMSE 最高 (0.1346)，高功率波动误差是低功率的 24 倍 |
| Phase 7 | Oracle 根因实验 | ✅ **完全确认**：完美天气预报 RMSE 改进 97.4%，H16 瓶颈是**未来气象信息缺失** |

**Oracle 实验关键数据：**

| 模型 | RMSE | R² | 改进 |
|------|------|-----|------|
| Current-only | 0.0757 | 0.9068 | - |
| **Oracle (完美天气预报)** | **0.0020** | **0.9999** | **97.4%** |

> **核心结论**：H16（4h）预测精度下降的根本原因是**未来气象信息缺失**，而非模型容量不足。引入 GFS/ECMWF 数值天气预报数据可预期 RMSE 降低 40-50%。

**优先改进建议：**

| 优先级 | 方向 | 预期收益 |
|--------|------|----------|
| 🔴 P0 | 引入 GFS/ECMWF 数值天气预报 | RMSE 降低 40-50% |
| 🔴 P0 | 集成辐照预报模型 | RMSE 降低 30-40% |

## 核心代码模块

### exp_p04_common.py
共享工具模块，包含：路径管理（CONFIG_DIR, SAMPLES_DIR, MODELS_DIR, METRICS_DIR, FIGURES_DIR 等）、配置加载（base/h1/h4/h16 三套配置）、日志设置、指标计算（MAE, RMSE, MAPE, R²）、可视化函数。

### exp_p04_models.py
模型定义模块（EXP-P04），包含：`LSTMRegressor`、`BiLSTMRegressor`、`CNNLSTMRegressor`、`CNNBiLSTMRegressor`、`MiniPatchTSTRegressor`、`build_model()` 工厂函数。

### exp_p04_cv.py
时序交叉验证模块，`RollingWindowCV`：Rolling Window 时序交叉验证，避免时间序列数据泄露。

### exp_p04_features.py
特征工程模块：`add_lag_features()` 功率 lag 特征、`add_ramp_features()` 多尺度 ramp 特征（15min/1h/4h）、`add_rolling_features()` 滚动均值/标准差特征。

### exp_p05_residual.py
残差预测建模工具：`build_residual_dataset()` 构建残差样本、`train_residual_model()` 残差模型训练、`reconstruct_power()` 功率重构。

### exp_p05_baselines.py
强基线模型实现：Persistence（当前值作为预测）、Moving Average（历史功率滑动平均）、Ridge Regression（L2 正则化线性回归）、XGBoost（200 棵，最大深度 6）、LightGBM（200 棵，最大深度 6）。

### step10_data_audit/
数据审计工具：`wrf_feature_engineering.py`（WRF 气象数据特征工程）、各 Phase 独立审计脚本与日志。

## 数据流

```
raw CSV → EXP-P01 → preprocessed CSV → EXP-P02/P03 → samples → models → predictions → metrics/figures/reports
                                                               ↘ EXP-P04 (多 horizon Optuna)
                                                                   → h1/h4/h16 样本 → Optuna 调参 → 多 seed 复现 → 综合报告
                                                               ↘ EXP-P05 (残差预测 + 强基线)
                                                                   → 增强特征样本 → 树模型 + 残差 DL → 分段评价 → 综合报告
                                                               ↘ EXP-P10 (数据审计)
                                                                   → 7 阶段审计 → Oracle 根因实验 → P0 改进建议
```

## 命名约定

- 实验编号：EXP-P01, EXP-P02, EXP-P03, EXP-P04, EXP-P05, EXP-P10
- 脚本命名：`run_exp_<编号>_<功能>.py`
- 模块命名：`exp_<编号>_<模块名>.py`
- Horizon 标识：`h1` (15min) / `h4` (1h) / `h16` (4h)

## 重要说明

1. **数据划分**：所有实验严格按时间顺序划分，禁止随机打乱
2. **标准化**：仅在 train 集拟合 scaler，val/test 使用 transform
3. **测试集保护**：测试集不参与训练、验证或超参数搜索
4. **可复现性**：保存 scaler 参数和随机种子
5. **Step3 vs Step4/5 不可直接对比**：Step3 使用 56/14/30 划分，Step4/5 使用 70/15/15 划分

## 产出文件索引

### EXP-P01 产出
- `data/prediction/step1_preprocessing/processed/stations/Site_*.csv`
- `data/prediction/step1_preprocessing/processed/solar_*.csv`

### EXP-P02 产出
- `data/prediction/step2_baseline_models/samples/*.npy`
- `data/prediction/step2_baseline_models/models/*.pt/.joblib`
- `data/prediction/step2_baseline_models/predictions/*.csv`
- `data/prediction/step2_baseline_models/metrics/*.csv`
- `data/prediction/step2_baseline_models/reports/EXP-P02_preliminary_conclusion.md`

### EXP-P03 产出
- `data/prediction/step3_hybrid_models/samples/*.npy`
- `data/prediction/step3_hybrid_models/models/*.pt`
- `data/prediction/step3_hybrid_models/predictions/*.csv`
- `data/prediction/step3_hybrid_models/metrics/*.csv`
- `data/prediction/step3_hybrid_models/figures/*.png`
- `data/prediction/step3_hybrid_models/reports/*.md`

### EXP-P04 产出
- `data/prediction/step4_optuna_hybrid/samples/h{1,4,16}/*.npy`
- `data/prediction/step4_optuna_hybrid/models/h{1,4,16}/*.pt`
- `data/prediction/step4_optuna_hybrid/predictions/h{1,4,16}/*.csv`
- `data/prediction/step4_optuna_hybrid/metrics/h{1,4,16}/*.json`
- `data/prediction/step4_optuna_hybrid/figures/h{1,4,16}/*.png`
- `data/prediction/step4_optuna_hybrid/figures/comparison_*.png`
- `data/prediction/step4_optuna_hybrid/reports/FINAL_综合实验报告.md`

### EXP-P05 产出
- `data/prediction/step5_new_experiments/samples/*.npy`
- `data/prediction/step5_new_experiments/models/**/*.pt`
- `data/prediction/step5_new_experiments/predictions/*.csv`
- `data/prediction/step5_new_experiments/metrics/*.json`
- `data/prediction/step5_new_experiments/figures/**/*.png`
- `data/prediction/step5_new_experiments/reports/FINAL_综合实验报告.md`

### EXP-P10 产出
- `experiments/prediction/step10_data_audit/STEP10_MAIN_LOG.md`
- `experiments/prediction/step10_data_audit/PHASE{1-7}_*_LOG.md`
- `experiments/prediction/step10_data_audit/figures/**/*.png`

### 日志文件
- `logs/prediction/step1_data_cleaning_alignment/EXP-P01.log`
- `logs/prediction/step2_baseline_models/EXP-P02_*.log`
- `logs/prediction/step3_hybrid_models/EXP-P03_*.log`
- `logs/prediction/step4_optuna_hybrid/EXP-P04_h{1,4,16}_{prepare,optuna|final|reproduce|report}.log`
- `logs/prediction/step5_new_experiments/EXP-P05_h{1,4,16}_*.log`
