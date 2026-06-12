# 光伏预测与调度协同研究工程

## 项目概述

本项目用于短期光伏发电功率预测研究，通过对比单一时序模型（LSTM、BiLSTM）与混合深度学习模型（CNN-LSTM、CNN-BiLSTM、AFSA-PatchTST），验证混合模型在光伏功率预测任务中的优越性。

## 当前进度

| 实验 | 名称 | 脚本 | 状态 |
|------|------|------|------|
| EXP-P01 | 数据清洗与时间对齐 | `experiments/prediction/step1_data_cleaning_alignment/run_exp_p01_preprocessing.py` | ✅ 已完成 |
| EXP-P02 | 五类基础模型对比 | `experiments/prediction/step2_baseline_models/` | ✅ 已完成 |
| EXP-P03 | 混合深度学习模型对比 | `experiments/prediction/step3_hybrid_models/` | ✅ 已完成 |

## 快速开始

### 环境要求

- Python 3.13+
- PyTorch
- NumPy, Pandas, Scikit-learn
- Matplotlib

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

# Step 3: 混合模型训练
python experiments/prediction/step3_hybrid_models/run_exp_p03_prepare_samples.py
python experiments/prediction/step3_hybrid_models/run_exp_p03_train_lstm.py
python experiments/prediction/step3_hybrid_models/run_exp_p03_train_bilstm.py
python experiments/prediction/step3_hybrid_models/run_exp_p03_train_cnn_lstm.py
python experiments/prediction/step3_hybrid_models/run_exp_p03_train_cnn_bilstm.py
python experiments/prediction/step3_hybrid_models/run_exp_p03_afsa_final.py
python experiments/prediction/step3_hybrid_models/run_exp_p03_summarize.py
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
│       │           ├── Site_1_preprocessed.csv
│       │           ├── Site_2_preprocessed.csv
│       │           ├── Site_3_preprocessed.csv
│       │           ├── Site_4_preprocessed.csv
│       │           ├── Site_5_preprocessed.csv
│       │           ├── Site_6_preprocessed.csv
│       │           ├── Site_7_preprocessed.csv
│       │           └── Site_8_preprocessed.csv
│       │
│       ├── step2_baseline_models/               # EXP-P02 基础模型
│       │   ├── config/
│       │   │   └── exp_p02_config.json          # 实验配置
│       │   ├── scalers/                         # 标准化器
│       │   │   ├── feature_scaler.joblib        # 序列特征标准化器
│       │   │   ├── scaler_flat.joblib           # 扁平特征标准化器
│       │   │   └── scaler_seq.joblib            # 序列标准化器
│       │   ├── samples/                         # 训练样本
│       │   │   ├── X_train_seq.npy              # 训练输入序列
│       │   │   ├── X_val_seq.npy                # 验证输入序列
│       │   │   ├── X_test_seq.npy               # 测试输入序列
│       │   │   ├── y_train.npy                  # 训练目标
│       │   │   ├── y_val.npy                    # 验证目标
│       │   │   ├── y_test.npy                   # 测试目标
│       │   │   ├── test_timestamps.csv          # 测试集时间戳
│       │   │   ├── scaler_params.json           # 标准化器参数
│       │   │   └── site1_window_meta.json       # 窗口元数据
│       │   ├── models/                          # 训练好的模型
│       │   │   ├── lstm.pt                     # LSTM 模型
│       │   │   ├── bilstm.pt                   # BiLSTM 模型
│       │   │   ├── cnn_lstm.pt                 # CNN-LSTM 模型
│       │   │   ├── cnn_bilstm.pt               # CNN-BiLSTM 模型
│       │   │   ├── bp.joblib                   # BP 神经网络模型
│       │   │   ├── svr.joblib                  # SVR 模型
│       │   │   └── rf.joblib                   # 随机森林模型
│       │   ├── predictions/                     # 预测结果
│       │   │   ├── lstm_test.csv
│       │   │   ├── bilstm_test.csv
│       │   │   ├── cnn_lstm_test.csv
│       │   │   ├── cnn_bilstm_test.csv
│       │   │   ├── bp_test.csv
│       │   │   ├── svr_test.csv
│       │   │   └── randomforest_test.csv
│       │   ├── metrics/                         # 训练指标与历史
│       │   │   ├── lstm_train_history.csv
│       │   │   ├── bilstm_train_history.csv
│       │   │   ├── cnn_lstm_train_history.csv
│       │   │   ├── cnn_bilstm_train_history.csv
│       │   │   ├── bp_train_history.csv
│       │   │   └── baseline_comparison_metrics.csv
│       │   └── reports/
│       │       └── EXP-P02_preliminary_conclusion.md  # 初步结论报告
│       │
│       └── step3_hybrid_models/                 # EXP-P03 混合模型
│           ├── config/
│           │   └── exp_p03_config.json          # 实验配置
│           ├── samples/                         # 滑动窗口样本
│           │   ├── X_train_seq.npy              # 训练输入 [samples, 16, 13]
│           │   ├── X_val_seq.npy                # 验证输入
│           │   ├── X_test_seq.npy               # 测试输入
│           │   ├── y_train.npy                  # 训练目标
│           │   ├── y_val.npy                    # 验证目标
│           │   ├── y_test.npy                   # 测试目标
│           │   ├── test_timestamps.csv          # 测试集时间戳
│           │   ├── scaler_params.json           # 标准化器参数
│           │   └── window_meta.json             # 窗口元数据
│           ├── models/                          # 训练好的模型
│           │   ├── lstm.pt                     # LSTM 模型
│           │   ├── bilstm.pt                   # BiLSTM 模型
│           │   ├── cnn_lstm.pt                 # CNN-LSTM 模型
│           │   ├── cnn_bilstm.pt               # CNN-BiLSTM 模型
│           │   └── afsa_patchtst.pt            # AFSA-PatchTST 模型
│           ├── predictions/                     # 测试集预测结果
│           │   ├── lstm_test.csv
│           │   ├── bilstm_test.csv
│           │   ├── cnn_lstm_test.csv
│           │   ├── cnn_bilstm_test.csv
│           │   └── afsa_patchtst_test.csv
│           ├── metrics/                         # 训练历史与指标
│           │   ├── lstm_train_history.csv
│           │   ├── bilstm_train_history.csv
│           │   ├── cnn_lstm_train_history.csv
│           │   ├── cnn_bilstm_train_history.csv
│           │   ├── afsa_patchtst_train_history.csv
│           │   ├── afsa_patchtst_metrics.csv
│           │   ├── afsa_patchtst_best_params.json
│           │   ├── afsa_patchtst_search_history.csv
│           │   └── exp_p03_model_comparison.csv
│           ├── figures/                         # 可视化图表
│           │   ├── loss_lstm.png                # LSTM 训练损失曲线
│           │   ├── loss_bilstm.png              # BiLSTM 训练损失曲线
│           │   ├── loss_cnn_lstm.png            # CNN-LSTM 训练损失曲线
│           │   ├── loss_cnn_bilstm.png          # CNN-BiLSTM 训练损失曲线
│           │   ├── pred_lstm.png                # LSTM 预测曲线
│           │   ├── pred_bilstm.png              # BiLSTM 预测曲线
│           │   ├── pred_cnn_lstm.png            # CNN-LSTM 预测曲线
│           │   ├── pred_cnn_bilstm.png          # CNN-BiLSTM 预测曲线
│           │   ├── pred_afsa_patchtst.png       # AFSA-PatchTST 预测曲线
│           │   ├── prediction_overlay_all_models.png  # 多模型叠加预测图
│           │   ├── metrics_comparison.png       # 指标柱状图对比
│           │   ├── training_time_comparison.png # 训练耗时对比图
│           │   └── afsa_patchtst_search_curve.png     # AFSA 搜索曲线
│           └── reports/
│               ├── exp_p03_report.md            # 简要实验报告
│               └── hybrid_deep_learning_experiment_report.md  # 完整实验报告
│
├── experiments/                                 # 实验脚本目录
│   ├── README.md                                # 实验目录说明
│   ├── logs/                                    # 实验日志副本
│   │   └── prediction/
│   │       └── step3_hybrid_models/
│   │           ├── EXP-P03_CNN-LSTM.log
│   │           └── EXP-P03_CNN_BiLSTM.log
│   │
│   ├── prediction/                              # 预测任务实验
│   │   ├── step1_data_cleaning_alignment/       # EXP-P01 数据预处理
│   │   │   ├── README.md
│   │   │   └── run_exp_p01_preprocessing.py     # 数据清洗与时间对齐入口脚本
│   │   │
│   │   ├── step2_baseline_models/               # EXP-P02 基础模型
│   │   │   ├── README.md
│   │   │   ├── exp_p02_common.py                # 共享工具函数
│   │   │   ├── exp_p02_torch_utils.py           # PyTorch 训练工具
│   │   │   ├── run_exp_p02_prepare_samples.py   # 样本准备
│   │   │   ├── run_exp_p02_train_lstm.py        # LSTM 训练
│   │   │   ├── run_exp_p02_train_bilstm.py      # BiLSTM 训练
│   │   │   ├── run_exp_p02_train_cnn_lstm.py    # CNN-LSTM 训练
│   │   │   ├── run_exp_p02_train_cnn_bilstm.py  # CNN-BiLSTM 训练
│   │   │   ├── run_exp_p02_train_bp.py          # BP 神经网络训练
│   │   │   ├── run_exp_p02_train_svr.py         # SVR 训练
│   │   │   ├── run_exp_p02_train_rf.py          # 随机森林训练
│   │   │   └── run_exp_p02_summarize_results.py # 结果汇总
│   │   │
│   │   └── step3_hybrid_models/                 # EXP-P03 混合模型
│   │       ├── exp_p03_common.py                # 共享路径/配置/指标工具
│   │       ├── exp_p03_models.py                # 模型定义（LSTM/BiLSTM/CNN/PatchTST）
│   │       ├── exp_p03_torch_utils.py           # PyTorch 训练/预测工具
│   │       ├── exp_p03_afsa.py                  # AFSA 人工鱼群算法实现
│   │       ├── debug_patchtst.py                # PatchTST 调试脚本
│   │       ├── run_exp_p03_prepare_samples.py   # 样本准备（滑动窗口）
│   │       ├── run_exp_p03_train_lstm.py        # LSTM 训练入口
│   │       ├── run_exp_p03_train_bilstm.py      # BiLSTM 训练入口
│   │       ├── run_exp_p03_train_cnn_lstm.py    # CNN-LSTM 训练入口
│   │       ├── run_exp_p03_train_cnn_bilstm.py  # CNN-BiLSTM 训练入口
│   │       ├── run_exp_p03_afsa_patchtst.py     # AFSA-PatchTST 完整流程
│   │       ├── run_exp_p03_afsa_final.py         # AFSA-PatchTST 快速训练（已知最优参数）
│   │       └── run_exp_p03_summarize.py          # 结果汇总与可视化
│   │
│   └── scheduling/                              # 预留调度任务目录
│       └── README.md
│
└── logs/                                        # 运行日志目录
    ├── README.md
    └── prediction/
        ├── step1_data_cleaning_alignment/
        │   └── EXP-P01.log
        ├── step2_baseline_models/
        │   ├── EXP-P02_prepare.log
        │   ├── EXP-P02_LSTM.log
        │   ├── EXP-P02_BiLSTM.log
        │   ├── EXP-P02_RF.log
        │   ├── EXP-P02_SVR.log
        │   ├── EXP-P02_BP.log
        │   └── EXP-P02_BiLSTM.log
        └── step3_hybrid_models/
            ├── EXP-P03_prepare.log
            ├── EXP-P03_LSTM.log
            ├── EXP-P03_BiLSTM.log
            ├── EXP-P03_CNN_LSTM.log
            ├── EXP-P03_CNN_BiLSTM.log
            ├── EXP-P03_AFSA.log
            ├── EXP-P03_AFSA_Final.log
            └── EXP-P03_summarize.log
```

## 核心模块说明

### EXP-P01: 数据清洗与时间对齐

| 文件 | 说明 |
|------|------|
| `run_exp_p01_preprocessing.py` | 数据清洗、缺失值填补、时间对齐主脚本 |
| `Site_*_preprocessed.csv` | 8 个站点的预处理后数据 |
| `solar_dispatch_panel_common_window.csv` | 多站调度面板共同时间窗口 |
| `solar_site_quality_summary.csv` | 各站点数据质量汇总 |

**关键处理：**
- 15 分钟统一时间轴
- 多站点时间对齐（截止 2020-07-01 23:45:00）
- 缺失值填补与异常值修复

### EXP-P02: 五类基础模型对比

| 模型 | 脚本 | 说明 |
|------|------|------|
| LSTM | `run_exp_p02_train_lstm.py` | 单向 LSTM 基线 |
| BiLSTM | `run_exp_p02_train_bilstm.py` | 双向 LSTM 基线 |
| CNN-LSTM | `run_exp_p02_train_cnn_lstm.py` | CNN + LSTM 混合 |
| CNN-BiLSTM | `run_exp_p02_train_cnn_bilstm.py` | CNN + BiLSTM 混合 |
| BP | `run_exp_p02_train_bp.py` | 反向传播神经网络 |
| SVR | `run_exp_p02_train_svr.py` | 支持向量回归 |
| Random Forest | `run_exp_p02_train_rf.py` | 随机森林 |

**关键结果：** BiLSTM 测试集 RMSE=0.0465、R²=0.971，为五类模型最优。

### EXP-P03: 混合深度学习模型对比

| 模型 | 脚本 | 说明 |
|------|------|------|
| LSTM | `run_exp_p03_train_lstm.py` | 单向时序建模基线 |
| BiLSTM | `run_exp_p03_train_bilstm.py` | 双向时序建模基线 |
| CNN-LSTM | `run_exp_p03_train_cnn_lstm.py` | 验证 CNN 局部特征提取效果 |
| CNN-BiLSTM | `run_exp_p03_train_cnn_bilstm.py` | 验证 CNN+双向时序建模 |
| AFSA-PatchTST | `run_exp_p03_afsa_final.py` | 人工鱼群算法优化 PatchTST |

**实验设计：**
- lookback=16, horizon=1
- 时序 70%/14%/16% 划分（train/val/test）
- 标准化器仅在 train 集拟合
- 测试集不参与 early stopping 和超参数搜索

**关键结果：**

| 模型 | MAE | RMSE | MAPE | R² |
|------|-----|------|------|-----|
| LSTM | 0.026526 | 0.055880 | 20.84% | 0.957510 |
| BiLSTM | 0.026995 | 0.052554 | 20.94% | 0.962417 |
| **CNN-LSTM** | **0.024367** | **0.047724** | **21.08%** | **0.969009** |
| CNN-BiLSTM | 0.025944 | 0.049592 | 21.44% | 0.966535 |
| AFSA-PatchTST | 0.031536 | 0.056095 | 24.95% | 0.957183 |

**结论：**
- CNN-LSTM 优于 LSTM：CNN 能有效提取局部波动特征
- CNN-BiLSTM 优于 BiLSTM：局部特征与双向时序建模互补
- 最佳模型为 CNN-LSTM（RMSE=0.047724）

## 核心代码模块

### exp_p03_common.py
共享工具模块，包含：
- 路径管理（MODELS_DIR, METRICS_DIR, FIGURES_DIR 等）
- 配置加载、日志设置
- 样本加载（X_train_seq, y_train 等）
- 指标计算（MAE, RMSE, MAPE, R²）
- 可视化函数（loss curve, prediction curve, overlay, metrics bar）

### exp_p03_models.py
模型定义模块，包含：
- `LSTMRegressor`: 单向 LSTM
- `BiLSTMRegressor`: 双向 LSTM
- `CNNLSTMRegressor`: CNN-LSTM 混合模型
- `CNNBiLSTMRegressor`: CNN-BiLSTM 混合模型
- `PatchTSTRegressor`: Patch 时间序列 Transformer
- `build_model()`: 模型构建工厂函数

### exp_p03_torch_utils.py
PyTorch 训练工具模块，包含：
- `get_device()`: 获取计算设备
- `make_loader()`: 创建 DataLoader
- `predict()`: 模型预测
- `train_with_early_stop()`: 带 early stopping 的训练循环

### exp_p03_afsa.py
人工鱼群算法（AFSA）实现，包含：
- `evaluate_params()`: 评估超参数组合
- `afsa_search()`: AFSA 搜索主循环
- `_move_towards()`: 向最优邻居移动
- `_compute_center()`: 计算邻居中心
- `save_search_history()`: 保存搜索历史
- `plot_search_curve()`: 绘制搜索曲线
- `run_full_training()`: 使用最优参数完整训练

## 数据流

```
raw CSV → EXP-P01 → preprocessed CSV → EXP-P02/P03 → samples (.npy) → models (.pt) → predictions (.csv) → metrics/figures/reports
```

## 命名约定

- 实验编号：EXP-P01, EXP-P02, EXP-P03
- 脚本命名：`run_exp_<编号>_<功能>.py`
- 模块命名：`exp_<编号>_<模块名>.py`
- 日志命名：`EXP-<编号>_<描述>.log`
- 模型文件：`<model_name>.pt`
- 预测文件：`<model_name>_test.csv`
- 指标文件：`<model_name>_train_history.csv`

## 重要说明

1. **数据划分**：所有实验严格按时间顺序划分，禁止随机打乱
2. **标准化**：仅在 train 集拟合 scaler，val/test 使用 transform
3. **测试集保护**：测试集不参与训练、验证或超参数搜索
4. **可复现性**：保存 scaler 参数和随机种子

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

### 日志文件
- `logs/prediction/step1_data_cleaning_alignment/EXP-P01.log`
- `logs/prediction/step2_baseline_models/EXP-P02_*.log`
- `logs/prediction/step3_hybrid_models/EXP-P03_*.log`
