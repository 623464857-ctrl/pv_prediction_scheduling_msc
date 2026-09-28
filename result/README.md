# 实验结果文件夹 (Result Directory)

本文件夹存放亳州 (Bozhou) 光伏预测实验的最终结果文件。

## 目录结构

```
result/                              # 与 experiments/prediction 同级
├── figures/                         # 预测结果图表
│   ├── bozhou_all_horizons_prediction_curve.png    # 多预测步长对比图 (PNG)
│   ├── bozhou_all_horizons_prediction_curve.pdf   # 多预测步长对比图 (PDF)
│   ├── bozhou_all_horizons_prediction_curve_4days.png  # 4天详细预测图
│   └── bozhou_all_horizons_prediction_curve_4days.pdf   # 4天详细预测图
│
├── logs/                            # 日志文件
│   └── *.log                        # 实验运行日志
│
├── metrics/                         # 评估指标
│   ├── bozhou_h1/                   # H=1 (15分钟预测)
│   │   ├── bozhou_cnn_bilstm_optuna.json
│   │   └── bozhou_hybrid_search_ablation.json
│   ├── bozhou_h4/                   # H=4 (1小时预测)
│   │   ├── bozhou_cnn_bilstm_optuna.json
│   │   └── bozhou_hybrid_search_ablation.json
│   └── bozhou_h16/                  # H=16 (4小时预测)
│       ├── bozhou_cnn_bilstm_optuna.json
│       └── bozhou_hybrid_search_ablation.json
│
├── models/                          # 训练好的模型
│   ├── bozhou_h1/                   # H=1 模型
│   │   └── bozhou_cnn_bilstm_seed42.pt
│   ├── bozhou_h4/                   # H=4 模型
│   │   └── bozhou_cnn_bilstm_seed42.pt
│   └── bozhou_h16/                  # H=16 模型
│       └── bozhou_cnn_bilstm_seed42.pt
│
├── predictions/                     # 预测结果文件
│   ├── bozhou_h1/predictions.csv
│   ├── bozhou_h4/predictions.csv
│   └── bozhou_h16/predictions.csv
│
├── samples/                         # 数据样本 (用于模型输入)
│   ├── bozhou_h1_lb16/              # H=1, Lookback=16
│   │   ├── meta.json
│   │   ├── scaler_params.json
│   │   ├── test_timestamps.csv
│   │   ├── X_test_seq.npy
│   │   ├── X_train_seq.npy
│   │   ├── X_val_seq.npy
│   │   ├── y_test.npy
│   │   ├── y_train.npy
│   │   └── y_val.npy
│   ├── bozhou_h4_lb48/             # H=4, Lookback=48
│   │   └── ...
│   └── bozhou_h16_lb96/            # H=16, Lookback=96
│       └── ...
│
├── scripts/                         # 绘图脚本
│   ├── plot_bozhou_predictions.py    # 从预测结果CSV绘图
│   └── plot_bozhou_from_model.py    # 从模型加载后绘图
│
└── README.md                        # 本文件
```

## 文件说明

### 图表文件 (figures/)

| 文件名 | 说明 |
|--------|------|
| `bozhou_all_horizons_prediction_curve.png/pdf` | H=1, H=4, H=16 三个预测步长的对比图 |
| `bozhou_all_horizons_prediction_curve_4days.png/pdf` | 4天时间范围的详细预测曲线图 |

### 指标文件 (metrics/)

包含超参数优化 (Optuna) 和消融实验 (Ablation) 的评估结果。

### 模型文件 (models/)

训练好的 CNN-BiLSTM 模型权重文件，使用随机种子 42。

## 使用方法

### 绘图脚本

#### 1. 从预测结果绘图 (推荐)

如果已有预测结果 CSV 文件：

```bash
# 进入 result/scripts 目录
cd result/scripts

# 绘制标准多步长对比图
python plot_bozhou_predictions.py

# 绘制4天详细视图
python plot_bozhou_predictions.py --4days

# 指定自定义数据目录
python plot_bozhou_predictions.py --data-dir ../../data/prediction/step2_hyperparameter_search/predictions

# 指定输出路径
python plot_bozhou_predictions.py --output ../figures/custom_plot.png
```

#### 2. 从模型加载绘图

需要模型文件、样本数据和 scaler 参数：

```bash
cd result/scripts

# 运行模型并绘图
python plot_bozhou_from_model.py

# 指定自定义目录
python plot_bozhou_from_model.py --models-dir ../../data/prediction/step2_hyperparameter_search/models

# 只处理特定 horizon
python plot_bozhou_from_model.py --horizons 1 4
```

### 加载模型进行预测

```python
import torch
from pathlib import Path

model_path = Path("result/models/bozhou_h1/bozhou_cnn_bilstm_seed42.pt")
model = torch.load(model_path)
```

## 项目结构

```
pv_prediction_scheduling_msc_new/    # 项目根目录
├── result/                          # 实验结果文件夹 (本目录)
│   ├── figures/                     # 图表
│   ├── logs/                        # 日志
│   ├── metrics/                     # 评估指标
│   ├── models/                      # 模型权重
│   ├── predictions/                 # 预测结果
│   ├── samples/                     # 数据样本
│   ├── scripts/                     # 绘图脚本
│   └── README.md                    # 本文件
│
├── experiments/                     # 实验代码
│   └── prediction/                  # 预测实验
│       ├── step1_preprocessing/    # 数据预处理
│       ├── step2_hyperparameter_search/  # 超参数搜索
│       ├── step3_deep_learning/    # 深度学习模型
│       ├── step4_evaluation/        # 评估
│       └── step5_reporting/         # 报告
│
└── data/                            # 原始数据和处理后数据
    ├── data-get/                    # 数据获取
    ├── data-analysis/              # 数据分析
    └── prediction/                  # 预测相关数据
```

## 相关信息

- **数据集**: 亳州光伏电站数据
- **预测步长**: H=1 (15分钟), H=4 (1小时), H=16 (4小时)
- **模型**: CNN-BiLSTM
- **创建日期**: 2026-09-28
