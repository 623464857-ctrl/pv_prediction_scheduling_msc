# 光伏预测与调度研究项目

光伏功率预测与优化调度模型研究项目，基于明月湖光伏电站数据。

## 项目结构

```
pv_prediction_scheduling_msc_new/
├── data/                           # 数据目录
│   ├── raw/                        # 原始数据
│   ├── data-analysis/             # 数据分析
│   │   ├── reports/               # 分析报告
│   │   └── charts/                # 分析图表
│   └── prediction/                # 预测相关数据
│       ├── step1_preprocessing/    # 数据预处理
│       ├── step2_hyperparameter_search/  # 超参数搜索
│       ├── step3_deep_learning/   # 深度学习模型
│       ├── step4_evaluation/      # 模型评估
│       └── step5_reporting/       # 报告生成
│
├── experiments/                     # 实验脚本
│   └── prediction/
│       ├── step1_preprocessing/   # 数据预处理脚本
│       ├── step2_hyperparameter_search/  # 超参数搜索脚本
│       ├── step3_deep_learning/   # 深度学习训练脚本
│       ├── step4_evaluation/      # 模型评估脚本
│       └── step5_reporting/       # 报告生成脚本
│
├── logs/                           # 日志文件
├── reports/                        # 研究报告
└── archive/                        # 历史代码存档
```

---

## 模型架构详解

### 一、数据预处理 (Step 1)

数据预处理是整个预测流水线的基础，确保数据质量满足模型训练要求。

#### 1.1 数据清洗流程

```
原始数据 → 字段统一 → 物理约束清洗 → 白天/夜间处理 → 缺失值插值 → 偏度校正 → 标准化
```

| 处理阶段 | 方法 | 说明 |
|---------|------|------|
| 字段统一 | 别名映射 | 支持多种列名格式（中文、英文、缩写） |
| 物理约束 | 边界裁剪 | 根据物理意义设定合理范围（如辐照度0-1600 W/m²） |
| 夜间处理 | 功率清零 | 太阳高度角<0时，功率置0 |
| 缺失值 | 线性插值 | 时间序列线性填补 |
| 偏度校正 | PowerTransformer | Yeo-Johnson变换，减少特征偏度 |
| 标准化 | StandardScaler | 零均值单位方差标准化 |

#### 1.2 核心特征

| 特征名 | 单位 | 说明 |
|-------|------|------|
| temperature_c | °C | 环境温度 |
| ghi_wm2 | W/m² | 水平面总辐照度 |
| dni_wm2 | W/m² | 直接法向辐照度 |
| dhi_wm2 | W/m² | 水平面散射辐照度 |
| relative_humidity_pct | % | 相对湿度 |
| atmosphere_hpa | hPa | 大气压 |
| wind_speed_ms | m/s | 风速 |

#### 1.3 数据集划分

- **训练集**: 67% (时间序列前段)
- **验证集**: 13%
- **测试集**: 20% (时间序列末端)

---

### 二、超参数搜索 (Step 2)

#### 2.1 搜索算法

项目采用 **Optuna + AFSA 混合搜索策略**：

| 策略 | 描述 |
|-----|------|
| Optuna (RandomSampler) | 随机采样 |
| Optuna (CmaEsSampler) | 协方差矩阵自适应进化策略 |
| AFSA | 人工鱼群算法 (Artificial Fish Swarm Algorithm) |
| S3 | Optuna → AFSA 串行（先粗搜索再精调） |
| S4 | Optuna + AFSA 并行（双种群同时搜索） |
| S6 | 多策略集成（加权投票） |

#### 2.2 搜索空间

| 参数 | 范围/选项 |
|-----|----------|
| learning_rate | [0.0005, 0.001, 0.005, 0.01] |
| batch_size | [32, 64, 128, 256] |
| hidden_size | [32, 64, 128] |
| num_layers | [1, 2, 3] |
| dropout | [0.1, 0.2, 0.3, 0.4, 0.5] |
| cnn_channels | [(16,32), (32,64), (64,128)] |

#### 2.3 交叉验证

采用 **滚动时间窗口交叉验证** (Rolling Time Series CV)：
- 滚动窗口数: 3
- 每折训练比例: 67%

---

### 三、深度学习模型 (Step 3)

#### 3.1 CNN-BiLSTM 模型架构

```
┌─────────────────────────────────────────────────────────────┐
│                        Input Layer                          │
│                    (seq_len, n_features)                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    CNN Block (1D)                           │
│  Conv1d → BatchNorm → ReLU → Dropout                        │
│  Conv1d → BatchNorm → ReLU → Dropout                        │
│               (kernel_size=3, padding=1)                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   BiLSTM Layer                              │
│         Bidirectional LSTM (hidden_size * 2)                │
│              num_layers=2, dropout=0.2                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Fully Connected Layer                     │
│                  Linear(hidden_size*2 → horizon)            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       Output                               │
│                       (horizon)                             │
└─────────────────────────────────────────────────────────────┘
```

#### 3.2 模型配置

| 预测步长 (H) | 回溯窗口 (L) | 时间跨度 | 隐藏层大小 | LSTM层数 |
|-------------|-------------|---------|-----------|---------|
| H=1 | L=16 | 4小时 | 64 | 2 |
| H=4 | L=48 | 12小时 | 64 | 2 |
| H=16 | L=96 | 24小时 | 64 | 2 |

#### 3.3 超参数

| 参数 | 值 |
|-----|-----|
| CNN通道数 | (32, 64) |
| 卷积核大小 | 3 |
| Dropout | 0.2 |
| 学习率 | 0.001 |
| 优化器 | Adam |
| 早停耐心 | 8 epochs |
| 最大训练轮次 | 50 |

#### 3.4 损失函数

```
Loss = MSE(y_true, y_pred) = Mean((y_true - y_pred)²)
```

---

### 四、模型评估 (Step 4)

#### 4.1 评估指标

| 指标 | 公式 | 说明 |
|-----|------|------|
| MAE | Mean(|y_true - y_pred|) | 平均绝对误差 |
| RMSE | √Mean((y_true - y_pred)²) | 均方根误差 |
| MAPE | Mean(|(y_true - y_pred) / y_true|) × 100 | 平均绝对百分比误差 |
| R² | 1 - SS_res / SS_tot | 决定系数 |

#### 4.2 早停策略

```python
if val_loss > best_val_loss * patience_factor:
    patience_counter += 1
    if patience_counter >= patience:
        stop training
```

---

### 五、预测结果

明月湖数据集 CNN-BiLSTM 模型测试集性能（5个随机种子平均，seed=42 用于绘图）：

| Horizon | MAE   | RMSE  | MAPE (%) | R²    |
|---------|-------|-------|----------|-------|
| H=1     | 0.0336 | 0.0810 | 112.74 | 0.5754 |
| H=4     | 0.0436 | 0.0970 | 139.83 | 0.4864 |
| H=16    | 0.0539 | 0.1081 | 153.36 | 0.3583 |

> 注：MAPE 为平均绝对百分比误差；指标值来源于 `mingyuehu_cnn_bilstm_reproduce.json`（seed=42）

---

## 数据集

- **数据来源**: 明月湖光伏电站
- **装机容量**: 281.6 kW
- **时间分辨率**: 15分钟
- **时间范围**: 2026年6月 - 2026年8月
- **数据划分**: 训练集 / 验证集 / 测试集

---

## 实验流程

```bash
# Step 1: 数据预处理
python -m experiments.prediction.step1_preprocessing.run_exp_p01_mingyuehu

# Step 2: 超参数搜索
python -m experiments.prediction.step2_hyperparameter_search.run_exp_p04_optuna_mingyuehu --horizon 1

# Step 3: 深度学习训练与预测
python -m experiments.prediction.step3_deep_learning.run_exp_p04_reproduce_mingyuehu --horizon 1

# Step 4: 模型评估
python -m experiments.prediction.step4_evaluation.run_exp_p04_validation_mingyuehu

# Step 5: 生成报告
python -m experiments.prediction.step5_reporting.run_exp_p04_report_mingyuehu
```

---

## 技术栈

- Python 3.8+
- **PyTorch** - 深度学习框架
- **Scikit-learn** - 数据预处理与评估
- **Optuna** - 超参数优化
- **AFSA** - 人工鱼群算法
- **Pandas/NumPy** - 数据处理
- **Matplotlib** - 可视化
- **SciPy** - 科学计算（插值、统计）
