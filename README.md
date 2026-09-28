# 光伏预测与调度研究项目

光伏功率预测与优化调度模型研究项目，基于亳州 (Bozhou) 光伏电站数据。

## 项目结构

```
pv_prediction_scheduling_msc_new/
├── data/                           # 数据目录
│   ├── raw/                        # 原始数据
│   ├── data-analysis/              # 数据分析报告和图表
│   │   ├── reports/                # 分析报告
│   │   ├── charts/                 # 分析图表
│   │   ├── bozhou_weekly_comparison/
│   │   ├── power_surge_drop_analysis/
│   │   └── power_weather_correlation/
│   ├── data-get/                   # 数据获取脚本
│   └── prediction/                # 预测相关数据
│       ├── step1_preprocessing/    # 数据预处理输出
│       │   ├── processed/          # 预处理后的数据
│       │   │   └── stations/       # 各站点数据
│       │   └── data_quality/       # 数据质量报告
│       ├── step2_hyperparameter_search/  # 超参数搜索结果
│       ├── step3_deep_learning/    # 深度学习模型
│       ├── step4_evaluation/       # 模型评估结果
│       └── step5_reporting/        # 报告生成
│
├── experiments/                     # 实验脚本
│   └── prediction/
│       ├── step1_preprocessing/    # 数据预处理脚本
│       ├── step2_hyperparameter_search/  # 超参数搜索脚本
│       ├── step3_deep_learning/    # 深度学习训练脚本
│       ├── step4_evaluation/       # 模型评估脚本
│       └── step5_reporting/        # 报告生成脚本
│
├── logs/                           # 日志文件
├── reports/                        # 研究报告
└── result/                        # 实验结果输出
```

---

## 一、数据预处理 (Step 1)

数据预处理是整个预测流水线的基础，确保数据质量满足模型训练要求。

### 1.1 数据清洗流程

亳州数据集预处理共包含 12 个步骤：

```
原始数据 → 数据合并 → 字段标准化 → 时间序列重建 → 物理边界清洗
    → 功率约束 → Hampel异常检测 → 辐照-功率一致性修正 → 短时插值
    → 长缺失回填 → 白天/夜间分离 → 停机标记 → 衍生特征构建
```

| 处理阶段 | 方法 | 说明 |
|---------|------|------|
| 字段标准化 | 别名映射 | 统一列名格式（temp→temperature_c等） |
| 物理边界清洗 | 边界裁剪 | 设定合理范围（辐照度0-1600 W/m²等） |
| 功率约束 | 边界裁剪 | 负功率清零，超容量标记为NaN |
| 异常检测 | Hampel滤波器 | 基于中位数±nσ检测天气/功率异常 |
| 辐照-功率一致性 | 物理约束 | 无辐照时功率过高则标记为异常 |
| 短时缺失插值 | 分特征策略 | ≤4步：线性/样条/前后填充，按特征类型选择 |
| 长缺失回填 | 分层分组中位数 | 月+小时、星期+小时等多层级回填 |
| 白天/夜间 | GHI阈值 | GHI>20 W/m²为白天，夜间功率强制置零 |
| 停机标记 | 条件检测 | 白天条件但功率为零标记为疑似停机 |

### 1.2 核心特征

| 特征名 | 单位 | 说明 |
|-------|------|------|
| temperature_c | °C | 环境温度 |
| ghi_wm2 | W/m² | 水平面总辐照度 |
| dni_wm2 | W/m² | 直接法向辐照度 |
| dhi_wm2 | W/m² | 水平面散射辐照度 |
| relative_humidity_pct | % | 相对湿度 |
| cloud_cover_pct | % | 云量覆盖率 |
| uv_index | — | 紫外线指数 |
| precip_rate_mmhr | mm/h | 降水率 |
| visibility_km | km | 能见度 |
| weather_category | — | 天气类型编码（0-10） |

### 1.3 调度/预测衍生特征

| 特征名 | 说明 |
|-------|------|
| power_pu | 归一化功率（除以装机容量） |
| power_ramp_15m | 15分钟功率变化量 |
| sin_hour / cos_hour | 小时周期特征（正弦/余弦编码） |
| sin_ghi / cos_ghi | 辐照度周期特征 |
| sin_rh / cos_rh | 湿度周期特征 |
| is_daytime | 白天/夜间标记 |
| is_potential_shutdown | 疑似停机标记 |

### 1.4 数据集划分

- **训练集**: 70% (时间序列前段)
- **验证集**: 15%
- **测试集**: 15% (时间序列末端)

---

## 二、超参数搜索 (Step 2)

### 2.1 搜索算法

项目采用 **Optuna + AFSA 混合搜索策略**，支持6种搜索策略：

| 策略 | 描述 |
|-----|------|
| S1 | 纯 Optuna (RandomSampler) - 随机采样 |
| S2 | 纯 AFSA - 人工鱼群算法 (Artificial Fish Swarm Algorithm) |
| S3 | Optuna → AFSA 串行（先粗搜索再精调） |
| S4 | Optuna + AFSA 并行（双种群同时搜索） |
| S5 | Optuna (CmaEsSampler) + AFSA |
| S6 | 多策略集成（所有策略加权投票） |

### 2.2 搜索空间

| 参数 | 范围/选项 |
|-----|----------|
| learning_rate | [0.0005, 0.001, 0.005, 0.01] |
| batch_size | [32, 64, 128, 256] |
| hidden_size | [32, 64, 128] |
| num_layers | [1, 2, 3] |
| dropout | [0.1, 0.2, 0.3, 0.4, 0.5] |
| cnn_channels | [(16,32), (32,64), (64,128)] |

### 2.3 交叉验证

采用 **滚动时间窗口交叉验证** (Rolling Time Series CV)：
- 滚动窗口数: 3
- 每折训练比例: 67%

---

## 三、深度学习模型 (Step 3)

### 3.1 CNN-BiLSTM 模型架构

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
│                  Fully Connected Layer                       │
│                  Linear(hidden_size*2 → horizon)            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       Output                               │
│                       (horizon)                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 模型配置

| 预测步长 (H) | 回溯窗口 (L) | 时间跨度 | 隐藏层大小 | LSTM层数 |
|-------------|-------------|---------|-----------|---------|
| H=1 | L=16 | 4小时 | 64 | 2 |
| H=4 | L=48 | 12小时 | 64 | 2 |
| H=16 | L=96 | 24小时 | 64 | 2 |

### 3.3 损失函数

项目支持多种损失函数，包括标准 MSE 和毛刺感知损失函数：

#### 基础损失函数
| 损失函数 | 说明 |
|---------|------|
| MSE | 均方误差 (默认) |

#### 毛刺感知损失函数 (Surge-Aware Loss)

| 损失函数 | 说明 |
|---------|------|
| VolatilityWeightedLoss | 波动加权损失 - 对高波动样本给予更高权重 |
| RampWeightedLoss | 爬坡加权损失 - 针对功率骤变时刻的加权 |
| PhysicsGuidedCompoundLoss | 物理引导复合损失 |
| FocalLossForVolatility | 波动焦点损失 |
| EdgeAwareLoss | 边缘感知损失 |
| SurgeAwareCompoundLoss | 毛刺感知复合损失 |
| DirectionAwareLoss | 方向感知损失 |
| SurgeIntensityWeightedLoss | 毛刺强度加权损失 |
| EnhancedSurgeAwareCompoundLoss | 增强版毛刺感知复合损失 |

### 3.4 毛刺感知模型变体

| 模型 | 说明 |
|-----|------|
| SurgeCNNBiLSTM | 接受毛刺特征作为额外输入通道 |
| SurgeGateCNNBiLSTM | 使用门控机制融合毛刺特征 |
| SurgeAttentionCNNBiLSTM | 使用注意力机制关注毛刺时刻 |

---

## 四、模型评估 (Step 4)

### 4.1 评估指标

| 指标 | 公式 | 说明 |
|-----|------|------|
| MAE | Mean(\|y_true - y_pred\|) | 平均绝对误差 |
| RMSE | √Mean((y_true - y_pred)²) | 均方根误差 |
| MAPE | Mean(\|y_true - y_pred\| / y_true) × 100 | 平均绝对百分比误差 |
| R² | 1 - SS_res / SS_tot | 决定系数 |

### 4.2 早停策略

```python
if val_loss > best_val_loss * patience_factor:
    patience_counter += 1
    if patience_counter >= patience:
        stop training
```

---

## 五、数据集

| 项目 | 数值 |
|------|------|
| 数据来源 | 亳州光伏电站 |
| 装机容量 | ~220 kW |
| 时间分辨率 | 15分钟 |
| 时间范围 | 2026年3月 - 9月 |

---

## 实验流程

```bash
# Step 1: 数据预处理
python -m experiments.prediction.step1_preprocessing.exp_01_preprocess_bozhou
python -m experiments.prediction.step1_preprocessing.exp_01_split_data
python -m experiments.prediction.step1_preprocessing.exp_01_analyze_correlation

# Step 2: 超参数搜索
python -m experiments.prediction.step2_hyperparameter_search.exp_02_prepare_samples_bozhou
python -m experiments.prediction.step2_hyperparameter_search.exp_02_hybrid_search_bozhou --horizon 1

# Step 3: 深度学习训练
python -m experiments.prediction.step3_deep_learning.exp_03_train_bozhou

# Step 4: 模型评估
python -m experiments.prediction.step4_evaluation.exp_04_evaluate_bozhou

# Step 5: 生成报告
python -m experiments.prediction.step5_reporting.exp_05_step_audit
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

---

## 脚本文件说明

详细脚本说明请参见 [experiments/prediction/SCRIPTS_README.md](experiments/prediction/SCRIPTS_README.md)
