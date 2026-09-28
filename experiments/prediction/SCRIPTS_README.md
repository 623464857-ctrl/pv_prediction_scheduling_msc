# 实验脚本文件结构文档

本文档说明了 `experiments/prediction/` 目录下各脚本的功能和关系。

---

## 目录结构

```
experiments/prediction/
├── step1_preprocessing/           # 数据预处理
├── step2_hyperparameter_search/   # 超参数搜索
├── step3_deep_learning/           # 深度学习模型训练
├── step4_evaluation/             # 模型评估与可视化
└── step5_reporting/              # 报告生成
```

---

## 1. step1_preprocessing - 数据预处理

| 脚本文件 | 功能描述 |
|---------|---------|
| `exp_01_preprocess_bozhou.py` | 亳州数据集预处理（清洗、标准化） |
| `exp_01_split_data.py` | 数据集划分（70/15/15 时间顺序划分） |
| `exp_01_analyze_correlation.py` | 特征相关性分析（特征间相关性 + 特征与功率相关性） |
| `exp_01_plot_distributions.py` | 特征分布和散点图生成 |
| `exp_01_scaler_comparison.py` | StandardScaler vs Yeo-Johnson 对比实验 |
| `exp_01_prove_scaler_redundancy.py` | 证明 StandardScaler 冗余性的理论分析 |

---

## 2. step2_hyperparameter_search - 超参数搜索

| 脚本文件 | 功能描述 |
|---------|---------|
| `exp_02_common.py` | 共享路径、配置、日志和指标计算工具（含亳州专用函数） |
| `exp_02_cv_split.py` | 滚动窗口交叉验证划分工具 |
| `exp_02_hybrid_optimizer.py` | Optuna-AFSA 混合超参搜索策略（S1-S6） |
| `exp_02_prepare_samples_bozhou.py` | 亳州数据集样本准备 |
| `exp_02_hybrid_search_bozhou.py` | 亳州数据集混合搜索 |
| `exp_02_quick_experiment_bozhou.py` | 亳州数据集快速实验 |

---

## 3. step3_deep_learning - 深度学习模型训练

### 模型定义
| 脚本文件 | 功能描述 |
|---------|---------|
| `exp_03_models.py` | CNN-BiLSTM 模型定义 |
| `exp_03_torch_utils.py` | PyTorch 训练工具（数据加载、训练循环、预测） |

### 亳州训练脚本
| 脚本文件 | 功能描述 |
|---------|---------|
| `exp_03_train_bozhou.py` | 亳州数据集模型训练（使用Optuna最佳参数） |

### 毛刺感知模块
| 脚本文件 | 功能描述 |
|---------|---------|
| `exp_03_surge_loss.py` | 所有毛刺感知损失函数（基础版 + 增强版） |
| `exp_03_surge_features.py` | 毛刺特征提取（检测、提取、注入） |
| `exp_03_surge_training.py` | 毛刺感知训练流程（数据集、训练函数） |
| `exp_03_prepare_surge_samples.py` | 毛刺感知样本准备 |
| `exp_03_surge_models.py` | 毛刺感知模型变体（SurgeCNNBiLSTM、门控、注意力等） |

---

## 4. step4_evaluation - 模型评估与可视化

| 脚本文件 | 功能描述 |
|---------|---------|
| `exp_04_evaluate_bozhou.py` | 亳州数据集预测、评估指标计算、预测曲线绘图 |

---

## 5. step5_reporting - 报告生成

| 脚本文件 | 功能描述 |
|---------|---------|
| `exp_05_step_audit.py` | 实验分步审计（记录每步运行状态、生成 manifest） |

---

## 6. 其他脚本

| 脚本文件 | 位置 | 功能描述 |
|---------|------|---------|
| `fetch_bozhou.py` | `data/data-get/` | 亳州天气数据获取（Weatherbit API） |
| `data_quality_check.py` | `data/prediction/step1_preprocessing/` | 数据质量检查和处理 |

---

## 脚本功能关系图

```
数据获取
    ↓
data_quality_check.py (数据质量检查)
    ↓
exp_01_preprocess_bozhou.py (数据预处理)           ← step1
    ↓
exp_01_split_data.py (数据集划分)
    ↓
exp_01_analyze_correlation.py (相关性分析)
    ↓
exp_02_prepare_samples_bozhou.py (样本准备)          ← step2
    ↓
exp_02_hybrid_search_bozhou.py (超参数搜索)
    ↓
exp_03_train_bozhou.py (模型训练)                   ← step3
    ↓
exp_04_evaluate_bozhou.py (评估与绘图)              ← step4
    ↓
exp_05_step_audit.py (审计记录)                    ← step5
```

---

## 使用示例

### 数据预处理 (step1)
```bash
# 亳州数据预处理
python -m experiments.prediction.step1_preprocessing.exp_01_preprocess_bozhou

# 数据集划分
python -m experiments.prediction.step1_preprocessing.exp_01_split_data

# 相关性分析
python -m experiments.prediction.step1_preprocessing.exp_01_analyze_correlation
```

### 超参数搜索 (step2)
```bash
# 样本准备
python -m experiments.prediction.step2_hyperparameter_search.exp_02_prepare_samples_bozhou

# 混合搜索
python -m experiments.prediction.step2_hyperparameter_search.exp_02_hybrid_search_bozhou
```

### 模型训练 (step3)
```bash
# 训练亳州数据集模型
python -m experiments.prediction.step3_deep_learning.exp_03_train_bozhou
```

### 模型评估 (step4)
```bash
# 预测、评估指标计算、绘图
python -m experiments.prediction.step4_evaluation.exp_04_evaluate_bozhou
```

---

## 更新历史

- **2026-09-28**: 统一文件命名为 `exp_0x_<功能>.py` 格式
- **2026-09-28**: 分离训练与评估功能到 step3 和 step4
- **2026-09-28**: 整合 `analyze_power_correlation.py` 和 `analyze_correlation.py` 为 `exp_01_analyze_correlation.py`
- **2026-09-28**: 整合 `exp_p04_surge_loss.py` 和 `exp_p05_surge_loss.py` 为 `exp_03_surge_loss.py`
