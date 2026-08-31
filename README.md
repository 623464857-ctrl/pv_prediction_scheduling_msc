# 光伏预测与调度协同研究工程

## 项目概述

本项目用于 **Site_4** 短期光伏发电功率预测研究，当前主线为：

1. **EXP-P01** — 数据清洗与时间对齐  
2. **EXP-P04** — Optuna 超参搜索 + CNN-BiLSTM 残差预测（多 Horizon）  
3. **EXP-P05** — CNN-BiLSTM 残差建模深化与混合搜索  

早期对比实验（P02 基线模型、P03 混合模型对比、WRF 消融、多站质量评估）已从仓库移除。

## 当前进度

| 实验 | 名称 | 状态 |
|------|------|------|
| EXP-P01 | 数据清洗与时间对齐（Site_4） | ✅ 已完成 |
| EXP-P04 | Optuna 超参优化 + CNN-BiLSTM 残差预测 | 🔄 代码已重构，待重跑 |
| EXP-P05 | CNN-BiLSTM 残差建模与混合搜索 | ✅ 历史产出保留 |
| EXP-P10 | 数据划分一致性审计与 Oracle 根因实验 | ✅ 已完成 |

## 快速开始

### 环境要求

- Python 3.13+
- PyTorch
- NumPy, Pandas, Scikit-learn
- Matplotlib, Optuna

### 运行实验

```powershell
# Step 1: 数据预处理（需先 git lfs pull 拉取原始 CSV）
python experiments/prediction/step1_data_cleaning_alignment/run_exp_p01_preprocessing.py

# Step 4: Optuna 超参优化 + CNN-BiLSTM 残差预测
python -m experiments.prediction.step4_optuna_hybrid.run_exp_p04_prepare_samples --horizon 1
python -m experiments.prediction.step4_optuna_hybrid.run_exp_p04_optuna --horizon 1
python -m experiments.prediction.step4_optuna_hybrid.run_exp_p04_train_final --horizon 1
python -m experiments.prediction.step4_optuna_hybrid.run_exp_p04_reproduce --horizon 1
python -m experiments.prediction.step4_optuna_hybrid.run_exp_p04_report --horizon 1
# (--horizon 可选 1/4/16，分别对应 15min/1h/4h 预测)

# Step 5: 历史产出位于 data/prediction/step5_new_experiments/
# 入口脚本已从仓库移除，如需重跑可从 git 历史恢复
```

## 目录结构

```
pv_prediction_scheduling_msc/
├── README.md
├── data/
│   ├── raw/                                    # 原始数据（Git LFS）
│   └── prediction/
│       ├── step1_preprocessing/                # EXP-P01
│       │   └── processed/stations/Site_4_preprocessed.csv
│       ├── step4_optuna_hybrid/                # EXP-P04
│       │   ├── config/exp_p04_*.json
│       │   ├── samples/h{H}_lb{LB}/            # 滑动窗口样本
│       │   ├── models/h{1,4,16}/
│       │   ├── metrics/h{1,4,16}/
│       │   └── reports/
│       └── step5_new_experiments/              # EXP-P05 历史产出
│
├── experiments/prediction/
│   ├── step1_data_cleaning_alignment/        # P01 脚本
│   ├── step4_optuna_hybrid/                  # P04 脚本
│   └── step10_data_audit/                    # P10 审计报告
│
└── logs/prediction/
```

## 核心实验结果

### EXP-P04: Optuna 超参优化与 CNN-BiLSTM 残差预测

**实验设计：**
- 三个预测步长：h1=15min, h4=1h, h16=4h
- 13 维 step1 特征（辐照/气象 + 时间周期 + ramp + 质量分）
- 残差目标：`Δy = y(t+H) - y(t-1)`，重构 `y_hat = y_anchor + Δy_hat`
- 样本目录：`samples/h{H}_lb{LB}/`（默认 lookback=16）
- Optuna-AFSA 混合搜索 S2-S6（各 20 trials），综合评分选最优策略
- 最终训练 50 epochs（patience=8），多 seed（42/43/44/45/46）复现

> 历史多模型对比结果已作废；重构后仅保留 CNN-BiLSTM，需重跑后更新指标。

### EXP-P05: CNN-BiLSTM 残差建模（历史产出）

历史实验以 **CNN-BiLSTM 残差预测** 为核心，产出位于 `data/prediction/step5_new_experiments/`。基线对比（XGB/LGBM 等）相关文件已清理。

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
CNN-BiLSTM 残差模型定义与 `build_model()` 工厂函数。

### exp_p04_cv.py
时序交叉验证模块，`RollingWindowCV`：Rolling Window 时序交叉验证，避免时间序列数据泄露。

### exp_p04_features.py
特征工程模块（lag / ramp / rolling 统计，无未来泄漏）。

### step10_data_audit/
数据审计工具：各 Phase 独立审计脚本与日志。

## 数据流

```
raw CSV → EXP-P01 → Site_4_preprocessed.csv
                  ↘ EXP-P04: prepare_samples → h{H}_lb{LB}/ → Optuna → train → reproduce → report
                  ↘ EXP-P05: CNN-BiLSTM 残差建模（历史产出）
                  ↘ EXP-P10: 7 阶段数据审计
```

## 命名约定

- 实验编号：EXP-P01, EXP-P04, EXP-P05, EXP-P10
- 脚本命名：`run_exp_<编号>_<功能>.py`
- 模块命名：`exp_<编号>_<模块名>.py`
- Horizon 标识：`h1` (15min) / `h4` (1h) / `h16` (4h)
- 样本目录：`h{horizon}_lb{lookback}/`（如 `h1_lb16/`）

## 重要说明

1. **数据划分**：所有实验严格按时间顺序划分，禁止随机打乱
2. **标准化**：仅在 train 集拟合 scaler，val/test 使用 transform
3. **测试集保护**：测试集不参与训练、验证或超参数搜索
4. **可复现性**：保存 scaler 参数和随机种子
5. **Site_4 专用**：当前仅保留 Site_4 预处理数据；原始 CSV 需 `git lfs pull`

## 产出文件索引

### EXP-P01 产出
- `data/prediction/step1_preprocessing/processed/stations/Site_4_preprocessed.csv`

### EXP-P04 产出
- `data/prediction/step4_optuna_hybrid/samples/h{H}_lb{LB}/*.npy`
- `data/prediction/step4_optuna_hybrid/models/h{1,4,16}/*.pt`
- `data/prediction/step4_optuna_hybrid/predictions/h{1,4,16}/*.csv`
- `data/prediction/step4_optuna_hybrid/metrics/h{1,4,16}/*.json`
- `data/prediction/step4_optuna_hybrid/reports/EXP-P04_h*_详细实验汇报.md`

### EXP-P05 产出
- `data/prediction/step5_new_experiments/predictions/h*/cnn_bilstm_residual_test.csv`
- `data/prediction/step5_new_experiments/metrics/h*/residual_metrics.json`
- `data/prediction/step5_new_experiments/reports/FINAL_综合实验报告.md`

### EXP-P10 产出
- `experiments/prediction/step10_data_audit/STEP10_MAIN_LOG.md`
- `experiments/prediction/step10_data_audit/PHASE{1-7}_*_LOG.md`

### 日志文件
- `logs/prediction/step1_data_cleaning_alignment/EXP-P01.log`
- `logs/prediction/step4_optuna_hybrid/EXP-P04_h*_*.log`
- `logs/prediction/step5_new_experiments/EXP-P05_h*_*.log`
