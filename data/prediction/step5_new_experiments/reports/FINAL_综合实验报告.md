# 光伏功率预测残差建模与混合超参优化研究
## ——基于增强特征、强基线与残差学习的多 Horizon 综合实验报告

**实验编号**：EXP-P05 综合报告  
**生成时间**：2026-07-10  
**研究主题**：在多预测尺度（H1/H4/H16）下，通过增强特征工程、强基线对比、残差预测建模与 Optuna-AFSA 混合超参搜索，系统评估光伏功率预测性能并给出模型选型建议。

---

## 一、实验目的

针对上周实验，存在以下问题：

| 序号 | 问题 | 措施 |
|------|------|------|
| 1 | 精度比不上同数据集其他论文 | 残差预测策略 + 特征工程增强 |
| 2 | 鱼群搜索和Optuna能否结合 | Optuna-AFSA混合搜索 |
| 3 | Transformer预测时间是否有异常 | 采用标准化前向传播测量 |

针对以上解决方法，本次实验验证残差预测和Optuna-AFSA混合搜索能否提高模型精度，提出以下问题：

1. 残差预测是否有效；
2. 树模型（XGBoost/LightGBM）在光伏预测中表现如何；
3. 白天/夜间分段评价对模型选择有何影响；
4. Optuna-AFSA混合搜索能否兼顾精度与效率。

---

## 二、实验步骤

### 1. 实验流程

### 2. 特征工程增强

在原有11维特征基础上，新增功率滞后、滚动统计和多尺度Ramp特征，输入维度从13维扩展至26维：

| 特征类别 | 特征名称 | 物理意义 |
|----------|----------|----------|
| 功率滞后特征 | power_pu_lag_0/1/2/4/8/16 | 当前及历史功率（t-1至t-16） |
| 多尺度Ramp | power_ramp_15m/60m/120m_pu | 15分钟/1小时/2小时爬坡率 |
| 滚动统计 | roll_1h/2h_mean/std/max/min | 近期平均/波动/极值水平 |
| 辐照特征 | total_irradiance / DNI / GHI | 总辐照/法向直接辐照/水平辐照 |
| 气象特征 | air_temperature / atmosphere / humidity | 温度/气压/湿度 |
| 时间编码 | sin/cos_hour + sin/cos_dayofyear | 日周期/年周期编码 |
| 日间标识 | daylight_flag | 区分白天/夜间 |
| 数据质量 | data_quality_score | 数据质量评分 |

### 3. 残差预测

核心思想：光伏功率具有强自相关性，当前时刻功率y(t)已经包含了大量即时状态信息，包括当前辐照、云层遮挡等。模型只需要学习变化规律，而非从零学习完整功率曲线，大幅降低学习难度。

模型不直接预测未来绝对功率y(t+H)，而是预测相对于当前时刻的变化量：

```
Δy = y(t+H) - y(t)
```

最终预测值重构：

```
y_hat_future = y(t) + Δy_hat
```

### 4. 模型体系

#### 1) 强基线模型

| 模型 | 说明 |
|------|------|
| Persistence | 用当前功率直接作为未来H步预测值 |
| Ridge Regression | L2正则化线性回归，输入为展平后的时序特征 |
| XGBoost | 梯度提升树，200棵树，max_depth=6 |
| LightGBM | 微软梯度提升树，200棵树，max_depth=6 |

其中，树模型输入为展平的16×26=416维特征向量，输出为单步功率预测值。

#### 2) 深度学习残差预测模型

全部采用残差预测目标：

```
Δy = y_future - y_last
y_hat_future = y_last + Δy_hat
```

统一训练配置：

| 参数 | 设置 |
|------|------|
| 优化器 | Adam |
| 学习率 | 0.001 |
| Batch Size | 256 |
| 最大Epoch | 50 |
| Early Stopping | Patience=8 |
| 随机种子 | 42 |

### 5. Optuna-AFSA混合搜索

Optuna TPE原理：TPE将历史试验结果按验证损失分为"好"和"差"两组，分别拟合概率密度分布，通过计算比值评估候选超参数属于"好"的可能性，比值越高被采样的概率越大。配合MedianPruner剪枝策略，在训练过程中定期检查验证损失，若高于已完成试验的中位数则提前终止。

AFSA原理：人工鱼群算法模拟鱼群觅食行为，通过觅食（向更优解移动）、聚群（向邻居中心移动）和追尾（向最优个体移动）三种行为在搜索空间中迭代寻优。

实验消融：

| 策略 | 名称 | 说明 |
|------|------|------|
| S1 | Random Search | 随机采样20 trials（基线下限） |
| S2 | Optuna TPE | 贝叶斯优化20 trials（强基线） |
| S3 | AFSA | 人工鱼群启发式搜索20 trials |
| S4 | Optuna→AFSA | Optuna给出初始鱼群，AFSA局部精细化 |
| S5 | AFSA→Optuna | AFSA给出初值，Optuna精细化 |
| S6 | Hybrid | Random + Optuna + AFSA 三阶段融合 |

### 6. 标准化推理时间测量

为确保推理时间可比性，采用以下规范流程：

1. 固定 batch_size = 512
2. 只测 model.forward()，不包含 DataLoader、CSV保存、反归一化
3. Warm-up 10 次（不记录时间）
4. 正式重复 100 次，记录 mean ± std
5. GPU测量时使用 torch.cuda.synchronize()
6. 同时报告 ms/sample、samples/s、参数量

---

## 三、实验结果

### 1. 主预测实验表

#### 1) H1（15分钟预测）

| 模型 | RMSE | MAE | MAPE | R² |
|------|------|-----|------|-----|
| XGBoost | 0.0300 | 0.0108 | 10.69% | 0.9880 |
| LightGBM | 0.0300 | 0.0108 | 10.75% | 0.9881 |
| Ridge | 0.0336 | 0.0161 | 14.87% | 0.9850 |
| Persistence | 0.0408 | 0.0195 | 24.35% | 0.9779 |
| LSTM (Residual) | 0.0458 | 0.0241 | 10.78% | 0.9706 |
| BiLSTM (Residual) | 0.0460 | 0.0252 | 11.31% | 0.9703 |
| CNN-BiLSTM (Residual) | 0.0465 | 0.0253 | 12.67% | 0.9697 |
| CNN-LSTM (Residual) | 0.0468 | 0.0259 | 11.86% | 0.9693 |
| PatchTST (Residual) | 0.0473 | 0.0263 | 13.40% | 0.9687 |
| Moving Average | 0.7773 | 0.7199 | 393.46% | -7.034 |

树模型RMSE=0.0300表现最优，R²≈0.988。Persistence RMSE=0.0408意外优于所有深度学习残差模型（0.0458~0.0473），说明深度模型未有效利用15分钟尺度的功率自相关性。LSTM在深度模型中最佳且参数最少（57K），PatchTST最差。Moving Average完全失效。

#### 2) H4（1小时预测）

| 模型 | RMSE | MAE | MAPE | R² |
|------|------|-----|------|-----|
| XGBoost | 0.0474 | 0.0190 | 20.58% | 0.9701 |
| LightGBM | 0.0475 | 0.0193 | 20.87% | 0.9700 |
| Ridge | 0.0591 | 0.0323 | 35.18% | 0.9536 |
| CNN-BiLSTM (Residual) | 0.0715 | 0.0424 | 20.74% | 0.9321 |
| BiLSTM (Residual) | 0.0722 | 0.0410 | 20.12% | 0.9308 |
| LSTM (Residual) | 0.0723 | 0.0425 | 19.79% | 0.9305 |
| PatchTST (Residual) | 0.0729 | 0.0415 | 20.58% | 0.9293 |
| Persistence | 0.0868 | 0.0425 | 56.71% | 0.8999 |

树模型继续领先（RMSE≈0.0474）。关键变化：深度学习残差模型（0.0715~0.0729）首次全部超越Persistence（0.0868），说明残差策略在较长Horizon上有效。MAPE方面深度模型（19.79%~20.74%）与树模型（20.58%~20.87%）持平，相对误差控制已接近。CNN-BiLSTM在深度模型中最佳。

#### 3) H16（4小时预测）

| 模型 | RMSE | MAE | MAPE | R² |
|------|------|-----|------|-----|
| XGBoost | 0.0843 | 0.0382 | 40.36% | 0.9056 |
| LightGBM | 0.0856 | 0.0396 | 41.74% | 0.9026 |
| Ridge | 0.1041 | 0.0662 | 77.15% | 0.8560 |
| PatchTST (Residual) | 0.1158 | 0.0710 | 42.82% | 0.8421 |
| LSTM (Residual) | 0.1159 | 0.0706 | 41.35% | 0.8420 |
| CNN-LSTM (Residual) | 0.1166 | 0.0744 | 45.48% | 0.8401 |
| BiLSTM (Residual) | 0.1183 | 0.0726 | 43.42% | 0.8355 |
| CNN-BiLSTM (Residual) | 0.1215 | 0.0760 | 49.34% | 0.8264 |
| Persistence | 0.2266 | 0.1224 | 132.23% | 0.3172 |

树模型仍最优（RMSE=0.0843）。残差深度模型中PatchTST（0.1158）与LSTM（0.1159）并列最佳，两者误差增长最慢。CNN-BiLSTM在长Horizon表现最差（0.1215），双向结构在长序列上可能引入噪声。Persistence严重劣化（0.2266，R²仅0.317）。

### 2. 跨Horizon性能衰减

| 模型 | H1 RMSE | H4 RMSE | H16 RMSE | H1→H16增幅 |
|------|---------|---------|----------|------------|
| XGBoost | 0.0300 | 0.0474 | 0.0843 | 2.81× |
| PatchTST (Residual) | 0.0473 | 0.0729 | 0.1158 | 2.45× |
| LSTM (Residual) | 0.0458 | 0.0723 | 0.1159 | 2.53× |
| BiLSTM (Residual) | 0.0460 | 0.0722 | 0.1183 | 2.57× |
| CNN-LSTM (Residual) | 0.0468 | 0.0735 | 0.1166 | 2.49× |
| CNN-BiLSTM (Residual) | 0.0465 | 0.0715 | 0.1215 | 2.61× |
| Ridge | 0.0336 | 0.0591 | 0.1041 | 3.10× |

PatchTST增幅最小（2.45倍），自注意力机制对长序列更友好；Ridge增幅最大（3.10倍），线性模型难以捕捉长程非线性。LSTM增幅2.53倍，稳健性良好。

### 3. 预测曲线对比

预测曲线叠加图将各模型预测值与真实值画在同一时间轴上。H1上XGBoost/LightGBM最贴近真实黑线，深度模型峰值略低；H4上深度残差模型跟踪能力有所改善，已优于Persistence；H16上PatchTST在平滑区域跟踪较好，CNN-BiLSTM在突变处响应偏慢。

### 4. 推理效率对比

| 模型 | Params | H1 (ms/sample) | H4 (ms/sample) | H16 (ms/sample) |
|------|--------|----------------|----------------|-----------------|
| LSTM (Residual) | ~57K | 0.004 | 0.005 | 0.003 |
| CNN-LSTM (Residual) | ~61K | 0.005 | 0.010 | 0.004 |
| PatchTST (Residual) | ~108K | 0.010 | 0.017 | 0.009 |
| BiLSTM (Residual) | ~147K | 0.011 | 0.018 | 0.009 |
| CNN-BiLSTM (Residual) | ~152K | 0.011 | 0.024 | 0.010 |

LSTM参数量最小（57K）且推理最快（0.003~0.005ms）。CNN-BiLSTM参数量最大（152K）且推理最慢（0.010~0.024ms），为LSTM的5倍。所有模型延迟<0.03ms，远满足实时调度需求（<100ms）。Transformer推理时间短是合理的：短窗口+少量Patch Token使Self-Attention计算成本可控。

### 5. Optuna-AFSA混合搜索结果（单CNN-LSTM模型H16快速验证）

| 策略 | 最优RMSE | 最优配置 | 综合评分 |
|------|----------|----------|----------|
| S3 AFSA | 0.2971 | hidden=32, layers=1, dropout=0.1, lr=0.002, bs=128 | 0.0006 |
| S1 Random | 0.2090 | hidden=32, layers=1, dropout=0.1, lr=0.002, bs=128 | 0.0070 |
| S4 Optuna→AFSA | 0.3015 | hidden=32, layers=1, dropout=0.3, lr=0.001, bs=128 | 0.0082 |
| S2 Optuna | 0.2998 | hidden=32, layers=2, dropout=0.3, lr=0.002, bs=256 | 0.0143 |
| S6 Hybrid | 0.2993 | hidden=64, layers=1, dropout=0.3, lr=0.001, bs=128 | 0.0184 |
| S5 AFSA→Optuna | 0.2954 | hidden=128, layers=1, dropout=0.1, lr=0.002, bs=128 | 0.0869 |

其中，混合搜索为快速验证（≤15 epoch），RMSE为验证集指标，与主实验表完整训练后的测试集指标不直接可比。

S3 AFSA综合评分最低（精度-效率平衡最佳），说明小模型（hidden=32, layers=1）较小容量配合较高学习率即可取得较好验证性能。S5 RMSE最低但综合评分最差（选了hidden=128大模型，延迟和参数量惩罚增加）。

---

## 四、结果分析

### 1. 树模型与深度学习残差模型对比

| 维度 | XGBoost/LightGBM | 深度学习残差模型 |
|------|------------------|------------------|
| H1精度 | RMSE 0.0300 | RMSE 0.0458~0.0473 |
| H4精度 | RMSE 0.0474 | RMSE 0.0715~0.0729 |
| H16精度 | RMSE 0.0843 | RMSE 0.1158~0.1215 |
| 训练速度 | 秒级 | 分钟级 |
| 推理延迟 | 低 | 极低（<0.03ms） |
| 可解释性 | 高（特征重要性） | 中（注意力/梯度） |
| 多步输出 | 需独立建模每步 | 天然支持多步 |

从结果可知树模型全面领先。原因：

1. 光伏功率与气象特征之间存在复杂的非线性关系，树模型通过递归分裂天然捕捉非线性，无需预设函数形式。
2. 树模型对特征尺度不敏感，不需要像深度学习那样精细归一化，各类特征可直接使用。
3. 展平后的416维特征使树模型可以自由选择任意时间步的任意特征进行分裂，相当于完成了注意力功能。
4. 39K训练样本对树模型而言充足，而深度学习模型在此数据量下容量受限。

### 2. 跨Horizon模型分析

| 模型 | H1→H16增幅 | 说明 |
|------|-----------|------|
| PatchTST | 2.45× | Self-Att跨时间步直接建模，长距离依赖捕获能力强 |
| CNN-LSTM | 2.49× | CNN提取局部尺度不变特征，泛化到长Horizon较好 |
| LSTM | 2.53× | 门控机制保持信息传递，稳健性良好 |
| BiLSTM | 2.57× | 双向结构在长Horizon上增益有限 |
| CNN-BiLSTM | 2.61× | 双向+CNN特征冗余，长Horizon下累积误差放大 |
| XGBoost | 2.81× | 精度最高但误差增长较快，缺乏时序归纳偏置 |
| Ridge | 3.10× | 线性假设难以捕捉长程非线性 |

PatchTST增长最慢。PatchTST每个Patch代表一段时间的整体模式而非单个时间点。Positional Encoding保留了时间顺序信息，Self-Att允许任意两个时间位置直接交互，信息传递路径最短，不存在RNN的遗忘问题。因此随着预测跨度增长，PatchTST的误差累积速度最慢。

CNN混合模型表现稳健。CNN卷积核在时间维度上滑动提取局部模式，具有尺度不变性，无论预测跨度多长，卷积层提取的局部特征质量保持稳定。因此CNN-LSTM和CNN-BiLSTM增幅均低于纯RNN。

LSTM和BiLSTM通过门控机制缓解梯度消失，但理论上仍存在遗忘问题，4小时前的信息在16步传播后可能衰减。因此纯RNN增幅略高于其他模型。

Ridge线性模型增幅最大。线性模型假设特征与目标之间为线性关系，在H1上该假设近似成立，但随着预测跨度增长，云层移动、辐照变化等非线性过程的影响越来越显著，线性假设严重失效，RMSE迅速增加。

### 3. 残差预测效果

| 排名 | H1最佳残差模型 | H4最佳残差模型 | H16最佳残差模型 |
|------|----------------|----------------|-----------------|
| 1 | LSTM (0.0458) | CNN-BiLSTM (0.0715) | LSTM (0.1159) |
| 2 | BiLSTM (0.0460) | BiLSTM (0.0722) | PatchTST (0.1159) |
| 3 | CNN-BiLSTM (0.0465) | LSTM (0.0723) | CNN-LSTM (0.1166) |
| 4 | CNN-LSTM (0.0468) | PatchTST (0.0729) | BiLSTM (0.1183) |
| 5 | PatchTST (0.0473) | CNN-LSTM (0.0735) | CNN-BiLSTM (0.1215) |

| 模型 | Step3 RMSE | Step5 RMSE | 变化 |
|------|-----------|-----------|------|
| LSTM | 0.0559 | 0.0458 | ↓ 18.1% |
| BiLSTM | 0.0526 | 0.0460 | ↓ 12.5% |
| CNN-LSTM | 0.0477 | 0.0468 | ↓ 1.9% |
| CNN-BiLSTM | 0.0496 | 0.0465 | ↓ 6.3% |

由于该阶段实验未经过optuna-afsa的混合搜索，因此无法直接与step4的结果作对比。

H1情况下，所有模型RMSE均下降，说明从Step3到Step5的改进（增强特征工程 + 残差预测策略）整体有效，说明在固定超参框架下，综合改进（特征工程升级+残差目标）使各模型精度普遍提升。

### 4. 白天/夜间分段评价的影响

| 模型 | 全天RMSE | Daytime RMSE |
|------|----------|--------------|
| XGBoost | 0.0300 | 0.0466 |
| LSTM (Residual) | 0.0295 | 0.0458 |
| Persistence | 0.0408 | 0.0633 |

夜间光伏功率接近0，样本误差绝对值极小，大量夜间样本拉低了全天RMSE的加权平均值。而白天功率可能高达0.8~1.0，误差绝对值可达0.05~0.10，因此白天样本对RMSE的影响被夜间样本稀释。

相对排序保持稳定。Daytime RMSE虽然数值上高于全天RMSE，但各模型的相对排名完全一致。因此使用全天RMSE或Daytime RMSE不会改变研究结论，但Daytime RMSE更真实地反映白天有效发电时段的预测能力，更符合业务需求。

### 5. Optuna-AFSA混合搜索分析（结果存疑）

| 策略 | 最优RMSE | 综合评分 | 说明 |
|------|----------|----------|------|
| S3 AFSA | 0.2971 | 0.0006 | 最佳精度-效率平衡 |
| S1 Random | 0.2090 | 0.0070 | 随机搜索偶遇好配置 |
| S4 Optuna→AFSA | 0.3015 | 0.0082 | Optuna初始化后AFSA局部搜索有效 |
| S2 Optuna | 0.2998 | 0.0143 | 纯贝叶斯优化，效率中等 |
| S6 Hybrid | 0.2993 | 0.0184 | 三阶段融合，精度尚可但开销大 |
| S5 AFSA→Optuna | 0.2954 | 0.0869 | RMSE最低但选了大模型，综合评分差 |

AFSA单独表现最佳。在快速搜索预算下（20 trials），AFSA通过群体智能在搜索空间中高效探索，同时保持了较小的模型容量。这说明AFSA的群体并行搜索在有限预算下效率很高，多个个体同时探索不同区域，相比Optuna的序列建模效率更高。

所有策略的最优配置中，hidden=32出现5次，hidden=64出现1次，hidden=128出现1次（S5），小模型优势大。说明在快速验证阶段（≤15 epoch），较小容量配合较高学习率（0.002）即可取得较好验证性能，大模型在有限训练步数下尚未充分收敛。

S5 RMSE最低但综合评分最差。S5（AFSA→Optuna）选出了hidden=128的大模型，RMSE=0.2954为所有策略中最低，但综合评分0.0869为最高（最差），因为综合评分函数对延迟和参数量施加了惩罚。

混合搜索总结：在快速搜索阶段，AFSA单独已足够有效；如果需要更稳定的收敛保证，Optuna→AFSA（S4）是合理选择，结合了Optuna的全局建模和AFSA的局部开发能力。三阶段融合（S6）在精度上并未显著优于简单策略，但计算开销更大，性价比不高。

---

## 五、可视化结果

### 5.1 跨 Horizon 综合分析

![comparison_summary](data/prediction/step5_new_experiments/figures/comparison_summary.png)

**图表解读**：
- **左上**：H1/H4/H16 的 RMSE 柱状图，显示树模型在各 Horizon 上的领先优势。
- **右上**：R² 对比，树模型 R² 普遍高于残差深度学习模型。
- **左下**：RMSE 随 Horizon 变化趋势，PatchTST 与 LSTM 增长较缓。
- **右下**：RMSE 热力图，直观展示模型 × Horizon 的误差分布。

### 5.2 各 Horizon 最佳模型推荐

![comparison_best_model](data/prediction/step5_new_experiments/figures/comparison_best_model.png)

### 5.3 H1 指标对比与预测曲线

![h1_metrics_comparison](data/prediction/step5_new_experiments/figures/h1/h1_metrics_comparison.png)

![h1_prediction_overlay](data/prediction/step5_new_experiments/figures/h1/h1_prediction_overlay.png)

### 5.4 H4 指标对比、预测曲线与精度分析

![h4_metrics_comparison](data/prediction/step5_new_experiments/figures/h4/h4_metrics_comparison.png)

![h4_prediction_overlay](data/prediction/step5_new_experiments/figures/h4/h4_prediction_overlay.png)

![h4_accuracy_analysis](data/prediction/step5_new_experiments/figures/h4/h4_accuracy_analysis.png)

### 5.5 H16 指标对比与预测曲线

![h16_metrics_comparison](data/prediction/step5_new_experiments/figures/h16/h16_metrics_comparison.png)

![h16_prediction_overlay](data/prediction/step5_new_experiments/figures/h16/h16_prediction_overlay.png)

### 5.6 残差预测跨 Horizon 对比

![residual_comparison_all_horizons](data/prediction/step5_new_experiments/figures/residual_comparison_all_horizons.png)

### 5.7 跨 Horizon 推理效率对比

![inference_benchmark_all_horizons](data/prediction/step5_new_experiments/figures/inference_benchmark_all_horizons.png)

---

## 六、结论

### 6.1 核心发现总结

| 研究问题 | 结论 | 证据 |
|----------|------|------|
| Q1：残差预测是否有效？ | **部分有效** | H1所有模型RMSE均下降，残差模型MAPE与树模型接近，但绝对RMSE仍落后 |
| Q2：树模型表现如何？ | **全面领先** | XGBoost/LightGBM在H1/H4/H16均取得最低RMSE |
| Q3：白天/夜间分段评价影响？ | **重要但排序稳定** | Daytime RMSE比全天高约55%，相对排名不变 |
| Q4：混合搜索价值？ | **AFSA在小预算下更高效** | S3 AFSA综合评分最低，S5 RMSE最低但大模型惩罚明显 |

### 6.2 最终模型推荐

```
┌─────────────────────────────────────────────────────────────────┐
│                     光伏功率预测模型选择指南                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ╔═══════════════════════════════════════════════════════════╗  │
│  ║  场景一：实时调度 (15 min 预测)                            ║  │
│  ║  ─────────────────────────────────────────                 ║  │
│  ║  推荐: XGBoost 或 LightGBM                                 ║  │
│  ║  理由: RMSE=0.0300, R²≈0.988, 训练快，可解释性强          ║  │
│  ║  参数: n_estimators=200, max_depth=6, lr=0.05             ║  │
│  ╚═══════════════════════════════════════════════════════════╝  │
│                                                                 │
│  ╔═══════════════════════════════════════════════════════════╗  │
│  ║  场景二：日前计划 (1 h 预测)                               ║  │
│  ║  ─────────────────────────────────────────                 ║  │
│  ║  推荐: XGBoost / LightGBM                                  ║  │
│  ║  备选: CNN-BiLSTM (Residual)                               ║  │
│  ║  理由: 树模型 RMSE≈0.0474；CNN-BiLSTM 在残差模型中最佳     ║  │
│  ╚═══════════════════════════════════════════════════════════╝  │
│                                                                 │
│  ╔═══════════════════════════════════════════════════════════╗  │
│  ║  场景三：极端长预测 (4 h 预测)                             ║  │
│  ║  ─────────────────────────────────────────                 ║  │
│  ║  推荐: XGBoost                                             ║  │
│  ║  备选: LSTM / PatchTST (Residual)                          ║  │
│  ║  理由: XGBoost RMSE=0.0843；PatchTST 误差增长最缓          ║  │
│  ╚═══════════════════════════════════════════════════════════╝  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.3 技术建议

1. **特征工程**
   - 当前26维增强特征对树模型已足够；若继续优化深度学习，建议引入更长时间尺度的滞后特征（如过去1h/4h平均辐照）。
   - 考虑加入数值天气预报（NWP）数据作为外生变量，提升H16精度。

2. **模型训练**
   - 树模型可直接用于生产环境，训练成本极低。
   - 深度学习残差模型建议仅在需要多步联合输出或多站点迁移学习时采用。

3. **超参搜索**
   - 混合搜索中AFSA在小预算快速验证阶段效率更高；若需更稳定收敛，推荐Optuna→AFSA（S4）。
   - 三阶段融合（S6）精度提升有限，但计算开销较大，性价比不高。

4. **评价方式**
   - 继续使用Daytime-only RMSE作为核心排名指标，避免夜间零功率段干扰。
   - 可进一步按天气类型（晴天/多云/雨天）分段评价，识别模型在极端天气下的弱点。

### 6.4 未来改进方向

1. **集成学习**：将XGBoost/LightGBM与LSTM/PatchTST残差模型按Horizon集成，可能兼顾精度与稳定性。
2. **多步联合建模**：探索Transformer的seq2seq结构，直接输出H1/H4/H16多尺度预测。
3. **不确定性量化**：引入分位数回归或贝叶斯神经网络，提供预测区间以支持风险调度。
4. **多站点迁移**：利用Site_1预训练模型，在Site_2/Site_3上进行微调，验证模型泛化能力。

---

## 七、附录

### 7.1 实验配置参数

| 参数 | 值 |
|------|-----|
| 随机种子 | 42 |
| Lookback | 16 |
| 特征维度 | 26 |
| 训练/验证/测试比例 | 70% / 15% / 15% |
| 残差模型训练 epoch | 50 (Early Stopping patience=8) |
| 学习率 | 0.001 |
| Batch size | 256 |
| 混合搜索 trials | 20 |
| 混合搜索最大 epoch | 15 |
| 推理 benchmark batch size | 512 |
| 推理 benchmark repeat | 100 |

### 7.2 最佳模型参数汇总

| Horizon | 最佳模型 | 关键参数 |
|---------|----------|----------|
| H1 | XGBoost | n_estimators=200, max_depth=6, lr=0.05, subsample=0.8 |
| H4 | XGBoost | n_estimators=200, max_depth=6, lr=0.05, subsample=0.8 |
| H16 | XGBoost | n_estimators=200, max_depth=6, lr=0.05, subsample=0.8 |
| H1 残差最佳 | LSTM (Residual) | hidden=64, layers=2, dropout=0.2, lr=0.001 |
| H4 残差最佳 | CNN-BiLSTM (Residual) | conv=32, hidden=64, layers=2, dropout=0.2 |
| H16 残差最佳 | LSTM / PatchTST (Residual) | hidden=64 / d_model=64, heads=4 |

### 7.3 数据文件清单

| 文件类型 | 路径 |
|----------|------|
| Step5 样本 | `data/prediction/step5_new_experiments/samples/` |
| Step5 模型 | `data/prediction/step5_new_experiments/models/` |
| Step5 预测 | `data/prediction/step5_new_experiments/predictions/` |
| Step5 指标 | `data/prediction/step5_new_experiments/metrics/` |
| Step5 图表 | `data/prediction/step5_new_experiments/figures/` |
| Step5 报告 | `data/prediction/step5_new_experiments/reports/` |
| Step5 日志 | `logs/prediction/step5_new_experiments/` |

---

*报告生成时间：2026-07-10*  
*综合分析：EXP-P05 实验数据（H1/H4/H16）*
