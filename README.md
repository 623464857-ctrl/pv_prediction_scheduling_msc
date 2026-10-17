# 光伏发电功率预测与系统调度协同优化

面向高速公路服务区光储充微电网 (亳州站), 两层框架:

```
预测层: 历史功率 + 天气 ──CNN-BiLSTM──> 未来 15 分钟光伏功率
                                              │
调度层: 光伏预测 + 充电负荷 + 电价 ──MILP 滚动优化──> 储能充放电 / 充电桩调度 / 购售电计划
                                              │
实际结算: 用真实光伏执行计划, 计算成本和偏差 (光伏高估罚得重, 低估罚得轻)
```

## 目录结构

```
experiments/
├── common.py                    所有路径和公共常量 (改目录只改这里)
├── prediction/                  预测层, 按 p1 → p6 顺序运行
│   ├── p1_preprocess.py         原始功率+天气 → Bozhou_1_preprocessed.csv
│   ├── p2_quality_check.py      数据质量检查 → Bozhou_1_cleaned.csv
│   ├── p3_prepare_samples.py    构造 24 个特征 + 滑动窗口 + 标准化 → samples/
│   ├── p4_search_hparams.py     超参数搜索 (Optuna-TPE + 鱼群算法 AFSA) → params/
│   ├── p5_train.py              训练 → models/
│   ├── p6_evaluate.py           验证集/测试集预测、指标、曲线图 → predictions/, figures/
│   ├── models.py                CNN-BiLSTM 模型
│   ├── afsa.py                  人工鱼群算法
│   └── torch_utils.py           训练循环 (早停, 进度条)
└── scheduling/                  调度层, 按 s1 → s2 顺序运行
    ├── config.py                电价、储能、充电桩、偏差考核等全部参数
    ├── s1_prepare_data.py       光储充运行数据 (5 分钟 xlsx) → schedule_15min.csv
    ├── s2_run_dispatch.py       逐日滚动调度 + 实际结算 + 各策略对比
    ├── dispatch_model.py        调度优化模型 (开题报告式 1-5 ~ 1-14)
    ├── strategies.py            6 种对比策略, 非对称风险修正
    └── settlement.py            按实际光伏执行计划并结算费用

data/
├── raw/亳州/                     原始光伏功率 (15 min) 和 Weatherbit 天气
├── schedule_data/               光储充运行数据 (5 min, 2025-11 ~ 2026-08)
├── processed/                   清洗后的数据 (预测层和调度层的输入)
├── data-analysis/               数据分析图表
└── data-get/fetch_bozhou.py     天气数据下载脚本

result/
├── prediction/                  样本、超参数、模型、预测结果、图
└── scheduling/                  逐时段调度结果、汇总表、图

logs/                            每个脚本的运行日志
_archive/                        已不使用的旧脚本和旧结果
```

## 运行

```bash
# 预测层 (H=1, 提前 15 分钟)
python -m experiments.prediction.p3_prepare_samples
python -m experiments.prediction.p4_search_hparams          # 可选, 约 1 小时; 目前未运行, 使用默认超参数
python -m experiments.prediction.p5_train --seeds 42 43 44
python -m experiments.prediction.p6_evaluate --seeds 42 43 44
python -m experiments.prediction.models                     # 打印每一层 tensor 形状

# 调度层
python -m experiments.scheduling.s1_prepare_data
python -m experiments.scheduling.s2_run_dispatch            # 约 10 分钟
```

p1、p2 一般不需要重跑: 用当前的原始天气文件 (重新下载过, 坐标已更新) 重跑会得到与现有
`Bozhou_1_cleaned.csv` 不同的结果, 现有实验都基于现有的 cleaned 文件。

## 预测层

详细说明 (每个脚本的输入、输出、逻辑, 当前结果): [docs/预测部分说明.md](docs/预测部分说明.md)

- **输入**: 过去 16 个时段 (4 小时) × 24 个特征, 包括功率滞后/爬坡/滚动统计、GHI、温度、湿度、云量、能见度、时间编码;
  按训练集均值/标准差标准化
- **输出**: 下一个 15 分钟的 `power_pu` (= kW / 220)
- **模型**: Conv1d×2 → 双向 LSTM → Linear
- **划分**: 按时间 70% / 15% / 15%, 测试集为 2026-08-04 10:00 ~ 08-31
- **基准**: 持续性预测 (下一时刻 = 当前时刻), 测试集 RMSE 0.0911

## 调度层

详细说明 (每个脚本的输入、输出、逻辑, 当前结果): [docs/调度部分说明.md](docs/调度部分说明.md)

- **目标**: 总运行成本最小 = 购电费 − 售电收入 + 储能运行成本 + 弃光惩罚 + 柔性负荷补偿
- **约束**: 功率平衡、光伏出力上限、储能充放电互斥与 SOC、充电负荷最多推迟 2 小时、联络线容量、购售电互斥
- **求解**: pulp + CBC, 每 15 分钟滚动求解一次 [当前, 当天结束]
- **非对称风险修正**: 光伏高估 (缺电) 的考核比低估重, 按报童模型取预测误差分布的偏低分位数做计划
- **参数**: 电价时段按该站储能实际的充放电规律推断, 价格水平为假设值, 见 `experiments/scheduling/config.py`,
  请替换为电费单上的实际电价

## 环境

Python 3.11, PyTorch, scikit-learn, pandas, optuna, pulp, matplotlib, tqdm。
训练请用 CPU 或 CUDA, 不要用 Apple GPU (MPS): 实测在 MPS 上训练该模型不收敛。
