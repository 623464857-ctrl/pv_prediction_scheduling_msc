# EXP-P01：数据清洗与时间对齐

## 实验目的

将 `data/raw/` 下 8 个光伏站点原始 CSV 转为结构统一、15 分钟对齐、异常可控、缺失可补、质量可追溯的数据集，支撑超短期功率预测。

## 运行

在项目根目录执行：

```powershell
python experiments/prediction/step1_data_cleaning_alignment/run_exp_p01_preprocessing.py
```

## 输入 / 输出

| 类型 | 路径 |
|------|------|
| 输入 | `data/raw/*.csv` |
| 输出 | `data/prediction/step1_preprocessing/processed/` |
| 日志 | `logs/prediction/step1_data_cleaning_alignment/EXP-P01.log` |

## 流程概要

字段统一 → 15min 时间轴 → 物理清洗 → Hampel → 短时插值 → 夜间置零 → 剖面回填 → 鲁棒标准化 → 衍生特征 → 质量评分 → 多站面板
