# 工程结构说明

## 项目定位

光伏功率**预测**与**调度协同**研究工程。当前阶段在 `prediction/` 下完成数据清洗与时间对齐（EXP-P01）。

## 目录树

```
pv_prediction_scheduling_msc/
├── README.md
├── docs/
│   └── PROJECT_STRUCTURE.md          # 本文件
├── data/
│   ├── raw/                          # 原始 CSV（只读）
│   └── prediction/
│       └── step1_preprocessing/      # EXP-P01 产出
│           ├── OUTPUT_FILE_COMMENTS.md
│           └── processed/
├── experiments/
│   ├── README.md
│   ├── prediction/
│   │   └── step1_data_cleaning_alignment/
│   │       ├── README.md
│   │       └── run_exp_p01_preprocessing.py
│   └── scheduling/                   # 预留
├── logs/                             # 实验日志专夹
│   ├── README.md
│   └── prediction/step1_data_cleaning_alignment/
│       └── EXP-P01.log
```

## 约定

| 约定 | 说明 |
|------|------|
| 一实验一脚本 | 每实验目录仅一个 `run_exp_*.py` 入口 |
| 原始数据不动 | 只读 `data/raw/` |
| 结果注释 | 脚本文件头 + `OUTPUT_FILE_COMMENTS.md` |
| 日志专夹 | 仅 `logs/` 存放运行日志 |

## 实验编号

| 编号 | 名称 |
|------|------|------|
| EXP-P01 | 数据清洗与时间对齐 |


## 运行依赖

```
data/raw/*.csv  →  EXP-P01  →  data/prediction/step1_preprocessing/processed/
                              →  logs/prediction/.../EXP-P01.log
```
