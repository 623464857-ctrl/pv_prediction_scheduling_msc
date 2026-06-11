# 光伏预测与调度协同研究工程

## 当前进度

| 实验 | 名称 | 脚本 | 状态 |
|------|------|------|------|
| EXP-P01 | 数据清洗与时间对齐 | `experiments/prediction/step1_data_cleaning_alignment/run_exp_p01_preprocessing.py` | 已完成 |
| EXP-P02 | 五类基础模型对比 | `experiments/prediction/step2_baseline_models/` | 已完成 |

## 运行 EXP-P01

```powershell
python experiments/prediction/step1_data_cleaning_alignment/run_exp_p01_preprocessing.py
```

**输入**：`data/raw/*.csv`  
**输出**：`data/prediction/step1_preprocessing/processed/`  
**日志**：`logs/prediction/step1_data_cleaning_alignment/EXP-P01.log`

## 目录结构

详见 [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)。

产出字段说明见 [data/prediction/step1_preprocessing/OUTPUT_FILE_COMMENTS.md](data/prediction/step1_preprocessing/OUTPUT_FILE_COMMENTS.md)。

## EXP-P02 关键结果

- **BiLSTM** 测试集 RMSE=0.0465、R²=0.971，为五类模型最优
- LSTM 次之；BP 优于 SVR / Random Forest
- 详见 [EXP-P02 初步结论](data/prediction/step2_baseline_models/reports/EXP-P02_preliminary_conclusion.md)

## EXP-P01 关键结果

- 处理 **8** 个站点，15 分钟统一时间轴
- **Site 8** 内部补齐 **768** 个时间缺口（`row_inserted_by_reindex`）
- 多站调度面板共同窗口截止 **2020-07-01 23:45:00**（对齐 Site 3）
- 修复强度最高站点：**Site_3**（37,242 修复单元）
