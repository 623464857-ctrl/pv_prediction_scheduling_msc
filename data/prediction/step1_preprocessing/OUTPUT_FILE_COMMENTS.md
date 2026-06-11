# EXP-P01 产出文件说明

本目录为实验 **EXP-P01（数据清洗与时间对齐）** 的运行结果。CSV 首行保持标准表头以便程序读取；字段与实验目的说明统一维护于本文档。

---

## `processed/stations/Site_{n}_preprocessed.csv`

**实验目的**：单站完整预处理长表，含清洗后特征、质量标记与鲁棒标准化列。

**主要字段**：
| 类别 | 字段示例 |
|------|----------|
| 时间 | `timestamp`（15 分钟规则网格） |
| 气象/功率 | `total_irradiance_wm2`, `power_mw`, `power_pu` 等 |
| 元数据 | `site_id`, `site_key`, `capacity_mw`, `source_file` |
| 时间轴 | `row_inserted_by_reindex`, `source_observed_flag` |
| 清洗标记 | `*_raw_missing_flag`, `*_invalid_flag`, `*_outlier_flag`, `*_imputed_flag` |
| 功率专用 | `power_mw_negative_clipped_flag`, `power_mw_invalid_flag` |
| 衍生 | `power_ramp_15m_mw`, `daylight_flag`, `sin_hour`, `cos_hour` 等 |
| 质量 | `imputed_feature_count`, `raw_issue_count`, `data_quality_score` |
| 标准化 | `*_robust_scaled` |

---

## `processed/solar_stations_long.csv`

**实验目的**：8 站预处理结果纵向合并，供多站联合分析与建模。

**说明**：每行含 `site_key`，字段结构与单站文件一致。

---

## `processed/solar_site_quality_summary.csv`

**实验目的**：站点级预处理质量汇总，用于横向比较各站数据完整性及修复强度。

**关键列**：
- `row_inserted_count`：时间轴重建插入行数（Site 8 = 768）
- `mean_data_quality_score` / `min_data_quality_score`
- `issue_repair_cell_count`：异常/缺失修复涉及单元累计数
- `time_start` / `time_end`：站点有效时间范围

---

## `processed/solar_feature_scaling_reference.csv`

**实验目的**：记录各站鲁棒标准化参数 `(x - median) / scale`，便于模型部署与反归一化。

**列**：`site_id`, `feature`, `median`, `scale`, `scale_method`（`iqr` / `std` / `unit`）

---

## `processed/solar_dispatch_panel_common_window.csv`

**实验目的**：多站共同时间窗口（截止 **2020-07-01 23:45:00**，对齐 Site 3）宽表面板，支撑调度协同分析。

**关键列**：
- `power_mw_Site_*`, `power_pu_Site_*`, `data_quality_score_Site_*`
- `fleet_power_mw`, `fleet_power_pu_mean`, `fleet_quality_score_mean`, `available_site_count`

**行数**：52,608（与 Site 3 时间覆盖一致）
