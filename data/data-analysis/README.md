# 明月湖数据集 - 数据分析报告

## 目录结构

```
data-analysis/
|-- README.md                          # 本文件，说明文档
|-- scripts/
|   |-- data_quality_check.py          # 数据质量诊断脚本（v2）
|-- reports/
|   |-- data_quality_report.txt        # 数据质量诊断报告（文本版）
|-- charts/
|   |-- (待生成：可视化图表)
|-- data/
    |-- 明月湖_merged_raw.csv          # 合并后的原始数据集
```

## 数据基本信息

| 项目 | 说明 |
|------|------|
| 原始数据 | 明月湖光伏发电.csv + 明月湖6-8月天气数据.csv |
| 时间范围 | 2026-06-09 ~ 2026-08-31 |
| 分辨率 | 15 分钟 |
| 总行数 | 8064 行 |
| 装机容量 | 281.6 kW |

## 合并后字段列表

| 字段名 | 说明 |
|--------|------|
| `timestamp` | 时间戳 |
| `power_kw` | 光伏发电功率 (kW) |
| `temperature_c` | 温度 (°C) |
| `apparent_temperature_c` | 体感温度 (°C) |
| `relative_humidity_pct` | 相对湿度 (%) |
| `dew_point_c` | 露点温度 (°C) |
| `pressure_hpa` | 气压 (hPa) |
| `wind_speed_ms` | 风速 (m/s) |
| `wind_dir_deg` | 风向 (°) |
| `wind_gust_ms` | 阵风风速 (m/s) |
| `visibility_km` | 能见度 (km) |
| `cloud_cover_pct` | 云量 (%) |
| `solar_altitude_deg` | 太阳高度角 (°) |
| `solar_azimuth_deg` | 太阳方位角 (°) |
| `ghi` | 水平面总辐射 (W/m²) |
| `dhi` | 水平面散射辐射 (W/m²) |
| `solar_radiation_wm2` | 太阳辐射强度 (W/m²) |
| `uv` | 紫外线指数 |
| `precip_rate_mmhr` | 降水率 (mm/h) |
| `snow_rate_mmhr` | 降雪率 (mm/h) |
| `part_of_day` | 时段（n=夜间，d=白天） |
| `weather_code` | 天气代码 |
| `weather_description` | 天气描述 |
| `weather_icon` | 天气图标 |
| `hour_of_day` | 小时（0-23） |

## 数据问题汇总

1. **夜间功率>0（353条）**：最大1.73 kW，建议Clip为0
2. **DHI>GHI（49条，0.6%）**：超出量极小（最大6 W/m²），可忽略或Clip

## 依赖

```
pandas
numpy
```

运行方式：
```bash
cd scripts
python data_quality_check.py
```
