# Site_1_optimized.csv 完整特征解读文档

## 文件概述

| 属性 | 值 |
|------|-----|
| 行数 | 70,176 |
| 列数 | 140 |
| 时间范围 | 2019-01-01 ~ 2020-12-31 |
| 时间分辨率 | 15分钟 |
| 站点容量 | 50 MW |

---

## 特征分类总览

```
📊 原始测量特征 (8列)
├── 时间戳 & 元信息 (5列)
├── 辐照度特征 (3列)
├── 气象特征 (2列)
└── 功率特征 (1列)

📋 质量标志特征 (28列)
├── 原始缺失标志 (7列)
├── 无效值标志 (7列)
├── 异常值标志 (7列)
└── 插值标志 (7列)

🕐 时间特征 (6列)

📐 归一化特征 (14列)

⚠️ 异常检测特征 (14列)

🏷️ 质量评估特征 (14列)

🔧 增强特征 - 辐照质量 (5列)

🌡️ 增强特征 - 温度修正 (10列)

⚡ 增强特征 - 效率分析 (11列)

⏰ 增强特征 - 时段交互 (10列)
```

---

## 一、原始测量特征 (Raw Measurements)

### 1.1 时间戳 & 元信息

| 列名 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `timestamp` | str | 数据时间戳 | 2019-01-01 00:00:00 |
| `site_id` | str | 站点ID | Site_1 |
| `site_key` | str | 站点唯一标识 | site_1 |
| `capacity_mw` | float | 装机容量(MW) | 50.0 |
| `source_file` | str | 原始数据来源文件 | - |

### 1.2 辐照度特征 (Irradiance)

| 列名 | 单位 | 说明 | 统计值 |
|------|------|------|--------|
| `total_irradiance_wm2` | W/m² | **总平面辐照度 (GTI)** - 倾斜面接收的总辐照 | mean=266.5, max=1359 |
| `direct_normal_irradiance_wm2` | W/m² | **直射辐照度 (DNI)** - 垂直于太阳的直射光 | mean=85.6, max=980 |
| `global_horizontal_irradiance_wm2` | W/m² | **水平面总辐照度 (GHI)** - 水平面接收的总辐照 | mean=67.4, max=989 |

**物理关系**: `GHI = DNI × cos(天顶角) + DHI (散射)`

### 1.3 气象特征 (Meteorological)

| 列名 | 单位 | 说明 | 统计值 |
|------|------|------|--------|
| `air_temperature_c` | °C | **空气温度** | mean=13.2, range=[-18.2, 41.2] |
| `atmosphere_hpa` | hPa | **大气压** | 约1013 hPa |
| `relative_humidity_pct` | % | **相对湿度** | mean=21.4, range=[0, 69.9] |

### 1.4 功率特征 (Power)

| 列名 | 单位 | 说明 | 统计值 |
|------|------|------|--------|
| `power_mw` | MW | **实际发电功率** | mean=9.67, max=48.32 |
| `power_pu` | p.u. | **归一化功率** (0-1) | mean=0.19, max=0.97 |

---

## 二、质量标志特征 (Quality Flags)

### 2.1 原始缺失标志 (`_raw_missing_flag`)

| 列名 | 说明 |
|------|------|
| `total_irradiance_wm2_raw_missing_flag` | GTI原始数据是否缺失 |
| `direct_normal_irradiance_wm2_raw_missing_flag` | DNI原始数据是否缺失 |
| `global_horizontal_irradiance_wm2_raw_missing_flag` | GHI原始数据是否缺失 |
| `air_temperature_c_raw_missing_flag` | 温度数据是否缺失 |
| `atmosphere_hpa_raw_missing_flag` | 气压数据是否缺失 |
| `relative_humidity_pct_raw_missing_flag` | 湿度数据是否缺失 |
| `power_mw_raw_missing_flag` | 功率数据是否缺失 |

### 2.2 无效值标志 (`_invalid_flag`)

| 列名 | 说明 |
|------|------|
| `total_irradiance_wm2_invalid_flag` | GTI超出合理范围 |
| `direct_normal_irradiance_wm2_invalid_flag` | DNI超出合理范围 |
| `global_horizontal_irradiance_wm2_raw_missing_flag` | GHI超出合理范围 |
| `air_temperature_c_invalid_flag` | 温度超出合理范围 |
| `atmosphere_hpa_invalid_flag` | 气压超出合理范围 |
| `relative_humidity_pct_invalid_flag` | 湿度超出合理范围 |
| `power_mw_invalid_flag` | 功率超出合理范围 |

**额外**: `power_mw_negative_clipped_flag` - 功率负值被截断标志

### 2.3 异常值标志 (`_outlier_flag`)

使用统计方法（IQR/标准差）检测的异常值：
- `*_outlier_flag` (7列) - 各变量的统计异常

### 2.4 插值标志 (`_imputed_flag`)

缺失或异常值被插值填补后标记：
- `*_imputed_flag` (7列) - 各变量是否经过插值

---

## 三、时间特征 (Temporal Features)

| 列名 | 说明 | 取值范围 |
|------|------|----------|
| `month` | 月份 | 1-12 |
| `dayofyear` | 年内第几天 | 1-366 |
| `hour` | 小时 | 0-23 |
| `minute` | 分钟 | 0, 15, 30, 45 |
| `sin_hour` | 小时的正弦编码 | [-1, 1] |
| `cos_hour` | 小时的余弦编码 | [-1, 1] |
| `sin_dayofyear` | 年的正弦编码 | [-1, 1] |
| `cos_dayofyear` | 年的余弦编码 | [-1, 1] |

**注意**: 正弦/余弦编码用于捕捉周期性，优于直接用整数（如hour=23和hour=0在循环上相邻）

---

## 四、归一化特征 (Normalized Features)

后缀 `_robust_scaled` 表示使用**稳健归一化**：

```
robust_scaled = (x - median) / IQR
```

| 列名 | 说明 |
|------|------|
| `total_irradiance_wm2_robust_scaled` | GTI归一化 |
| `direct_normal_irradiance_wm2_robust_scaled` | DNI归一化 |
| `global_horizontal_irradiance_wm2_robust_scaled` | GHI归一化 |
| `air_temperature_c_robust_scaled` | 温度归一化 |
| `atmosphere_hpa_robust_scaled` | 气压归一化 |
| `relative_humidity_pct_robust_scaled` | 湿度归一化 |
| `power_mw_robust_scaled` | 功率归一化 |
| `power_pu_robust_scaled` | 功率标幺归一化 |
| `power_ramp_15m_mw_robust_scaled` | 15分钟功率变化归一化 |
| `power_ramp_15m_pu_robust_scaled` | 15分钟功率变化标幺归一化 |

---

## 五、异常检测特征 (Anomaly Detection)

### 5.1 场景异常标志

| 列名 | 说明 | 物理含义 |
|------|------|----------|
| `anomaly_clear_sky_no_power` | 晴天无功率异常 | 晴空但功率=0 |
| `anomaly_cloudy_high_power` | 多云高功率异常 | 云量高但功率反而高 |
| `anomaly_night_power` | 夜间功率异常 | 夜间功率>0 |
| `anomaly_severe_clear_no_power` | 严重晴天无功率 | 晴天高温但无功率 |

### 5.2 功率相关异常

| 列名 | 说明 |
|------|------|
| `power_diff` | 功率偏差（实际-理论）|
| `power_diff_pct` | 功率偏差百分比 |
| `power_sudden_change` | 功率突变幅度 |
| `power_sudden_change_suspicious` | 可疑功率突变 |
| `power_mw_corrected` | 校正后的功率 |
| `in_outage` | 是否处于停机状态 |

### 5.3 异常级别

| 列名 | 取值 | 说明 |
|------|------|------|
| `irradiance_power_anomaly_level` | 0-3 | 辐照-功率异常级别 |

- 0: 正常
- 1: 轻微异常
- 2: 严重异常
- 3: 明显错误

---

## 六、质量评估特征 (Quality Assessment)

### 6.1 质量层级

| 列名 | 说明 | 取值 |
|------|------|------|
| `humidity_quality_tier` | 湿度数据质量层级 | 1-3 |
| `irradiance_quality_tier` | 辐照度质量层级 | 1-3 |
| `quality_grade` | 综合质量等级 | A/B/C |

### 6.2 综合评分

| 列名 | 范围 | 说明 |
|------|------|------|
| `data_quality_score` | [0, 1] | 整体数据质量评分 |
| `overall_quality_score` | [0, 1] | 综合质量评分 |

### 6.3 辐照度相关质量

| 列名 | 说明 |
|------|------|
| `theoretical_max_gti` | 理论最大GTI |
| `gti_utilization` | GTI利用率 |
| `gti_volatility` | GTI波动性 |

### 6.4 计数特征

| 列名 | 说明 |
|------|------|
| `imputed_feature_count` | 被插值的特征数 |
| `raw_issue_count` | 原始问题数 |

---

## 七、功率变化特征 (Power Dynamics)

| 列名 | 单位 | 说明 |
|------|------|------|
| `power_ramp_15m_mw` | MW | 15分钟功率变化量 |
| `power_ramp_15m_pu` | p.u. | 15分钟功率变化率 |
| `daylight_flag` | 0/1 | 是否白天 |

---

## 八、增强特征 - 辐照质量 (Irradiance Validation)

这些特征用于**辐照度物理一致性检查**：

| 列名 | 说明 | 正常范围 |
|------|------|----------|
| `gti_change_rate` | GTI变化率 | - |
| `gti_smoothness` | GTI平滑度 | 0-1 |
| `dni_dhi_ratio` | DNI/DHI比例 | 0-10 |
| `dni_dhi_anomaly` | DNI/DHI异常标志 | 0/1 |
| `dni_gti_ratio` | DNI/GTI比例 | 0-1 |
| `irr_physical_score` | 辐照物理合理性评分 | 0-1 |
| `irradiance_quality_score` | 辐照质量评分 | 0-1 |
| `irradiance_low_quality` | 低质量辐照标志 | 0/1 |
| `dni_exceeds_ghi_flag` | DNI超过GHI标志 | 0/1 |
| `diffuse_ratio` | 散射比例 | 0-1 |
| `diffuse_anomaly` | 散射异常标志 | 0/1 |
| `direct_contribution_ratio` | 直射贡献率 | 0-1 |

---

## 九、增强特征 - 温度修正 (Temperature Correction)

### 9.1 组件温度估算

| 列名 | 单位 | 说明 |
|------|------|------|
| `estimated_cell_temp` | °C | **估算光伏组件温度** |

**计算公式**:
```
T_cell = T_air + (NOCT - 20) × (GTI / 800) × 0.9
NOCT = 45°C
```

### 9.2 温度效率修正

| 列名 | 说明 | 正常范围 |
|------|------|----------|
| `temp_diff_from_stc` | 与STC(25°C)的温差 | - |
| `temp_efficiency_factor` | **温度效率修正因子** | 0.7-1.1 |
| `heat_stress_index` | 热应力指数 | 0-10 |
| `theoretical_power_stc` | STC下理论功率 | MW |
| `theoretical_power_corrected` | 温度修正后理论功率 | MW |
| `power_temp_efficiency_ratio` | 功率/温度效率比 | - |

**温度修正因子公式**:
```
η = 1 + α × (T_cell - 25)
α = -0.004 / °C (温度系数)
```

### 9.3 温度警告

| 列名 | 说明 | 阈值 |
|------|------|------|
| `high_temp_warning` | 高温警告 | 组件温度>50°C |
| `extreme_temp_warning` | 极端高温警告 | 组件温度>60°C |
| `heat_discomfort_index` | 热不适指数 | - |
| `dew_point_approx` | 估算露点温度 | °C |
| `dew_point_spread` | 露点温差 | °C |
| `temp_irradiance_product` | 温度×辐照度交互 | - |

---

## 十、增强特征 - 效率分析 (Efficiency Analysis)

### 10.1 瞬时效率

| 列名 | 说明 | 正常范围 |
|------|------|----------|
| `instant_efficiency` | **瞬时转换效率** | 0-1 |
| `power_irradiance_ratio` | 功率/辐照度比 | 0-1.5 |

**计算**:
```
η = Power / (GTI / 1000 × Capacity × 0.8)
```

### 10.2 效率趋势

| 列名 | 说明 |
|------|------|
| `efficiency_rolling_mean` | 效率滚动均值(4步) |
| `efficiency_rolling_std` | 效率滚动标准差 |
| `efficiency_change` | 效率变化 |

### 10.3 效率异常

| 列名 | 说明 | 阈值 |
|------|------|------|
| `low_efficiency_flag` | 低效率标志 | <0.6 |
| `high_efficiency_flag` | 高效率标志 | >0.95 |

### 10.4 功率残差

| 列名 | 说明 |
|------|------|
| `power_residual` | 实际功率 - 理论功率 |

---

## 十一、增强特征 - 时段交互 (Time Period Interaction)

### 11.1 时段编码

| 列名 | 说明 | 取值 |
|------|------|------|
| `solar_period` | 太阳高度时段 | 0-6 |
| `is_noon_period` | 是否中午时段 | 0/1 |
| `is_transition` | 是否过渡时段 | 0/1 |

**时段定义**:
```
0: 夜间 (hour < 6)
1: 早晨过渡 (6 <= hour < 9)
2: 上午稳定 (9 <= hour < 11)
3: 中午峰值 (11 <= hour < 14) ← is_noon_period
4: 下午稳定 (14 <= hour < 16)
5: 傍晚过渡 (16 <= hour < 19)
6: 日落后 (hour >= 19)
```

### 11.2 周期性编码

| 列名 | 说明 |
|------|------|
| `hour_sin` | 小时正弦编码 |
| `hour_cos` | 小时余弦编码 |

### 11.3 时段交互

| 列名 | 说明 |
|------|------|
| `noon_temp_interaction` | 中午×温度 |
| `noon_temp_squared` | 中午×温度² |
| `noon_irradiance_interaction` | 中午×辐照度 |
| `noon_efficiency_interaction` | 中午×效率 |

### 11.4 时间周期

| 列名 | 说明 |
|------|------|
| `is_peak_hour` | 是否峰值时段 (11-14时) |
| `is_transition_hour` | 是否过渡时段 |
| `quarter` | 季度 (1-4) |
| `is_summer` | 是否夏季 (6-8月) |
| `is_winter` | 是否冬季 (12-2月) |

### 11.5 太阳高度

| 列名 | 说明 |
|------|------|
| `high_sun_period` | 高太阳时段标志 (仰角>50°) |

---

## 十二、训练相关特征 (Training Features)

| 列名 | 说明 | 用途 |
|------|------|------|
| `sample_weight` | 样本权重 | 训练加权 |
| `use_for_training` | 是否用于训练 | 数据筛选 |
| `power_trend` | 功率趋势方向 | 趋势特征 |
| `power_streak` | 连续趋势计数 | 趋势持续性 |

---

## 特征用途速查表

### 用于模型输入的核心特征

| 优先级 | 特征 | 理由 |
|--------|------|------|
| ⭐⭐⭐ | `total_irradiance_wm2` | 核心预测特征 |
| ⭐⭐⭐ | `air_temperature_c` | 温度修正 |
| ⭐⭐⭐ | `power_pu` | 目标变量 |
| ⭐⭐ | `hour_sin`, `cos_hour` | 时间周期 |
| ⭐⭐ | `direct_normal_irradiance_wm2` | DNI补充 |
| ⭐⭐ | `relative_humidity_pct` | 湿度影响 |
| ⭐ | `estimated_cell_temp` | 热效应 |
| ⭐ | `temp_efficiency_factor` | 温度效率 |
| ⭐ | `is_noon_period` | 中午效应 |

### 用于数据质量筛选

| 条件 | 排除列 |
|------|--------|
| 整体质量 | `overall_quality_score < 0.5` |
| 辐照质量 | `irradiance_low_quality == 1` |
| 停机期间 | `in_outage == 1` |
| 严重异常 | `irradiance_power_anomaly_level >= 2` |

### 用于特征工程

| 目标 | 推荐特征组合 |
|------|-------------|
| 提高辐照预测 | `GTI + DNI + GHI + hour周期` |
| 温度修正 | `GTI + 温度 + temp_efficiency_factor` |
| 捕捉中午效应 | `is_noon_period + noon_temp_interaction` |
| 效率分析 | `instant_efficiency + efficiency_change` |

---

## 数据字典版本信息

| 属性 | 值 |
|------|-----|
| 文档版本 | v1.0 |
| 生成日期 | 2026-08-31 |
| 原始列数 | 20 |
| 当前列数 | 140 |
| 新增特征数 | 120 |
