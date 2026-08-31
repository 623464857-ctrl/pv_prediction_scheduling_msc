# Site1 数据优化计划

## 问题诊断摘要

基于数据诊断，发现以下核心问题：

| 问题类别 | 严重程度 | 问题描述 |
|----------|----------|----------|
| 湿度数据不可信 | 🔴 严重 | 7.1%被标记invalid，与WRF相关性仅0.1 |
| 辐照-功率反比 | 🔴 严重 | 162条高辐照低功率、21条低辐照高功率 |
| WRF时间偏移 | 🟠 中等 | 实测峰值13:00，WRF峰值12:00，相差1小时 |
| 功率突变频繁 | 🟠 中等 | 晴天存在86次大幅突变(>15MW/15min) |
| 季节模式异常 | 🟡 轻度 | 冬季功率反常高于夏季 |

---

## 优化阶段一：数据清洗重构 (EXP-P01-v2)

### 1.1 湿度数据处理优化

**现状问题**：
- 湿度>100%被标记为invalid并设为NaN
- 湿度<20%占比高达57.84%
- 实测与WRF湿度相关性仅0.1042

**优化方案**：

```python
# 方案A：扩展湿度物理边界
PHYSICAL_BOUNDS = {
    "relative_humidity_pct": (0, 100),  # 保持不变
}

# 方案B：增加湿度数据质量分层
def classify_humidity_quality(rh_series, wrf_rh_series):
    """
    低质量湿度：<20% 或 >100%
    中等质量湿度：与WRF偏差<30%
    高质量湿度：与WRF偏差<15%
    """
    pass

# 方案C（推荐）：使用WRF湿度替代
# WRF湿度与温度相关性高(0.93)，更可靠
```

**执行策略**：
1. 增加 `humidity_quality_tier` 列（0=不可用, 1=低, 2=中, 3=高）
2. 在特征工程中，对低质量湿度应用平滑处理
3. 考虑完全移除湿度特征，改用WRF派生特征（如露点差）

### 1.2 辐照-功率异常检测优化

**现状问题**：
- 高辐照(>500)但低功率(<5MW)：162条
- 低辐照(<100)但高功率(>20MW)：21条
- 夜间有功率：220条

**优化方案**：

```python
def detect_irradiance_power_anomalies(df, capacity_mw):
    """
    检测辐照-功率不一致的异常
    
    异常类型：
    1. 晴空无功率：辐照>500但功率<5%容量
    2. 阴天高功率：辐照<100但功率>40%容量  
    3. 夜间发电：辐照<5但功率>0
    4. 功率突降：前后时刻功率变化>30%
    """
    anomalies = pd.DataFrame()
    
    # 计算理论最大功率
    df['theoretical_max_power'] = capacity_mw * 0.95
    
    # 晴空无功率检测
    clear_sky = df['total_irradiance_wm2'] > 500
    no_power = df['power_mw'] < 0.05 * capacity_mw
    anomalies['clear_sky_no_power'] = clear_sky & no_power
    
    # 阴天高功率检测
    cloudy = df['total_irradiance_wm2'] < 100
    high_power = df['power_mw'] > 0.4 * capacity_mw
    anomalies['cloudy_high_power'] = cloudy & high_power
    
    # 夜间发电检测
    night = df['total_irradiance_wm2'] < 5
    power_exists = df['power_mw'] > 0.01 * capacity_mw
    anomalies['night_power'] = night & power_exists
    
    return anomalies

def classify_anomaly_severity(row):
    """
    分类异常严重程度：
    - 0级：正常
    - 1级：轻度异常（可能是多云、快速变化）
    - 2级：重度异常（传感器故障、设备停机）
    - 3级：明显错误（夜间发电）
    """
    pass
```

**处理策略**：

| 异常级别 | 处理方式 | 影响 |
|----------|----------|------|
| 0级 | 保留 | 正常数据 |
| 1级 | 标记但不删除 | 轻度异常 |
| 2级 | 标记为低质量 | 传感器/设备问题 |
| 3级 | 删除或置零 | 明显错误 |

### 1.3 功率突变平滑

**现状问题**：
- 晴天大幅突变>15MW：86次
- 最大单次变化35.4MW

**优化方案**：

```python
def smooth_power_anomalies(df, capacity_mw, threshold_pct=0.3):
    """
    1. 检测突变点：15分钟变化>30%容量
    2. 判断是上升还是下降
    3. 如果是突变（非渐变），进行约束或平滑
    """
    df = df.copy()
    
    # 计算功率变化率
    df['power_diff'] = df['power_mw'].diff()
    df['power_diff_pct'] = df['power_diff'] / capacity_mw
    
    # 检测突变
    sudden_change = df['power_diff_pct'].abs() > threshold_pct
    
    # 渐进变化（非突变）
    gradual_change = (df['power_diff_pct'].abs() > 0.05) & ~sudden_change
    
    # 标记类型
    df['change_type'] = 'normal'
    df.loc[gradual_change, 'change_type'] = 'gradual'
    df.loc[sudden_change, 'change_type'] = 'sudden'
    
    # 对突变进行处理：替换为前后时刻均值
    for idx in df[sudden_change].index:
        if idx > 0 and idx < len(df) - 1:
            prev_power = df.loc[idx-1, 'power_mw']
            next_power = df.loc[idx+1, 'power_mw']
            df.loc[idx, 'power_mw'] = (prev_power + next_power) / 2
    
    return df
```

### 1.4 设备停机检测

**现状问题**：
- 最长连续零功率：247个点(约61小时)

**优化方案**：

```python
def detect_equipment_outage(df, capacity_mw, min_zero_hours=24):
    """
    检测设备停机：
    1. 连续零功率 > 24小时
    2. 功率长期低于预期
    3. 返回停机时段列表
    """
    min_zero_points = min_zero_hours * 4  # 15分钟间隔
    
    # 识别连续零功率段
    zero_mask = df['power_mw'] < 0.01 * capacity_mw
    
    # 找出停机开始和结束
    outage_periods = []
    in_outage = False
    start_idx = None
    
    for idx, is_zero in enumerate(zero_mask):
        if is_zero and not in_outage:
            in_outage = True
            start_idx = idx
        elif not is_zero and in_outage:
            in_outage = False
            duration = idx - start_idx
            if duration >= min_zero_points:
                outage_periods.append({
                    'start': df.iloc[start_idx]['timestamp'],
                    'end': df.iloc[idx-1]['timestamp'],
                    'duration_hours': duration * 15 / 60
                })
    
    return pd.DataFrame(outage_periods)
```

---

## 优化阶段二：WRF数据对齐 (EXP-P01-v2 WRF模块)

### 2.1 时间偏移校正

**现状问题**：
- 实测辐照峰值：13:00
- WRF辐照峰值：12:00
- 相差1小时

**优化方案**：

```python
def correct_wrf_time_offset(df_obs, df_wrf, feature='wrf_gti_wm2'):
    """
    检测并校正WRF时间偏移
    
    方法：
    1. 计算实测与WRF的互相关
    2. 找到最大相关性对应的时间偏移
    3. 应用校正
    """
    from scipy.signal import correlate
    
    # 取日间数据进行对比
    obs_daytime = df_obs[df_obs['total_irradiance_wm2'] > 50][feature.replace('wrf_', '')]
    wrf_daytime = df_wrf[df_wrf['total_irradiance_wm2'] > 50][feature]
    
    # 计算互相关
    correlation = correlate(obs_daytime, wrf_daytime, mode='full')
    lags = np.arange(-len(obs_daytime)+1, len(obs_daytime))
    
    # 找到最大相关性对应的时间偏移
    best_lag = lags[np.argmax(correlation)]
    
    return best_lag

def apply_wrf_correction(df_wrf, lag_hours):
    """
    应用时间校正：
    - 正lag：WRF数据需要前移
    - 负lag：WRF数据需要后移
    """
    lag_points = lag_hours * 4  # 转换为15分钟点数
    if lag_points > 0:
        df_wrf[feature] = df_wrf[feature].shift(lag_points)
    return df_wrf
```

### 2.2 WRF辐照校正

**现状问题**：
- WRF显著高于实测(>1.5倍)：9301条
- WRF显著低于实测(<0.5倍)：14457条

**优化方案**：

```python
def correct_wrf_irradiance(df_merged, window_days=7):
    """
    使用滑动窗口对WRF辐照进行偏差校正
    
    原理：晴空时，实测与理论最大辐照的比值应该稳定
    """
    df = df_merged.copy()
    
    # 计算理论最大GHI（根据太阳高度角）
    # 这需要天文计算，此处简化处理
    
    # 计算每日实测-WRF偏差
    daily_ratio = df.groupby(df['timestamp'].dt.date).apply(
        lambda x: x['total_irradiance_wm2'].sum() / (x['wrf_gti_wm2'].sum() + 1e-6)
    )
    
    # 平滑偏差
    daily_ratio_smooth = daily_ratio.rolling(window_days, center=True).mean()
    
    # 应用校正
    # 校正后WRF = WRF * 平滑偏差
    pass
```

---

## 优化阶段三：特征工程增强 (EXP-P04 特征模块)

### 3.1 新增辐照质量特征

```python
def add_irradiance_quality_features(df):
    """
    新增辐照质量相关特征
    """
    df = df.copy()
    
    # 1. 辐照利用率
    # 实际辐照 / 理论最大辐照
    df['gti_utilization'] = df['total_irradiance_wm2'] / (
        df['theoretical_max_gti'] + 1e-6
    )
    
    # 2. DNI/DHI比例异常检测
    df['dni_dhi_ratio'] = df['direct_normal_irradiance_wm2'] / (
        df['global_horizontal_irradiance_wm2'] + 1e-6
    )
    
    # 3. 辐照变化率（检测云层快速变化）
    df['gti_change_rate'] = df['total_irradiance_wm2'].diff() / 15  # W/m²/分钟
    
    # 4. 辐照平滑度（滚动标准差/均值）
    df['gti_smoothness'] = df['total_irradiance_wm2'].rolling(4).std() / (
        df['total_irradiance_wm2'].rolling(4).mean() + 1e-6
    )
    
    # 5. 辐照数据质量标记
    # 低质量：比理论值高 或 波动剧烈
    df['gti_quality'] = 1.0  # 默认高质量
    df.loc[df['gti_utilization'] > 1.1, 'gti_quality'] = 0.5  # 超理论值
    df.loc[df['gti_smoothness'] > 0.5, 'gti_quality'] = 0.5   # 剧烈波动
    
    return df
```

### 3.2 新增功率-辐照一致性特征

```python
def add_power_irradiance_consistency_features(df, capacity_mw):
    """
    新增功率-辐照一致性特征
    """
    df = df.copy()
    
    # 1. 实际功率/理论功率（基于辐照估算）
    # 理论功率 ≈ GTI/1000 * 容量 * 效率(约0.8)
    df['theoretical_power'] = df['total_irradiance_wm2'] / 1000 * capacity_mw * 0.8
    df['power_vs_theoretical'] = df['power_mw'] / (df['theoretical_power'] + 1e-6)
    
    # 2. 功率效率（实际/理论，最大1.0）
    df['power_efficiency'] = df['power_vs_theoretical'].clip(upper=1.0)
    
    # 3. 异常功率标记
    df['power_anomaly_flag'] = 0
    df.loc[df['power_vs_theoretical'] < 0.1, 'power_anomaly_flag'] = 1  # 远低于理论
    df.loc[df['power_vs_theoretical'] > 1.0, 'power_anomaly_flag'] = 1  # 超过理论
    
    # 4. 晴空指数一致性
    # GTI清朗指数 vs 功率效率应该相关
    df['clear_sky_consistency'] = df['gti_utilization'] * df['power_efficiency']
    
    return df
```

### 3.3 新增数据质量权重

```python
def compute_sample_quality_weight(df):
    """
    计算每个样本的数据质量权重，用于训练时加权
    """
    df = df.copy()
    
    # 基础质量分
    quality_score = 1.0
    
    # 辐照异常扣分
    if df.get('gti_quality', 1.0) < 1.0:
        quality_score *= 0.7
    
    # 湿度低质量扣分
    if df.get('humidity_quality_tier', 3) < 2:
        quality_score *= 0.8
    
    # 功率异常扣分
    if df.get('power_anomaly_flag', 0) == 1:
        quality_score *= 0.5
    
    # 插值数据扣分
    if df.get('imputed_feature_count', 0) > 2:
        quality_score *= 0.7
    
    df['quality_weight'] = quality_score
    
    return df
```

---

## 优化阶段四：数据验证框架

### 4.1 物理一致性检验

```python
def validate_physics_consistency(df, capacity_mw):
    """
    物理一致性检验清单
    """
    issues = []
    
    # 1. 夜间无功率
    night_mask = df['total_irradiance_wm2'] < 5
    night_power = df.loc[night_mask, 'power_mw']
    if (night_power > 0.01 * capacity_mw).any():
        issues.append("夜间存在非零功率")
    
    # 2. 功率不超过容量
    over_capacity = df['power_mw'] > 1.05 * capacity_mw
    if over_capacity.any():
        issues.append(f"存在超容量功率，共{over_capacity.sum()}条")
    
    # 3. 晴空高辐照应有发电
    clear_mask = (df['total_irradiance_wm2'] > 700) & (df['hour'].between(10, 14))
    clear_no_power = df.loc[clear_mask, 'power_mw'] < 0.3 * capacity_mw
    if clear_no_power.any():
        issues.append(f"晴空高辐照但低发电，共{clear_no_power.sum()}条")
    
    # 4. 辐照不超过理论最大值
    theoretical_max = df['theoretical_max_gti'] * 1.05  # 5%容差
    over_theoretical = df['total_irradiance_wm2'] > theoretical_max
    if over_theoretical.any():
        issues.append(f"辐照超过理论最大值，共{over_theoretical.sum()}条")
    
    # 5. 温度-辐照关系检查
    # 高辐照通常对应较高温度
    high_irr_low_temp = (df['total_irradiance_wm2'] > 500) & (df['air_temperature_c'] < -20)
    if high_irr_low_temp.any():
        issues.append(f"高辐照低温异常，共{high_irr_low_temp.sum()}条")
    
    return issues
```

### 4.2 统计异常检测

```python
def detect_statistical_outliers(df, feature, z_threshold=4):
    """
    使用Z-score检测统计异常
    """
    from scipy import stats
    
    data = df[feature].dropna()
    z_scores = np.abs(stats.zscore(data))
    outliers = z_scores > z_threshold
    
    return df[feature].index[outliers]
```

---

## 执行计划

### 阶段一：数据清洗重构（预计1-2天）

| 步骤 | 任务 | 优先级 | 预计时间 |
|------|------|--------|----------|
| 1.1 | 重写湿度处理逻辑 | P0 | 4小时 |
| 1.2 | 实现辐照-功率异常检测 | P0 | 4小时 |
| 1.3 | 实现功率突变平滑 | P1 | 2小时 |
| 1.4 | 实现设备停机检测 | P1 | 2小时 |
| 1.5 | 增加数据质量分层标记 | P0 | 4小时 |
| 1.6 | 完整流程测试与验证 | P0 | 4小时 |

### 阶段二：WRF数据优化（预计1天）

| 步骤 | 任务 | 优先级 | 预计时间 |
|------|------|--------|----------|
| 2.1 | 实现时间偏移检测 | P0 | 4小时 |
| 2.2 | 应用时间校正 | P0 | 2小时 |
| 2.3 | WRF辐照偏差校正 | P1 | 4小时 |
| 2.4 | WRF质量验证 | P0 | 2小时 |

### 阶段三：特征工程增强（预计1天）

| 步骤 | 任务 | 优先级 | 预计时间 |
|------|------|--------|----------|
| 3.1 | 新增辐照质量特征 | P0 | 2小时 |
| 3.2 | 新增功率一致性特征 | P0 | 2小时 |
| 3.3 | 实现质量权重计算 | P1 | 2小时 |
| 3.4 | 更新特征文档 | P2 | 1小时 |

### 阶段四：验证与报告（预计1天）

| 步骤 | 任务 | 优先级 | 预计时间 |
|------|------|--------|----------|
| 4.1 | 实现物理一致性检验 | P0 | 2小时 |
| 4.2 | 实现统计异常检测 | P0 | 2小时 |
| 4.3 | 生成数据质量报告 | P0 | 2小时 |
| 4.4 | 可视化诊断结果 | P1 | 2小时 |

---

## 预期成果

1. **数据质量提升**：
   - 高质量数据占比 > 90%（当前约60%）
   - 异常数据自动识别率 > 95%

2. **特征可靠性提升**：
   - 辐照-功率相关系数 > 0.95
   - WRF-实测相关系数 > 0.90

3. **实验可重复性**：
   - 每条数据附带质量权重
   - 完整的异常标记可追溯

4. **文档产出**：
   - Site1数据质量诊断报告
   - 数据清洗流程说明
   - 特征工程规范文档

---

## 注意事项

1. **不要删除异常数据，而是标记**：删除会导致样本不均衡，标记后可用于分析
2. **保持与原始数据的可追溯性**：所有处理步骤都需要记录
3. **分层处理**：区分可修复异常（如传感器噪声）和不可修复异常（如设备停机）
4. **交叉验证**：清洗后的数据需要与相邻站点对比，验证合理性

---

## 后续建议

如果数据清洗后仍存在大量无法解释的异常，建议：

1. **联系数据提供方**：确认是否存在传感器故障记录
2. **获取外部数据源**：如官方气象数据、电网调度记录
3. **考虑站点替换**：如果Site1数据质量确实无法满足研究需求，可考虑使用其他站点

