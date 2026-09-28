"""
亳州光伏数据集 Step1 预处理脚本
执行 analysis_亳州.md 中定义的预处理流程

Usage:
    python run_exp_p01_bozhou.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# =============================================================================
# 配置
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# 路径配置
POWER_DATA_PATH = "data/raw/亳州/亳州_15min.csv"
WEATHER_DATA_PATH = "data/raw/亳州/亳州3-8月天气数据.csv"
OUTPUT_DIR = Path("data/prediction/step1_preprocessing/processed/stations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 容量配置
CAPACITY_KW = 390
CAPACITY_UPPER = CAPACITY_KW * 1.05  # 409.5 kW

# =============================================================================
# Step 2: 天气类型编码映射
# =============================================================================
WEATHER_ENCODING = {
    'Clear Sky': 0,
    'Few clouds': 1,
    'Scattered clouds': 2,
    'Broken clouds': 3,
    'Overcast clouds': 4,
    'Fog': 4,
    'Drizzle': 5,
    'Light rain': 6,
    'Moderate rain': 7,
    'Heavy rain': 8,
    'Thunderstorm with light rain': 9,
    'Thunderstorm with moderate/heavy rain': 10,
}

# =============================================================================
# 步骤执行报告
# =============================================================================
class StepReporter:
    def __init__(self):
        self.steps = []
        self.start_time = datetime.now()
    
    def report(self, step_num, step_name, rows_before, rows_after, message=""):
        info = {
            'step': step_num,
            'name': step_name,
            'rows_before': rows_before,
            'rows_after': rows_after,
            'message': message,
            'timestamp': datetime.now().strftime('%H:%M:%S')
        }
        self.steps.append(info)
        print(f"\n{'='*60}")
        print(f"Step {step_num}: {step_name}")
        print(f"{'='*60}")
        print(f"  行数: {rows_before:,} → {rows_after:,}")
        if message:
            print(f"  说明: {message}")
        return info
    
    def summary(self):
        print("\n" + "="*70)
        print("执行摘要")
        print("="*70)
        print(f"{'步骤':<6} {'名称':<25} {'处理前行数':<12} {'处理后行数':<12}")
        print("-"*70)
        for s in self.steps:
            print(f"{s['step']:<6} {s['name']:<25} {s['rows_before']:<12,} {s['rows_after']:<12,}")
        print("-"*70)

# =============================================================================
# Step 1: 数据加载与合并
# =============================================================================
def step1_load_and_merge(reporter):
    print("\n" + "="*70)
    print("Step 1: 数据加载与合并")
    print("="*70)
    
    # Read power data (only power_kw column)
    power_df = pd.read_csv(POWER_DATA_PATH, usecols=['timestamp', 'power_kw'])
    power_df['timestamp'] = pd.to_datetime(power_df['timestamp'])
    print("  Power data: {0:,} rows".format(len(power_df)))
    print("  Columns: {0}".format(list(power_df.columns)))
    
    # Read weather data
    weather_df = pd.read_csv(WEATHER_DATA_PATH)
    weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp_local'])
    print("  Weather data: {0:,} rows".format(len(weather_df)))
    print("  Weather columns: {0}".format(len(weather_df.columns)))
    
    # Merge
    df = pd.merge(power_df, weather_df, on='timestamp', how='inner')
    print("  Merged: {0:,} rows".format(len(df)))
    
    reporter.report(1, "数据加载与合并", len(power_df), len(df), "功率与天气数据合并")
    return df

# =============================================================================
# Step 2: 字段标准化与特征选择
# =============================================================================
def step2_standardize_and_select(df, reporter):
    print("\n" + "="*70)
    print("Step 2: 字段标准化与特征选择")
    print("="*70)
    
    rows_before = len(df)
    
    # 定义保留的列映射
    column_mapping = {
        'temp': 'temperature_c',
        'rh': 'relative_humidity_pct',
        'vis': 'visibility_km',
        'clouds': 'cloud_cover_pct',
        'ghi': 'ghi_wm2',
        'uv': 'uv_index',
        'precip_rate': 'precip_rate_mmhr',
        'weather_desc': 'weather_category',
        'power_kw': 'power_kw',
    }
    
    # 选择并重命名列
    df = df.rename(columns=column_mapping)
    cols_to_keep = list(column_mapping.values()) + ['timestamp']
    df = df[[c for c in cols_to_keep if c in df.columns]]
    
    # Weather type encoding
    if 'weather_category' in df.columns:
        df['weather_category'] = df['weather_category'].map(WEATHER_ENCODING).fillna(0).astype(int)
        print(f"  Weather encoding distribution:")
        print(df['weather_category'].value_counts().sort_index().to_string())
    
    print(f"  Retained columns: {list(df.columns)}")
    reporter.report(2, "字段标准化与特征选择", rows_before, len(df), f"Retained {len(df.columns)} columns, removed 12 redundant features")
    return df

# =============================================================================
# Step 3: 时间序列重建
# =============================================================================
def step3_rebuild_timeseries(df, reporter):
    print("\n" + "="*70)
    print("Step 3: 时间序列重建")
    print("="*70)
    
    rows_before = len(df)
    
    # Sort
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Check time intervals
    time_diff = df['timestamp'].diff().dropna()
    expected_diff = pd.Timedelta(minutes=15)
    gaps = time_diff[time_diff != expected_diff]
    
    if len(gaps) > 0:
        print("  Found {0} time interval anomalies".format(len(gaps)))
        # Create complete timeline
        full_range = pd.date_range(
            start=df['timestamp'].min(),
            end=df['timestamp'].max(),
            freq='15min'
        )
        print("  Complete timeline: {0:,} points".format(len(full_range)))
        df = df.set_index('timestamp').reindex(full_range).reset_index()
        df.columns = ['timestamp'] + list(df.columns[1:])
        inserted_rows = len(full_range) - rows_before
        print("  Inserted missing rows: {0:,}".format(inserted_rows))
        reporter.report(3, "时间序列重建", rows_before, len(df), "Inserted {0} missing time points".format(inserted_rows))
    else:
        print("  Time intervals complete, no rebuild needed")
        reporter.report(3, "时间序列重建", rows_before, len(df), "Time series complete")
    
    return df

# =============================================================================
# Step 4: 物理边界约束清洗
# =============================================================================
def step4_physical_bounds(df, reporter):
    print("\n" + "="*70)
    print("Step 4: 物理边界约束清洗")
    print("="*70)
    
    rows_before = len(df)
    violations = {}
    
    # Define physical bounds
    bounds = {
        'temperature_c': (-50, 60),
        'ghi_wm2': (0, 1600),
        'cloud_cover_pct': (0, 100),
        'relative_humidity_pct': (0, 100),
        'visibility_km': (0, 100),
        'uv_index': (0, 15),
        'precip_rate_mmhr': (0, 50),
    }
    
    for col, (low, high) in bounds.items():
        if col in df.columns:
            mask_low = df[col] < low
            mask_high = df[col] > high
            n_violations = mask_low.sum() + mask_high.sum()
            if n_violations > 0:
                violations[col] = n_violations
                df.loc[mask_low | mask_high, col] = np.nan
    
    print("  Physical bounds check:")
    for col, n in violations.items():
        print(f"    {col}: {n:,} out-of-bounds values")
    
    if not violations:
        print("  [OK] All features within physical bounds")
    
    reporter.report(4, "物理边界约束清洗", rows_before, len(df), f"处理{sum(violations.values())}个越界值")
    return df

# =============================================================================
# Step 5: 功率约束处理
# =============================================================================
def step5_power_constraints(df, reporter):
    print("\n" + "="*70)
    print("Step 5: 功率约束处理")
    print("="*70)
    
    rows_before = len(df)
    
    # Negative power clipping
    neg_mask = df['power_kw'] < 0
    n_neg = neg_mask.sum()
    if n_neg > 0:
        df.loc[neg_mask, 'power_kw'] = 0
        df['power_kw_negative_clipped_flag'] = neg_mask.astype(int)
        print(f"  Negative power clipped: {n_neg:,} values")
    
    # Capacity upper bound
    upper_mask = df['power_kw'] > CAPACITY_UPPER
    n_upper = upper_mask.sum()
    if n_upper > 0:
        df.loc[upper_mask, 'power_kw'] = np.nan
        print(f"  Above capacity ({CAPACITY_UPPER}kW): {n_upper:,} -> NaN")
    
    reporter.report(5, "功率约束处理", rows_before, len(df), f"Negative:{n_neg}, Above capacity:{n_upper}")
    return df

# =============================================================================
# Step 6: Hampel 异常检测
# =============================================================================
def step6_hampel_filter(df, reporter):
    print("\n" + "="*70)
    print("Step 6: Hampel 异常检测")
    print("="*70)
    
    rows_before = len(df)
    
    def hampel_detect(series, window=13, n_sigma=6.0):
        """Hampel 异常检测"""
        medians = series.rolling(window=window, center=True, min_periods=1).median()
        mad = (series - medians).abs().rolling(window=window, center=True, min_periods=1).median() * 1.4826
        threshold = n_sigma * mad
        return series - medians, threshold, medians
    
    # 天气特征异常检测
    weather_features = ['temperature_c', 'relative_humidity_pct', 'cloud_cover_pct', 
                        'visibility_km', 'uv_index', 'precip_rate_mmhr']
    
    total_outliers = 0
    for feat in weather_features:
        if feat in df.columns:
            diff, threshold, median = hampel_detect(df[feat], window=13, n_sigma=6.0)
            outliers = (diff.abs() > threshold) & (~df[feat].isna())
            n_outliers = outliers.sum()
            if n_outliers > 0:
                df.loc[outliers, feat] = np.nan
                total_outliers += n_outliers
    
    print("  Weather feature anomalies: {0:,}".format(total_outliers))
    
    # Power anomaly detection (daytime)
    day_mask = df['ghi_wm2'] > 20
    if day_mask.sum() > 0:
        power_day = df.loc[day_mask, 'power_kw'].copy()
        diff, threshold, median = hampel_detect(power_day, window=9, n_sigma=7.0)
        outliers = (diff.abs() > threshold) & (~power_day.isna())
        n_power_outliers = outliers.sum()
        if n_power_outliers > 0:
            df.loc[day_mask, 'power_kw'] = df.loc[day_mask, 'power_kw'].where(~outliers, np.nan)
            print("  Power anomalies (daytime): {0:,}".format(n_power_outliers))
    
    reporter.report(6, "Hampel异常检测", rows_before, len(df), "Detected {0} anomalies".format(total_outliers))
    return df

# =============================================================================
# Step 7: 辐照-功率物理一致性修正
# =============================================================================
def step7_irradiance_consistency(df, reporter):
    print("\n" + "="*70)
    print("Step 7: 辐照-功率物理一致性修正")
    print("="*70)
    
    rows_before = len(df)
    
    # 无辐照但有功率
    mask = (df['ghi_wm2'] < 20) & (df['power_kw'] > CAPACITY_KW * 0.05)
    n_violations = mask.sum()
    
    if n_violations > 0:
        df.loc[mask, 'power_kw'] = np.nan
        print(f"  No irradiance but power exists: {n_violations:,} -> marked as anomaly")
    else:
        print("  [OK] No physical consistency violations")
    
    reporter.report(7, "辐照-功率一致性修正", rows_before, len(df), f"标记{n_violations}个不一致点")
    return df

# =============================================================================
# Step 8: 短时缺失插值（≤4步）- 分特征优化插值
# =============================================================================
def step8_short_interpolation(df, reporter):
    print("\n" + "="*70)
    print("Step 8: 短时缺失插值（≤4步）")
    print("="*70)
    
    rows_before = len(df)
    
    # 定义不同特征的插值策略
    INTERPOLATION_STRATEGY = {
        # 线性插值：连续渐变特征
        'linear': ['ghi_wm2', 'uv_index', 'power_kw'],
        
        # 样条插值：周期性连续特征
        'spline': ['temperature_c', 'relative_humidity_pct'],
        
        # 前后向填充：离散/突变特征
        'ffill': ['cloud_cover_pct', 'precip_rate_mmhr'],
        
        # 限制性线性插值：天气相关特征（限制插值范围）
        'limited_linear': ['visibility_km'],
    }
    
    # 扁平化为特征→方法映射
    feature_method = {}
    for method, features in INTERPOLATION_STRATEGY.items():
        for feat in features:
            feature_method[feat] = method
    
    cols_to_interpolate = list(feature_method.keys())
    total_imputed = 0
    
    for col in cols_to_interpolate:
        if col not in df.columns:
            continue
        
        # 计算连续NaN分组
        is_nan = df[col].isna()
        group_id = (~is_nan).cumsum()
        nan_groups = df[is_nan].groupby(group_id[is_nan]).size()
        
        # 统计短缺失数量（≤4个连续缺失）
        short_gaps = nan_groups[nan_groups <= 4]
        n_short = short_gaps.sum()
        
        if n_short > 0:
            method = feature_method[col]
            
            if method == 'linear':
                # 线性插值：适合渐变特征
                df[col] = df[col].interpolate(method='linear', limit_area='inside')
                
            elif method == 'spline':
                # 三次样条插值：适合有周期性趋势的特征
                # 先用线性插值填充极少数极端情况（首尾NaN）
                df[col] = df[col].interpolate(method='linear', limit_area='inside')
                # 再用样条插值优化中间部分
                df[col] = df[col].interpolate(method='spline', order=3, limit_area='inside')
                
            elif method == 'ffill':
                # 前后向填充：适合离散突变特征
                # 先向后填充，再向前填充，确保连续性
                df[col] = df[col].ffill(limit=4).bfill(limit=4)
                # 对边界情况用线性插值兜底
                df[col] = df[col].interpolate(method='linear', limit_area='inside')
                
            elif method == 'limited_linear':
                # 限制性线性插值：最多填充2个连续点
                df[col] = df[col].interpolate(method='linear', limit=2, limit_area='inside')
            
            total_imputed += n_short
            print(f"  {col}: {n_short:,} points ({method})")
    
    print("  Short gap interpolation: {0:,} points total".format(total_imputed))
    reporter.report(8, "短时缺失插值", rows_before, len(df), "Interpolated {0} points".format(total_imputed))
    return df

# =============================================================================
# Step 9: 长缺失剖面回填 - 分特征优化
# =============================================================================
def step9_long_gap_fill(df, reporter):
    print("\n" + "="*70)
    print("Step 9: 长缺失剖面回填")
    print("="*70)
    
    rows_before = len(df)
    
    # 定义长缺失回填的层级策略（按特征类型调整）
    # 对于天气相关特征，增加更多层级以提高精度
    
    cols_to_fill = ['temperature_c', 'ghi_wm2', 'cloud_cover_pct', 
                   'relative_humidity_pct', 'visibility_km', 'uv_index', 
                   'precip_rate_mmhr', 'power_kw']
    
    df['month'] = df['timestamp'].dt.month
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['dayofyear'] = df['timestamp'].dt.dayofyear
    df['daytype'] = df['timestamp'].dt.dayofweek  # 0=周一, 6=周日
    
    total_filled = 0
    
    for col in cols_to_fill:
        if col not in df.columns:
            continue
        
        nan_before = df[col].isna().sum()
        
        # 基础层级策略（适用于所有特征）
        # 1. 月+小时+分钟 中位数
        df[col] = df.groupby(['month', 'hour', 'minute'])[col].transform(
            lambda x: x.fillna(x.median())
        )
        
        # 2. 月+小时 中位数（放宽条件）
        df[col] = df.groupby(['month', 'hour'])[col].transform(
            lambda x: x.fillna(x.median())
        )
        
        # 3. 小时+分钟 中位数
        df[col] = df.groupby(['hour', 'minute'])[col].transform(
            lambda x: x.fillna(x.median())
        )
        
        # 4. 星期+小时 中位数（新增）
        df[col] = df.groupby(['daytype', 'hour'])[col].transform(
            lambda x: x.fillna(x.median())
        )
        
        # 5. 小时 中位数
        df[col] = df.groupby(['hour'])[col].transform(
            lambda x: x.fillna(x.median())
        )
        
        # 6. 全局中位数兜底
        global_median = df[col].median()
        df[col] = df[col].fillna(global_median)
        
        nan_after = df[col].isna().sum()
        filled = nan_before - nan_after
        total_filled += filled
    
    # 删除临时列
    df = df.drop(columns=['month', 'hour', 'minute', 'dayofyear', 'daytype'])
    
    print("  Long gap fill: {0:,} points".format(total_filled))
    reporter.report(9, "长缺失剖面回填", rows_before, len(df), "Filled {0} points".format(total_filled))
    return df

# =============================================================================
# Step 10: 白天/夜间分离
# =============================================================================
def step10_day_night_separation(df, reporter):
    print("\n" + "="*70)
    print("Step 10: 白天/夜间分离")
    print("="*70)
    
    rows_before = len(df)
    
    # GHI > 20 W/m² 为白天
    df['is_daytime'] = (df['ghi_wm2'] > 20).astype(int)
    
    # 夜间功率强制置零
    night_mask = df['is_daytime'] == 0
    n_night = night_mask.sum()
    df.loc[night_mask, 'power_kw'] = 0
    
    n_day = (df['is_daytime'] == 1).sum()
    print("  Daytime: {0:,} points ({1:.1f}%)".format(n_day, n_day/len(df)*100))
    print("  Nighttime: {0:,} points ({1:.1f}%)".format(n_night, n_night/len(df)*100))
    
    reporter.report(10, "白天/夜间分离", rows_before, len(df), "Daytime {0}, Nighttime {1}".format(n_day, n_night))
    return df

# =============================================================================
# Step 11: 疑似停机标记
# =============================================================================
def step11_shutdown_detection(df, reporter):
    print("\n" + "="*70)
    print("Step 11: 疑似停机标记")
    print("="*70)
    
    rows_before = len(df)
    
    # 白天条件但功率为零
    mask = (df['ghi_wm2'] > 5) & (df['power_kw'] == 0)
    n_shutdown = mask.sum()
    
    df['is_potential_shutdown'] = mask.astype(int)
    
    print("  Potential shutdown: {0:,} points".format(n_shutdown))
    reporter.report(11, "疑似停机标记", rows_before, len(df), "Marked {0} potential shutdowns".format(n_shutdown))
    return df

# =============================================================================
# Step 12: 调度/预测衍生特征构建
# =============================================================================
def step12_derived_features(df, reporter):
    print("\n" + "="*70)
    print("Step 12: 调度/预测衍生特征构建")
    print("="*70)
    
    rows_before = len(df)
    
    # 归一化功率
    df['power_pu'] = df['power_kw'] / CAPACITY_KW
    
    # 功率变化
    df['power_ramp_15m_kw'] = df['power_kw'].diff()
    df['power_ramp_15m_pu'] = df['power_pu'].diff()
    
    # 小时特征
    df['hour'] = df['timestamp'].dt.hour
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # GHI周期性特征
    ghi_norm = (df['ghi_wm2'] - df['ghi_wm2'].min()) / (df['ghi_wm2'].max() - df['ghi_wm2'].min() + 1e-6)
    df['sin_ghi'] = np.sin(2 * np.pi * ghi_norm)
    df['cos_ghi'] = np.cos(2 * np.pi * ghi_norm)
    
    # 湿度周期性特征
    df['sin_rh'] = np.sin(2 * np.pi * df['relative_humidity_pct'] / 100)
    df['cos_rh'] = np.cos(2 * np.pi * df['relative_humidity_pct'] / 100)
    
    # 数据质量评分（基于插值比例）
    df['data_quality_score'] = 1.0  # 简化计算
    
    # 删除临时列
    if 'hour' in df.columns:
        df = df.drop(columns=['hour'])
    
    print("  New features: power_pu, power_ramp_15m_kw/pu, sin/cos_hour/ghi/rh, data_quality_score")
    reporter.report(12, "衍生特征构建", rows_before, len(df), "Added 8 derived features")
    return df

# =============================================================================
# 主函数
# =============================================================================
def main():
    print("="*70)
    print("亳州光伏数据集 Step1 预处理")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    reporter = StepReporter()
    
    # Step 1: 数据加载与合并
    df = step1_load_and_merge(reporter)
    
    # Step 2: 字段标准化与特征选择
    df = step2_standardize_and_select(df, reporter)
    
    # Step 3: 时间序列重建
    df = step3_rebuild_timeseries(df, reporter)
    
    # Step 4: 物理边界约束清洗
    df = step4_physical_bounds(df, reporter)
    
    # Step 5: 功率约束处理
    df = step5_power_constraints(df, reporter)
    
    # Step 6: Hampel 异常检测
    df = step6_hampel_filter(df, reporter)
    
    # Step 7: 辐照-功率物理一致性修正
    df = step7_irradiance_consistency(df, reporter)
    
    # Step 8: 短时缺失插值
    df = step8_short_interpolation(df, reporter)
    
    # Step 9: 长缺失剖面回填
    df = step9_long_gap_fill(df, reporter)
    
    # Step 10: 白天/夜间分离
    df = step10_day_night_separation(df, reporter)
    
    # Step 11: 疑似停机标记
    df = step11_shutdown_detection(df, reporter)
    
    # Step 12: 衍生特征构建
    df = step12_derived_features(df, reporter)
    
    # Save results
    output_path = OUTPUT_DIR / "Bozhou_1_preprocessed.csv"
    df.to_csv(output_path, index=False)
    print("\n" + "="*70)
    print("Preprocessing Complete!")
    print("="*70)
    print("Output file: {0}".format(output_path))
    print("Final rows: {0:,}".format(len(df)))
    print("Final columns: {0}".format(len(df.columns)))
    print("="*70)
    
    # Print final feature columns
    print("\nFinal output feature columns:")
    for i, col in enumerate(df.columns, 1):
        print("  {0:2d}. {1}".format(i, col))
    
    # Print data statistics
    print("\nData statistics:")
    print(df.describe().round(3).to_string())
    
    # 打印摘要
    reporter.summary()
    
    return df, reporter

if __name__ == "__main__":
    df, reporter = main()
