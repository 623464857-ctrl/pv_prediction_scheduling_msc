"""
预测 P1: 原始功率 + 天气数据预处理 (11 步清洗流程)

输入: data/raw/亳州/亳州_15min.csv, 亳州3-8月天气数据.csv
输出: data/processed/Bozhou_1_preprocessed.csv

python -m experiments.prediction.p1_preprocess
python -m experiments.prediction.p1_preprocess --method simple --sigma 3
python -m experiments.prediction.p1_preprocess --method two_stage --sigma 6
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# =============================================================================
# 命令行参数解析
# =============================================================================
parser = argparse.ArgumentParser(description="光伏数据预处理")
parser.add_argument("--method", type=str, default="two_stage",
                    choices=["simple", "two_stage"],
                    help="Hampel 异常检测方法: simple(单阶段) / two_stage(两级判断)")
parser.add_argument("--sigma", type=float, default=3.0,
                    help="Hampel sigma 阈值 (默认 3.0)")
parser.add_argument("--horizon", type=int, nargs="+", default=[1],
                    choices=[1, 4, 16],
                    help="预测步长 (用于生成配置名称)")
parser.add_argument("--mode", type=str, default="hybrid",
                    help="超参数模式 (用于生成配置名称)")
args_cli = parser.parse_args()

# =============================================================================
# 配置
# =============================================================================
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.common import RAW_POWER_CSV, RAW_WEATHER_CSV, CAPACITY_KW, PROCESSED_DIR

# 路径配置
POWER_DATA_PATH = RAW_POWER_CSV
WEATHER_DATA_PATH = RAW_WEATHER_CSV

# 容量配置
CAPACITY_UPPER = CAPACITY_KW * 1.05  # 409.5 kW

# Hampel 配置（从命令行参数读取）
HAMPEL_METHOD = args_cli.method        # 'simple' 或 'two_stage'
HAMPEL_WEATHER_SIGMA = args_cli.sigma  # 天气特征阈值
HAMPEL_POWER_SIGMA = args_cli.sigma    # 功率阈值
HAMPEL_DAY_START = 5        # 白天开始时间（5:00）
HAMPEL_DAY_END = 20         # 白天结束时间（20:00）

# 配置文件名标签
CFG_LABEL = f"{HAMPEL_METHOD}_sigma{int(HAMPEL_WEATHER_SIGMA)}"

# 数据划分比例（用于避免预处理时的数据泄露）
TRAIN_RATIO = 0.70

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
    weather_df = pd.read_csv(WEATHER_DATA_PATH, encoding="gbk")
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
    
    # 定义保留的列映射（仅保留4个天气特征 + 功率）
    column_mapping = {
        'temp': 'temperature_c',
        'rh': 'relative_humidity_pct',
        'clouds': 'cloud_cover_pct',
        'ghi': 'ghi_wm2',
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
    
    # Define physical bounds（仅针对保留的特征）
    bounds = {
        'temperature_c': (-50, 60),
        'ghi_wm2': (0, 1600),
        'cloud_cover_pct': (0, 100),
        'relative_humidity_pct': (0, 100),
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
# Step 6: 辐照-功率物理一致性修正
# =============================================================================
def step6_irradiance_consistency(df, reporter):
    print("\n" + "="*70)
    print("Step 6: 辐照-功率物理一致性修正")
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
    
    reporter.report(6, "辐照-功率一致性修正", rows_before, len(df), f"标记{n_violations}个不一致点")
    return df

# =============================================================================
# 局部突变豁免（趋势续接一致性）
# =============================================================================
def _theil_sen_slope(offsets, values):
    """Theil–Sen 稳健斜率：values 随 offsets 变化的所有两点斜率的中位数"""
    n = len(values)
    slopes = []
    for a in range(n):
        for c in range(a + 1, n):
            d_off = offsets[c] - offsets[a]
            if d_off:
                slopes.append((values[c] - values[a]) / d_off)
    return float(np.median(slopes)) if slopes else 0.0


def _trend_side(series, pos, direction, excluded, n_neighbors=3, max_gap=6):
    """估计 pos 某一侧的局部趋势。

    取该侧最近的 n_neighbors 个"未被标记"的样本（跨过已标记点继续向外找，
    但最多向外找 max_gap 步），用 Theil–Sen 斜率外推/回推到 pos 时刻。

    返回 (趋势值 cont, 容差 tol)：
      - 左侧(direction=-1)：cont = 最近值 + 前向斜率（趋势再延续一步）
      - 右侧(direction=+1)：cont = 最近值 - 前向斜率（趋势从右侧回推一步）
      - 容差 = 该侧样本离散度(MAD) 与 斜率量级 的较大者
    """
    pts = []
    k = 1
    while len(pts) < n_neighbors and k <= max_gap:
        j = pos + direction * k
        if not (0 <= j < len(series)):
            break
        v = series.iloc[j]
        if pd.notna(v) and j not in excluded:
            pts.append((k, float(v)))
        k += 1
    if len(pts) < 2:
        return None

    offsets = [o for o, _ in pts]
    values = [v for _, v in pts]
    slope = _theil_sen_slope(offsets, values)   # 沿 offsets 增大方向的每步斜率
    if direction < 0:
        slope = -slope                            # 左侧 offsets 逆时间 → 转为前向斜率
        cont = values[0] + slope                  # 趋势再向前延续一步
    else:
        cont = values[0] - slope                  # 趋势从右侧回推一步

    med = float(np.median(values))
    mad = float(np.median(np.abs(np.array(values) - med))) * 1.4826
    tol = max(mad, 0.5 * abs(slope), 0.5)
    return cont, tol


def exempt_trend_consistent(power, outlier_mask, n_neighbors=3, max_gap=6):
    """局部突变豁免。

    对每个被标记的异常点：若其取值落在"左侧趋势再延续一步"或"右侧趋势回推一步"
    的容差范围内，说明它只是前后趋势的正常延续（局部突变），不是孤立错值，
    应解除标记。逐点迭代：被解除的点会重新作为邻居参与后续点的趋势估计。
    """
    series = pd.Series(power).reset_index(drop=True)
    mask = pd.Series(outlier_mask).reset_index(drop=True).fillna(False).astype(bool)
    excluded = set(np.where(mask.values)[0].tolist())

    restored = []
    for pos in sorted(excluded):
        val = series.iloc[pos]
        if pd.isna(val):
            continue
        left = _trend_side(series, pos, -1, excluded, n_neighbors, max_gap)
        right = _trend_side(series, pos, +1, excluded, n_neighbors, max_gap)
        consistent = False
        for side in (left, right):
            if side is not None and abs(val - side[0]) <= side[1]:
                consistent = True
                break
        if consistent:
            restored.append(pos)
            excluded.discard(pos)

    new_mask = mask.copy()
    if restored:
        new_mask.iloc[restored] = False
    return new_mask, restored


# =============================================================================
# Step 7: 分时段去趋势 Hampel 异常检测
# =============================================================================
def step7_hampel_filter(df, reporter):
    print("\n" + "="*70)
    print("Step 7: 两级 Hampel 异常检测")
    print("  夜间 (20:00-4:00):  普通 Hampel (6.0σ)")
    print("  白天时段:           两级判断 (小时中位数 → 局部窗口11)")
    print("="*70)
    
    rows_before = len(df)
    
    def check_trend_consistency(val, before_vals, after_vals, threshold):
        """检查值是否符合前后变化趋势
        
        比较当前值与前两个/后两个值的差值方向是否一致。
        例如：如果前两个值逐渐增大，且当前值也在增大范围内，则判定为符合趋势。
        """
        # 需要至少前后各1个值才能判断趋势
        if len(before_vals) < 1 or len(after_vals) < 1:
            return False
        
        # 获取前后各两个值（如果可用）
        b1 = before_vals.iloc[-1] if len(before_vals) >= 1 else np.nan
        b2 = before_vals.iloc[-2] if len(before_vals) >= 2 else np.nan
        a1 = after_vals.iloc[0] if len(after_vals) >= 1 else np.nan
        a2 = after_vals.iloc[1] if len(after_vals) >= 2 else np.nan
        
        # 计算前后差值
        before_diff = b1 - b2 if not (np.isnan(b1) or np.isnan(b2)) else 0
        after_diff = a2 - a1 if not (np.isnan(a2) or np.isnan(a1)) else 0
        
        # 当前值与b1的差值
        current_before_diff = val - b1 if not np.isnan(b1) else 0
        # 当前值与a1的差值
        current_after_diff = a1 - val if not np.isnan(a1) else 0
        
        # 判断趋势一致性
        # 趋势1：值逐渐增大（前后差值为正，且当前值在合理范围内）
        if before_diff > 0 and after_diff > 0:
            # 值在增大，检查当前值是否在合理范围内
            expected_min = b1 - threshold * 0.5  # 允许小幅回落
            expected_max = a1 + threshold * 1.5  # 允许合理增长
            if expected_min <= val <= expected_max:
                return True
        
        # 趋势2：值逐渐减小（前后差值为负）
        elif before_diff < 0 and after_diff < 0:
            expected_min = a1 - threshold * 1.5  # 允许合理下降
            expected_max = b1 + threshold * 0.5  # 允许小幅回升
            if expected_min <= val <= expected_max:
                return True
        
        # 趋势3：值基本稳定（小幅波动）
        elif abs(before_diff) <= threshold * 0.5 and abs(after_diff) <= threshold * 0.5:
            # 值应该在前后值的合理范围内
            neighbor_min = min(b1, a1) - threshold
            neighbor_max = max(b1, a1) + threshold
            if neighbor_min <= val <= neighbor_max:
                return True
        
        return False
    
    def hampel_detect(series, window=13, n_sigma=6.0, min_threshold=0.5):
        """普通 Hampel 异常检测
        
        修复边界效应：要求 min_periods >= window//2+1，
        确保 centered rolling window 在每个点都有足够数据，
        避免数据开头/结尾因窗口不对称导致误判。
        
        修复误判：对接近零的值（|value| < 1.0）不做异常检测，
        因为夜间功率读数接近零，任何微小波动都不应被视为异常。
        
        趋势一致性：对符合前后变化趋势的异常值，判断为正常。
        """
        min_periods = max(window // 2 + 1, window)
        medians = series.rolling(window=window, center=True, min_periods=min_periods).median()
        mad = (series - medians).abs().rolling(window=window, center=True, min_periods=min_periods).median() * 1.4826
        threshold = (n_sigma * mad).clip(lower=min_threshold)
        diff = series - medians
        # 不标记接近零的值为异常（夜间/无功率状态）
        near_zero_mask = series.abs() < 1.0
        outlier_mask = (abs(diff) > threshold) & (~series.isna()) & (~near_zero_mask)
        
        # 趋势一致性检查：恢复符合前后变化趋势的异常值
        flagged_positions = np.where(outlier_mask.values)[0]
        for pos in flagged_positions:
            val = series.iloc[pos]
            if pd.isna(val):
                continue
            
            before_pos = max(0, pos - 2)
            after_pos = min(len(series), pos + 3)
            
            before_vals = series.iloc[before_pos:pos].dropna()
            after_vals = series.iloc[pos+1:after_pos].dropna()
            
            pos_threshold = threshold.iloc[pos] if pos < len(threshold) else min_threshold
            
            if check_trend_consistency(val, before_vals, after_vals, pos_threshold):
                outlier_mask.iloc[pos] = False
        
        return outlier_mask

    def hampel_detect_full(series, window=13, n_sigma=6.0, min_threshold=0.5):
        """返回完整结果的Hampel检测（同上边界修复 + 夜间近零值保护 + 趋势一致性）"""
        min_periods = max(window // 2 + 1, window)
        medians = series.rolling(window=window, center=True, min_periods=min_periods).median()
        mad = (series - medians).abs().rolling(window=window, center=True, min_periods=min_periods).median() * 1.4826
        threshold = (n_sigma * mad).clip(lower=min_threshold)
        diff = series - medians
        # 不标记接近零的值为异常（夜间/无功率状态）
        near_zero_mask = series.abs() < 1.0
        outlier_mask = (abs(diff) > threshold) & (~series.isna()) & (~near_zero_mask)
        
        # 趋势一致性检查：恢复符合前后变化趋势的异常值
        flagged_positions = np.where(outlier_mask.values)[0]
        for pos in flagged_positions:
            val = series.iloc[pos]
            if pd.isna(val):
                continue
            
            before_pos = max(0, pos - 2)
            after_pos = min(len(series), pos + 3)
            
            before_vals = series.iloc[before_pos:pos].dropna()
            after_vals = series.iloc[pos+1:after_pos].dropna()
            
            pos_threshold = threshold.iloc[pos] if pos < len(threshold) else min_threshold
            
            if check_trend_consistency(val, before_vals, after_vals, pos_threshold):
                outlier_mask.iloc[pos] = False
        
        return outlier_mask, medians, mad, threshold, diff
    
    def local_window_median(series, window=11):
        """计算局部窗口中位数趋势
        
        修复边界效应：仅在窗口对称时才计算，
        排除前后数据量差异超过1的边缘点。
        """
        trend = pd.Series(np.nan, index=series.index)
        half_window = window // 2

        for i in range(len(series)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(series), i + half_window + 1)
            window_vals = series.iloc[start_idx:end_idx].dropna()
            before_count = i - start_idx
            after_count = end_idx - i - 1
            # 允许窗口不对称，但前后差距不超过1
            if len(window_vals) >= 3 and abs(before_count - after_count) <= 1:
                trend.iloc[i] = window_vals.median()

        return trend
    
    def two_stage_hampel(series, hours, window=13, n_sigma=6.0, min_threshold=0.5):
        """两级 Hampel 异常检测
        
        第一级：按小时中位数粗检
        第二级：局部窗口11中位数复核（仅对第一级异常值）
        只有两级都判定为异常，才最终标记为异常
        
        修复：夜间部分使用完整连续序列计算滚动窗口，
        避免夜边界（20:00前跳回04:45）导致的误判。
        对温度/湿度等连续物理量尤其重要。
        """
        hour = hours.copy()
        
        # 时段掩码定义
        dawn_mask = (hour >= 5) & (hour < 10)
        noon_mask = (hour >= 10) & (hour < 15)
        evening_mask = (hour >= 15) & (hour < 20)
        night_mask = (hour >= 20) | (hour < 5)
        
        detrend_mask = dawn_mask | noon_mask | evening_mask
        
        outlier_mask = pd.Series(False, index=series.index)
        
        # === 夜间: 使用完整连续序列计算滚动窗口 ===
        if night_mask.sum() > 0:
            # 关键修复：在完整连续序列上计算Hampel统计量，
            # 避免"20:00前跳到04:45"的夜边界误判问题
            night_outliers_full, night_medians_full, night_mad_full, night_threshold_full, night_diff_full = hampel_detect_full(
                series,  # 使用完整连续序列
                window=window, n_sigma=n_sigma, min_threshold=min_threshold
            )
            
            # 仅在夜间行上应用判断
            night_series = series[night_mask].copy()
            night_outliers = night_outliers_full[night_mask].reset_index(drop=True)
            night_medians = night_medians_full[night_mask].reset_index(drop=True)
            night_mad = night_mad_full[night_mask].reset_index(drop=True)
            
            # 第二级：前后值对比（防止跨天数据影响）
            stage2_night_outliers = pd.Series(False, index=night_series.index)
            flagged_positions = np.where(night_outliers.values)[0]
            
            for pos in flagged_positions:
                val = night_series.iloc[pos]
                if pd.isna(val):
                    continue
                
                # 不标记接近零的值为异常（夜间/无功率状态）
                if abs(val) < 1.0:
                    continue
                
                # 获取前后窗口的值（仅在夜间行内）
                before_pos = max(0, pos - 3)
                after_pos = min(len(night_series) - 1, pos + 3)
                
                before_vals = night_series.iloc[before_pos:pos].dropna()
                after_vals = night_series.iloc[pos+1:after_pos+1].dropna()
                
                # 检查前后值是否与当前值一致
                is_consistent_with_neighbors = False
                
                if len(before_vals) > 0 and len(after_vals) > 0:
                    all_neighbors = pd.concat([before_vals, after_vals])
                    neighbor_median = all_neighbors.median()
                    neighbor_mad = abs(all_neighbors - neighbor_median).median() * 1.4826
                    neighbor_threshold = max(n_sigma * neighbor_mad, min_threshold)
                    
                    if abs(val - neighbor_median) <= neighbor_threshold:
                        is_consistent_with_neighbors = True
                    elif (len(before_vals) >= 1 and len(after_vals) >= 1 and
                          abs(val - before_vals.iloc[-1]) < neighbor_threshold / 2 and
                          abs(val - after_vals.iloc[0]) < neighbor_threshold / 2):
                        is_consistent_with_neighbors = True
                
                if is_consistent_with_neighbors:
                    continue
                
                # 趋势一致性检查：若符合前后变化趋势，判定为正常
                pos_threshold = max(n_sigma * night_mad.iloc[pos] if pos < len(night_mad) else 0, min_threshold)
                if check_trend_consistency(val, before_vals, after_vals, pos_threshold):
                    continue
                
                # 局部窗口复核（仅在夜间行内）
                start_pos = max(0, pos - 5)
                end_pos = min(len(night_series), pos + 6)
                local_vals = night_series.iloc[start_pos:end_pos].dropna()
                
                if len(local_vals) >= 3:
                    local_median = local_vals.median()
                    local_mad = abs(local_vals - local_median).median() * 1.4826
                    local_threshold = max(n_sigma * local_mad, min_threshold)
                    
                    if abs(val - local_median) > local_threshold:
                        stage2_night_outliers.iloc[pos] = True
            
            outlier_mask.loc[night_mask] = stage2_night_outliers.values
        
        # === 白天时段: 两级判断 ===
        if detrend_mask.sum() > 0:
            detrend_series = series[detrend_mask].copy()
            detrend_hours = hour[detrend_mask]
            
            # 第一级：按小时中位数去趋势 + Hampel
            hourly_median = detrend_series.groupby(detrend_hours).transform('median')
            detrended_hourly = detrend_series - hourly_median
            
            medians = detrended_hourly.rolling(window=window, center=True, min_periods=1).median()
            mad = (detrended_hourly - medians).abs().rolling(window=window, center=True, min_periods=1).median() * 1.4826
            threshold = (n_sigma * mad).clip(lower=min_threshold)
            diff = detrended_hourly - medians
            stage1_outliers = (abs(diff) > threshold) & (~detrended_hourly.isna())
            
            # 第二级：局部窗口复核 + 前后值对比
            stage2_outliers = pd.Series(False, index=detrend_series.index)
            
            flagged_positions = np.where(stage1_outliers.values)[0]
            
            for pos in flagged_positions:
                val = detrend_series.iloc[pos]
                if pd.isna(val):
                    continue
                
                # 不标记接近零的值为异常（夜间/无功率状态）
                if abs(val) < 1.0:
                    continue
                
                # 获取前后窗口的值
                before_pos = max(0, pos - 3)
                after_pos = min(len(detrend_series) - 1, pos + 4)
                
                before_vals = detrend_series.iloc[before_pos:pos].dropna()
                after_vals = detrend_series.iloc[pos+1:after_pos].dropna()
                
                # 检查前后值是否与当前值一致
                is_consistent_with_neighbors = False
                
                if len(before_vals) > 0 and len(after_vals) > 0:
                    all_neighbors = pd.concat([before_vals, after_vals])
                    neighbor_median = all_neighbors.median()
                    neighbor_mad = abs(all_neighbors - neighbor_median).median() * 1.4826
                    neighbor_threshold = max(n_sigma * neighbor_mad, min_threshold)
                    
                    # 如果当前值与邻居中位数差异不大，认为是正常的
                    if abs(val - neighbor_median) <= neighbor_threshold:
                        is_consistent_with_neighbors = True
                    # 或者前后值都存在且与当前值非常接近
                    elif (len(before_vals) >= 1 and len(after_vals) >= 1 and
                          abs(val - before_vals.iloc[-1]) < neighbor_threshold / 2 and
                          abs(val - after_vals.iloc[0]) < neighbor_threshold / 2):
                        is_consistent_with_neighbors = True
                
                if is_consistent_with_neighbors:
                    continue  # 前后一致，判定为正常
                
                # 趋势一致性检查：若符合前后变化趋势，判定为正常
                pos_threshold = max(n_sigma * mad.iloc[pos] if pos < len(mad) else 0, min_threshold)
                if check_trend_consistency(val, before_vals, after_vals, pos_threshold):
                    continue
                
                # 前后值不一致，进行局部窗口复核
                start_pos = max(0, pos - 5)
                end_pos = min(len(detrend_series), pos + 6)
                local_vals = detrend_series.iloc[start_pos:end_pos].dropna()
                
                if len(local_vals) >= 3:
                    local_median = local_vals.median()
                    local_mad = abs(local_vals - local_median).median() * 1.4826
                    local_threshold = max(n_sigma * local_mad, min_threshold)
                    
                    if abs(val - local_median) > local_threshold:
                        stage2_outliers.iloc[pos] = True
            
            outlier_mask.loc[detrend_mask] = stage2_outliers.values
        
        return outlier_mask
    
    # =============================================================================
    # 天气特征异常检测：已停用
    # =============================================================================
    # 原因（基于 data/processed/Bozhou_1_before_hampel.csv 的实测分析）：
    #   1. temperature_c / relative_humidity_pct / ghi_wm2 在该数据集中本就是
    #      平滑连续量，Hampel 检出数极少（sigma=3 下分别为 0 / 17 / 0 个），
    #      去除它们不会带来任何收益，反而因插值引入额外噪声。
    #   2. cloud_cover_pct 是 Open-Meteo NWP 云量场的直接输出，天然呈
    #      0/100 两端聚集 + 15 分钟阶梯状（51% 的值落在 0 或 100，65% 的步长 Δ=0），
    #      13.5% 的步长 |Δ|>=20。这些"突变"是模式云量诊断的真实结构，
    #      且与同一次 API 响应的降水/湿度/weather_code 完全自洽，不是脏数据。
    #      对其做 Hampel 会把真实的对流云进出台风误判为异常。
    #   3. ghi_wm2 在本数据集中是无云衰减的纯晴空曲线（同一 solar_alt 下
    #      clouds<=5 与 clouds>=95 的 GHI 中位数仅差 ~0.5 W/m2），本身就是
    #      确定性时间函数，做异常检测无意义。
    #
    # 因此：仅对 power_kw 做异常检测，天气特征原样保留。
    weather_features = []          # 保留结构以便未来按需恢复
    min_threshold_map = {}         # 保留结构以便未来按需恢复
    total_outliers = 0

    hour = df['timestamp'].dt.hour
    print("  Weather features: Hampel 检测已停用（仅功率做异常检测）")
    print(f"  -> temperature_c, relative_humidity_pct, cloud_cover_pct, ghi_wm2 全部原样保留")

    # 功率异常检测
    dawn_mask = (hour >= 5) & (hour < 10)
    noon_mask = (hour >= 10) & (hour < 15)
    evening_mask = (hour >= 15) & (hour < 20)
    detrend_power_mask = dawn_mask | noon_mask | evening_mask
    night_mask = ~detrend_power_mask
    
    n_power_outliers = 0
    power_outlier_mask = pd.Series(False, index=df.index)
    
    if HAMPEL_METHOD == "simple":
        # ===== Simple Hampel: 直接对原始功率应用Hampel检测 =====
        print(f"  Power anomalies [simple {HAMPEL_POWER_SIGMA}σ]:")
        
        # 白天: 先去小时趋势再检测
        if detrend_power_mask.sum() > 0:
            power_series = df.loc[detrend_power_mask, 'power_kw'].copy()
            power_hours = hour[detrend_power_mask]
            
            hourly_median = power_series.groupby(power_hours).transform('median')
            power_detrended = power_series - hourly_median
            
            outliers = hampel_detect(power_detrended, window=9, n_sigma=HAMPEL_POWER_SIGMA, min_threshold=0.5)
            n_outliers = outliers.sum()
            if n_outliers > 0:
                power_outlier_mask.loc[outliers.index] = outliers.values
                n_power_outliers += n_outliers
                print(f"    power_kw (daytime): {n_outliers:,} anomalies")
        
        # 夜间: 直接检测
        if night_mask.sum() > 0:
            night_power = df.loc[night_mask, 'power_kw'].copy()
            night_outliers = hampel_detect(night_power, window=9, n_sigma=HAMPEL_POWER_SIGMA, min_threshold=0.5)
            n_night_outliers = night_outliers.sum()
            if n_night_outliers > 0:
                power_outlier_mask.loc[night_outliers.index] = night_outliers.values
                n_power_outliers += n_night_outliers
                print(f"    power_kw (night): {n_night_outliers:,} anomalies")
    
    else:
        # ===== Two-stage Hampel: 两级判断 =====
        print(f"  Power anomalies [two-stage {HAMPEL_POWER_SIGMA}σ]:")
        
        # 白天功率两级判断
        if detrend_power_mask.sum() > 0:
            power_series = df.loc[detrend_power_mask, 'power_kw'].copy()
            power_hours = hour[detrend_power_mask]
            
            # 第一级：按小时中位数
            hourly_median = power_series.groupby(power_hours).transform('median')
            power_detrended = power_series - hourly_median
            
            power_medians = power_detrended.rolling(window=9, center=True, min_periods=1).median()
            power_mad = (power_detrended - power_medians).abs().rolling(window=9, center=True, min_periods=1).median() * 1.4826
            power_threshold = (HAMPEL_POWER_SIGMA * power_mad).clip(lower=0.5)
            power_diff = power_detrended - power_medians
            stage1_power_outliers = (abs(power_diff) > power_threshold) & (~power_detrended.isna())
            
            # 第二级：局部窗口复核 + 前后值对比
            stage2_power_outliers = pd.Series(False, index=power_series.index)
            flagged_positions = np.where(stage1_power_outliers.values)[0]
            
            for pos in flagged_positions:
                val = power_series.iloc[pos]
                if pd.isna(val):
                    continue
                
                before_pos = max(0, pos - 3)
                after_pos = min(len(power_series) - 1, pos + 3)
                
                before_vals = power_series.iloc[before_pos:pos].dropna()
                after_vals = power_series.iloc[pos+1:after_pos+1].dropna()
                
                is_consistent_with_neighbors = False
                
                if len(before_vals) > 0 and len(after_vals) > 0:
                    all_neighbors = pd.concat([before_vals, after_vals])
                    neighbor_median = all_neighbors.median()
                    neighbor_mad = abs(all_neighbors - neighbor_median).median() * 1.4826
                    neighbor_threshold = max(HAMPEL_POWER_SIGMA * neighbor_mad, 0.5)
                    
                    if abs(val - neighbor_median) <= neighbor_threshold:
                        is_consistent_with_neighbors = True
                    elif (len(before_vals) >= 1 and len(after_vals) >= 1 and
                          abs(val - before_vals.iloc[-1]) < neighbor_threshold / 2 and
                          abs(val - after_vals.iloc[0]) < neighbor_threshold / 2):
                        is_consistent_with_neighbors = True
                
                if is_consistent_with_neighbors:
                    continue
                
                start_pos = max(0, pos - 5)
                end_pos = min(len(power_series), pos + 6)
                local_vals = power_series.iloc[start_pos:end_pos].dropna()
                
                if len(local_vals) >= 3:
                    local_median = local_vals.median()
                    local_mad = abs(local_vals - local_median).median() * 1.4826
                    local_threshold = max(HAMPEL_POWER_SIGMA * local_mad, 0.5)
                    
                    if abs(val - local_median) > local_threshold:
                        stage2_power_outliers.iloc[pos] = True
            
            n_power_outliers = stage2_power_outliers.sum()
            if n_power_outliers > 0:
                power_outlier_mask.loc[stage2_power_outliers.index] = stage2_power_outliers.values
                print(f"    power_kw (daytime): {n_power_outliers:,} anomalies")
        
        # 夜间功率普通 Hampel
        if night_mask.sum() > 0:
            night_power = df.loc[night_mask, 'power_kw'].copy()
            night_outliers = hampel_detect(night_power, window=9, n_sigma=HAMPEL_POWER_SIGMA, min_threshold=0.5)
            n_night_outliers = night_outliers.sum()
            if n_night_outliers > 0:
                power_outlier_mask.loc[night_outliers.index] = night_outliers.values
                n_power_outliers += n_night_outliers
                print(f"    power_kw (night): {n_night_outliers:,} anomalies")
    
    # ===== 局部突变豁免：符合前后趋势续接的点解除标记 =====
    n_flagged_before = int(power_outlier_mask.sum())
    power_outlier_mask, restored = exempt_trend_consistent(df['power_kw'], power_outlier_mask)
    if restored:
        print(f"  局部突变豁免（符合前后趋势续接，非孤立错值）: {len(restored)} 个 → 解除标记")
        for p in restored:
            ts_val = df['timestamp'].iloc[p]
            print(f"      {ts_val}  原值 {df['power_kw'].iloc[p]:.2f} kW")
    n_power_outliers = int(power_outlier_mask.sum())
    if n_power_outliers > 0:
        df.loc[power_outlier_mask, 'power_kw'] = np.nan
    print(f"  功率异常点: {n_flagged_before} → {n_power_outliers}（豁免 {n_flagged_before - n_power_outliers}）")
    
    method_label = "单阶段Hampel" if HAMPEL_METHOD == "simple" else "两级Hampel"
    reporter.report(7, f"{method_label}异常检测", rows_before, len(df), f"仅功率检测: {n_power_outliers} power anomalies (天气特征未检测)")
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
        'linear': ['ghi_wm2', 'power_kw'],
        
        # 样条插值：周期性连续特征
        'spline': ['temperature_c', 'relative_humidity_pct'],
        
        # 前后向填充：离散/突变特征
        'ffill': ['cloud_cover_pct'],
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
# Step 9: 长缺失剖面回填 - 仅处理连续缺失 >4 的段
# =============================================================================
def step9_long_gap_fill(df, reporter):
    print("\n" + "="*70)
    print("Step 9: 长缺失剖面回填（仅处理连续缺失 >4 的段）")
    print("="*70)
    
    rows_before = len(df)
    
    cols_to_fill = ['temperature_c', 'ghi_wm2', 'cloud_cover_pct', 
                   'relative_humidity_pct', 'power_kw']
    
    df['month'] = df['timestamp'].dt.month
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['dayofyear'] = df['timestamp'].dt.dayofyear
    df['daytype'] = df['timestamp'].dt.dayofweek  # 0=周一, 6=周日
    
    total_filled = 0
    total_long_gaps = 0
    
    for col in cols_to_fill:
        if col not in df.columns:
            continue
        
        nan_before = df[col].isna().sum()
        
        # 找出连续缺失 >4 的长缺失段索引
        is_nan = df[col].isna()
        group_id = (~is_nan).cumsum()
        nan_groups = df[is_nan].groupby(group_id[is_nan]).size()
        long_gaps = nan_groups[nan_groups > 4]
        
        n_long_gaps = len(long_gaps)
        total_long_gaps += n_long_gaps
        
        if n_long_gaps == 0:
            print(f"  {col}: 无长缺失段，跳过")
            continue
        
        # 找出所有长缺失段的位置
        long_gap_indices = []
        for group_key in long_gaps.index:
            # 获取该组的索引
            group_indices = df.index[group_id[df.index] == group_key]
            long_gap_indices.extend(group_indices.tolist())
        
        # 只对这些长缺失位置应用剖面回填
        long_gap_mask = df.index.isin(long_gap_indices)
        
        # 计算分层中位数
        # 1. 月+小时+分钟 中位数
        median_mhm = df.groupby(['month', 'hour', 'minute'])[col].transform('median')
        # 2. 月+小时 中位数
        median_mh = df.groupby(['month', 'hour'])[col].transform('median')
        # 3. 小时+分钟 中位数
        median_hm = df.groupby(['hour', 'minute'])[col].transform('median')
        # 4. 星期+小时 中位数
        median_dh = df.groupby(['daytype', 'hour'])[col].transform('median')
        # 5. 小时 中位数
        median_h = df.groupby(['hour'])[col].transform('median')
        # 6. 全局中位数
        global_median = df[col].median()
        
        # 在长缺失位置填充：按优先级选择第一个非NaN的中位数
        for idx in long_gap_indices:
            if pd.isna(df.at[idx, col]):
                filled_val = None
                # 按优先级尝试不同的中位数
                if not pd.isna(median_mhm.at[idx]):
                    filled_val = median_mhm.at[idx]
                elif not pd.isna(median_mh.at[idx]):
                    filled_val = median_mh.at[idx]
                elif not pd.isna(median_hm.at[idx]):
                    filled_val = median_hm.at[idx]
                elif not pd.isna(median_dh.at[idx]):
                    filled_val = median_dh.at[idx]
                elif not pd.isna(median_h.at[idx]):
                    filled_val = median_h.at[idx]
                else:
                    filled_val = global_median
                
                df.at[idx, col] = filled_val
        
        nan_after = df[col].isna().sum()
        filled = nan_before - nan_after
        total_filled += filled
        print(f"  {col}: 填充 {n_long_gaps} 个长缺失段, {filled} points")
    
    # 删除临时列
    df = df.drop(columns=['month', 'hour', 'minute', 'dayofyear', 'daytype'])
    
    print("  Long gap fill: {0:,} points, {1} segments".format(total_filled, total_long_gaps))
    reporter.report(9, "长缺失剖面回填", rows_before, len(df), "Filled {0} points in {1} long segments".format(total_filled, total_long_gaps))
    return df
    
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
# Step 11: 调度/预测衍生特征构建
# =============================================================================
def step11_derived_features(df, reporter):
    print("\n" + "="*70)
    print("Step 12: 调度/预测衍生特征构建")
    print("="*70)
    
    rows_before = len(df)
    
    # 归一化功率
    df['power_pu'] = df['power_kw'] / CAPACITY_KW
    
    # 功率变化
    df['power_ramp_15m_pu'] = df['power_pu'].diff()
    
    # 小时特征
    df['hour'] = df['timestamp'].dt.hour
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # 删除临时列
    if 'hour' in df.columns:
        df = df.drop(columns=['hour'])
    
    print("  New features: power_pu, power_ramp_15m_pu, sin/cos_hour")
    reporter.report(11, "衍生特征构建", rows_before, len(df), "Added 3 derived features")
    return df

# =============================================================================
# 辅助函数：清理旧的CSV文件
# =============================================================================
def cleanup_old_csvs():
    """清理 processed 目录下旧的CSV文件，但保留异常检测前的基线CSV"""
    import os
    import re
    
    # 要删除的文件列表
    files_to_delete = []
    
    # 要保留的基线文件名
    baseline_files = ['Bozhou_1_before_hampel.csv']
    
    # 保留最新的 before_interpolation 文件（按时间戳）
    latest_before_interp = {}
    
    if PROCESSED_DIR.exists():
        for f in PROCESSED_DIR.glob('*.csv'):
            # 保留基线文件
            if f.name in baseline_files:
                continue
            
            # 处理 before_interpolation 文件：只保留最新的
            if 'before_interpolation' in f.name:
                # 提取sigma版本和时间戳
                match = re.match(r'Bozhou_1_before_interpolation_sigma(\d+)(?:_(\d+))?\.csv', f.name)
                if match:
                    sigma = match.group(1)
                    ts = match.group(2) or '000000'  # 无时间戳的给最小值
                    key = f'before_interpolation_sigma{sigma}'
                    if key not in latest_before_interp or ts > latest_before_interp[key][1]:
                        # 记录新的最新文件
                        if key in latest_before_interp:
                            files_to_delete.append(latest_before_interp[key][0])  # 删除旧的
                        latest_before_interp[key] = (f, ts)
                    else:
                        files_to_delete.append(f)  # 这个不是最新的，删除
                continue
            
            # 保留所有 preprocessed 配置文件（不同sigma阈值的结果）
            if 'preprocessed' in f.name:
                continue  # 保留所有配置的preprocessed文件
            files_to_delete.append(f)
    
    deleted_count = 0
    for f in files_to_delete:
        try:
            f.unlink()
            deleted_count += 1
            print(f"  [删除] {f.name}")
        except Exception as e:
            print(f"  [跳过] {f.name} ({e})")
    
    # 打印保留的基线文件
    print("  [保留] 异常检测前基线文件:")
    for key, (f, ts) in latest_before_interp.items():
        print(f"    {f.name}")
    for bf in baseline_files:
        bf_path = PROCESSED_DIR / bf
        if bf_path.exists():
            print(f"    {bf}")
    
    if deleted_count > 0:
        print(f"  共删除 {deleted_count} 个旧文件")
    
    return deleted_count

# =============================================================================
# 主函数
# =============================================================================
def main():
    print("="*70)
    print("亳州光伏数据集 Step1 预处理")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"配置: Hampel方法={HAMPEL_METHOD}, Sigma={HAMPEL_WEATHER_SIGMA}")
    print("="*70)
    
    # 清理旧的CSV文件（保留异常检测前的基线）
    print("\n[清理] 清除旧的CSV文件...")
    cleanup_old_csvs()
    
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
    
    # Step 6: 辐照-功率物理一致性修正
    df = step6_irradiance_consistency(df, reporter)
    
    # Step 7: 分时段去趋势 Hampel 异常检测
    df = step7_hampel_filter(df, reporter)
    
    # 保存插值前数据（带NULL缺失标记）
    ts = datetime.now().strftime('%H%M%S')
    before_interp_path = PROCESSED_DIR / f"Bozhou_1_before_interpolation_{CFG_LABEL}_{ts}.csv"
    before_interp_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(before_interp_path, index=False)
    print(f"\n  [保存] 插值前数据: {before_interp_path.name}")
    
    # Step 8: 短时缺失插值
    df = step8_short_interpolation(df, reporter)
    
    # Step 9: 长缺失剖面回填
    df = step9_long_gap_fill(df, reporter)
    
    # Step 10: 白天/夜间分离
    df = step10_day_night_separation(df, reporter)
    
    # Step 11: 衍生特征构建
    df = step11_derived_features(df, reporter)
    
    # Save results - 使用配置特定的路径
    output_path = PROCESSED_DIR / f"Bozhou_1_preprocessed_{CFG_LABEL}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
    
    # 导出路径供外部使用
    return df, reporter, output_path

if __name__ == "__main__":
    df, reporter, output_path = main()
