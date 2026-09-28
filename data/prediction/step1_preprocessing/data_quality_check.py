"""
数据质量检查与处理脚本
整合数据质量检查、辐照-功率一致性分析、异常值检测与处理功能

主要功能：
1. 数据质量检查与评分
2. 辐照-功率一致性分析
3. 异常值检测与处理
4. 生成处理报告
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import sys
import io

# 设置stdout编码为UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')


# =============================================================================
# 第一部分：数据质量检查
# =============================================================================

def check_data_quality(file_path, df=None):
    """
    执行全面的数据质量检查
    
    Args:
        file_path: 数据文件路径
        df: 可选，预加载的DataFrame
    Returns:
        dict: 各项评分
    """
    
    print("=" * 70)
    print("数据集质量检查报告")
    print("=" * 70)
    
    # 读取数据
    if df is None:
        df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    scores = {}
    
    # 1. 基本信息
    print("\n【1. 基本信息】")
    print(f"  文件路径: {file_path}")
    print(f"  数据行数: {len(df):,}")
    print(f"  数据列数: {len(df.columns)}")
    print(f"  时间范围: {df['timestamp'].min()} 至 {df['timestamp'].max()}")
    print(f"  时间跨度: {(df['timestamp'].max() - df['timestamp'].min()).days + 1} 天")
    
    # 2. 缺失值检查
    print("\n【2. 缺失值检查】")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(4)
    has_missing = False
    for col in df.columns:
        if missing[col] > 0:
            print(f"  {col}: {missing[col]:,} 个缺失 ({missing_pct[col]:.2f}%)")
            has_missing = True
    if not has_missing:
        print("  [OK] 所有列无缺失值")
    
    # 3. 时间连续性检查
    print("\n【3. 时间序列连续性】")
    expected_interval = pd.Timedelta(minutes=15)
    df['time_diff'] = df['timestamp'].diff()
    gaps = df[df['time_diff'] != expected_interval]['time_diff'].dropna()
    if len(gaps) > 0:
        print(f"  [!] 发现 {len(gaps)} 个时间间隔异常")
    else:
        print("  [OK] 所有时间间隔为15分钟")
    
    # 4. 日间/夜间分布
    print("\n【4. 日间/夜间分布】")
    daytime_count = df['is_daytime'].sum()
    nighttime_count = len(df) - daytime_count
    print(f"  日间 (is_daytime=1): {daytime_count:,} ({daytime_count/len(df)*100:.1f}%)")
    print(f"  夜间 (is_daytime=0): {nighttime_count:,} ({nighttime_count/len(df)*100:.1f}%)")
    
    # 5. 功率分布检查
    print("\n【5. 功率分布】")
    power_stats = df['power_kw'].describe()
    print(f"  最小值: {power_stats['min']:.4f} kW")
    print(f"  最大值: {power_stats['max']:.4f} kW")
    print(f"  均值: {power_stats['mean']:.4f} kW")
    print(f"  中位数: {power_stats['50%']:.4f} kW")
    
    # 6. 物理边界检查
    print("\n【6. 物理边界检查】")
    bounds_check = {
        'temperature_c': (-40, 50),
        'relative_humidity_pct': (0, 100),
        'visibility_km': (0, 50),
        'cloud_cover_pct': (0, 100),
        'ghi_wm2': (0, 1500),
        'power_kw': (0, 500),
        'power_pu': (0, 1.2)
    }
    
    all_bounds_ok = True
    bounds_violations = 0
    for col, (min_val, max_val) in bounds_check.items():
        if col in df.columns:
            violations = ((df[col] < min_val) | (df[col] > max_val)).sum()
            bounds_violations += violations
            if violations > 0:
                print(f"  [!] {col}: {violations} 个值超出边界 [{min_val}, {max_val}]")
                all_bounds_ok = False
    
    if all_bounds_ok:
        print("  [OK] 所有值在物理边界内")
    
    # 7. 数据质量分数检查
    print("\n【7. 数据质量分数】")
    if 'data_quality_score' in df.columns:
        dq_score = df['data_quality_score'].describe()
        print(f"  均值: {dq_score['mean']:.4f}")
        print(f"  最小值: {dq_score['min']:.4f}")
        print(f"  最大值: {dq_score['max']:.4f}")
    
    # 8. 特征相关性检查（与功率）
    print("\n【8. 特征与功率相关性】")
    numeric_features = ['temperature_c', 'relative_humidity_pct', 'visibility_km',
                       'cloud_cover_pct', 'ghi_wm2', 'uv_index', 'precip_rate_mmhr']
    correlations = {}
    for feat in numeric_features:
        if feat in df.columns:
            corr = df[feat].corr(df['power_kw'])
            correlations[feat] = corr
    
    # 按相关性排序
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for feat, corr in sorted_corr:
        print(f"  {feat}: r = {corr:.4f}")
    
    # 9. 每日数据完整性
    print("\n【9. 每日数据点数】")
    df['date'] = df['timestamp'].dt.date
    daily_counts = df.groupby('date').size()
    expected_per_day = 96  # 15分钟间隔，每天96个点
    
    incomplete_days = daily_counts[daily_counts != expected_per_day]
    complete_days = len(daily_counts) - len(incomplete_days)
    total_days = len(daily_counts)
    
    if len(incomplete_days) > 0:
        print(f"  [!] 发现 {len(incomplete_days)} 天数据不完整")
        print(f"  完整天数: {complete_days}")
    else:
        print(f"  [OK] 所有 {total_days} 天数据完整 (每天 {expected_per_day} 个点)")
    
    # 10. 异常值检测（基于IQR）
    print("\n【10. 异常值检测 (IQR方法)】")
    for col in ['power_kw', 'ghi_wm2', 'temperature_c', 'cloud_cover_pct']:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = ((df[col] < lower) | (df[col] > upper)).sum()
            pct = outliers / len(df) * 100
            print(f"  {col}: {outliers} 个异常值 ({pct:.2f}%)")
    
    return scores


def score_data_quality(df):
    """
    对数据进行质量评分
    
    Args:
        df: 数据集
    Returns:
        tuple: (总分, 各项评分字典, 评级)
    """
    
    print("\n" + "=" * 70)
    print("数据质量评分")
    print("=" * 70)
    
    scores = {}
    total_score = 100
    
    # 1. 时间完整性评分 (20分)
    print("\n【1. 时间完整性评分】(满分20分)")
    daily_counts = df.groupby(df['timestamp'].dt.date).size()
    expected_per_day = 96
    complete_days = (daily_counts == expected_per_day).sum()
    total_days = len(daily_counts)
    completeness_ratio = complete_days / total_days if total_days > 0 else 0
    time_score = completeness_ratio * 20
    scores['时间完整性'] = time_score
    print(f'  完整天数: {complete_days}/{total_days}')
    print(f'  得分: {time_score:.1f}/20')
    
    # 2. 缺失值评分 (15分)
    print("\n【2. 数据缺失评分】(满分15分)")
    missing_ratio = df.drop(['timestamp', 'date'], axis=1, errors='ignore').isnull().sum().sum() / (len(df) * (len(df.columns) - 2))
    missing_score = (1 - missing_ratio) * 15
    scores['数据完整性'] = missing_score
    print(f'  缺失率: {missing_ratio*100:.2f}%')
    print(f'  得分: {missing_score:.1f}/15')
    
    # 3. 物理边界评分 (15分)
    print("\n【3. 物理边界评分】(满分15分)")
    bounds_violations = 0
    bounds_check = [
        ('temperature_c', -40, 50),
        ('relative_humidity_pct', 0, 100),
        ('visibility_km', 0, 50),
        ('cloud_cover_pct', 0, 100),
        ('ghi_wm2', 0, 1500),
        ('power_kw', 0, 500),
        ('power_pu', 0, 1.2)
    ]
    for col, min_val, max_val in bounds_check:
        if col in df.columns:
            violations = ((df[col] < min_val) | (df[col] > max_val)).sum()
            bounds_violations += violations
    bounds_score = max(0, 15 - bounds_violations * 0.01)
    scores['物理边界'] = bounds_score
    print(f'  边界违规数: {bounds_violations}')
    print(f'  得分: {bounds_score:.1f}/15')
    
    # 4. 辐照-功率一致性评分 (25分)
    print("\n【4. 辐照-功率一致性评分】(满分25分)")
    daytime_high_ghi = df[(df['is_daytime'] == 1) & (df['ghi_wm2'] > 100)]
    if len(daytime_high_ghi) > 0:
        zero_power_count = (daytime_high_ghi['power_kw'] == 0).sum()
        zero_power_ratio = zero_power_count / len(daytime_high_ghi)
    else:
        zero_power_ratio = 0
    
    nighttime = df[df['is_daytime'] == 0]
    nighttime_nonzero = (nighttime['power_kw'] > 0.5).sum() if len(nighttime) > 0 else 0
    nighttime_ratio = nighttime_nonzero / len(nighttime) if len(nighttime) > 0 else 0
    
    valid_daytime = df[(df['is_daytime'] == 1) & (df['ghi_wm2'] > 100) & (df['power_kw'] > 0)]
    if len(valid_daytime) > 0:
        peak_power = 450
        efficiency = valid_daytime['power_kw'] / (valid_daytime['ghi_wm2'] / 1000 * peak_power) * 100
        low_eff_ratio = (efficiency < 5).sum() / len(valid_daytime)
    else:
        low_eff_ratio = 0
    
    consistency_score = 25 * (1 - zero_power_ratio * 10 - nighttime_ratio * 5 - low_eff_ratio * 3)
    consistency_score = max(0, consistency_score)
    scores['辐照-功率一致性'] = consistency_score
    print(f'  日间零功率比例: {zero_power_ratio*100:.3f}%')
    print(f'  夜间非零比例: {nighttime_ratio*100:.3f}%')
    print(f'  低效率点比例: {low_eff_ratio*100:.2f}%')
    print(f'  得分: {consistency_score:.1f}/25')
    
    # 5. 数据分布评分 (15分)
    print("\n【5. 数据分布评分】(满分15分)")
    power_mean = df['power_kw'].mean()
    power_std = df['power_kw'].std()
    mean_valid = 1 if 30 < power_mean < 300 else 0.5
    std_valid = 1 if 30 < power_std < 150 else 0.5
    distribution_score = (mean_valid + std_valid) / 2 * 15
    scores['数据分布'] = distribution_score
    print(f'  功率均值: {power_mean:.2f} kW')
    print(f'  功率标准差: {power_std:.2f} kW')
    print(f'  得分: {distribution_score:.1f}/15')
    
    # 6. 质量分数评分 (10分)
    print("\n【6. 内置质量分数评分】(满分10分)")
    if 'data_quality_score' in df.columns:
        dq_mean = df['data_quality_score'].mean()
        dq_score = dq_mean * 10
    else:
        dq_mean = 1.0
        dq_score = 10.0
    scores['内置质量分数'] = dq_score
    print(f'  平均质量分数: {dq_mean:.4f}')
    print(f'  得分: {dq_score:.1f}/10')
    
    # 总分
    total = sum(scores.values())
    
    print('\n' + '=' * 70)
    print('评分汇总')
    print('=' * 70)
    for name, score in scores.items():
        print(f'  {name}: {score:.1f}/100')
    print(f'\n  总分: {total:.1f}/100')
    
    # 评级
    if total >= 90:
        grade = 'A (优秀)'
    elif total >= 80:
        grade = 'B (良好)'
    elif total >= 70:
        grade = 'C (合格)'
    elif total >= 60:
        grade = 'D (勉强合格)'
    else:
        grade = 'F (不合格)'
    print(f'  评级: {grade}')
    print('=' * 70)
    
    return total, scores, grade


# =============================================================================
# 第二部分：辐照-功率一致性分析
# =============================================================================

def analyze_irradiance_power_consistency(df):
    """详细分析辐照-功率一致性"""
    
    print("\n" + "=" * 70)
    print("辐照-功率一致性分析")
    print("=" * 70)
    
    results = {}
    
    # 1. 日间高辐照零功率点分析
    print("\n【1. 日间高辐照零功率点分析】")
    daytime = df[df['is_daytime'] == 1].copy()
    
    ghi_thresholds = [50, 100, 200, 300, 500]
    
    for threshold in ghi_thresholds:
        high_ghi = daytime[daytime['ghi_wm2'] > threshold]
        if len(high_ghi) > 0:
            zero_power = high_ghi[high_ghi['power_kw'] == 0]
            pct = len(zero_power) / len(high_ghi) * 100
            print(f"  GHI > {threshold} W/m²: 共 {len(high_ghi)} 个点, 零功率 {len(zero_power)} 个 ({pct:.2f}%)")
            results[f'zero_power_ghi_{threshold}'] = {
                'total': len(high_ghi),
                'zero_count': len(zero_power),
                'percentage': pct
            }
    
    # 2. 详细查看日间零功率异常点
    print("\n【2. 日间零功率异常点详情】")
    daytime_high_ghi = daytime[(daytime['ghi_wm2'] > 100) & (daytime['power_kw'] == 0)]
    
    if len(daytime_high_ghi) > 0:
        print(f"  发现 {len(daytime_high_ghi)} 个日间高辐照零功率异常点")
    else:
        print("  [OK] 无日间高辐照零功率异常点")
    
    # 3. 辐照-功率比率分析
    print("\n【3. 辐照-功率比率分析】")
    valid_power = daytime[(daytime['ghi_wm2'] > 100) & (daytime['power_kw'] > 0)]
    
    if len(valid_power) > 0:
        peak_power = 450
        efficiency = valid_power['power_kw'] / (valid_power['ghi_wm2'] / 1000 * peak_power) * 100
        
        print(f"  有效样点数: {len(valid_power)}")
        print(f"  转换效率统计:")
        print(f"    最小: {efficiency.min():.2f}%")
        print(f"    最大: {efficiency.max():.2f}%")
        print(f"    均值: {efficiency.mean():.2f}%")
        
        low_efficiency = valid_power[efficiency < 5]
        if len(low_efficiency) > 0:
            print(f"\n  [!] 发现 {len(low_efficiency)} 个低效率点 (<5%)")
            results['low_efficiency_count'] = len(low_efficiency)
    
    # 4. 功率突变检测
    print("\n【4. 功率突变检测】")
    df['power_change'] = df['power_kw'].diff().abs()
    
    change_stats = df['power_change'].describe()
    print(f"  15分钟功率变化统计:")
    print(f"    均值: {change_stats['mean']:.2f} kW")
    print(f"    最大: {change_stats['max']:.2f} kW")
    
    threshold_3std = df['power_change'].mean() + 3 * df['power_change'].std()
    sudden_changes = df[df['power_change'] > threshold_3std]
    print(f"\n  阈值 (均值+3σ): {threshold_3std:.2f} kW")
    print(f"  异常突变点数: {len(sudden_changes)}")
    
    if len(sudden_changes) > 0:
        results['sudden_change_count'] = len(sudden_changes)
    
    if 'power_change' in df.columns:
        df.drop('power_change', axis=1, inplace=True)
    
    return results, daytime_high_ghi


def detect_low_efficiency_points(df, min_efficiency_threshold=5.0):
    """
    检测低效率点：日间辐照充足但功率异常低
    """
    daytime = df[df['is_daytime'] == 1].copy()
    valid = daytime[(daytime['ghi_wm2'] > 100) & (daytime['power_kw'] > 0)].copy()
    
    if len(valid) == 0:
        return pd.Index([])
    
    peak_power = 450
    efficiency = valid['power_kw'] / (valid['ghi_wm2'] / 1000 * peak_power) * 100
    
    low_eff_mask = efficiency < min_efficiency_threshold
    low_eff_indices = valid[low_eff_mask].index
    
    print(f"\n  [低效率点检测] 阈值: {min_efficiency_threshold}%")
    print(f"    日间有效点总数: {len(valid)}")
    print(f"    低效率点数量: {len(low_eff_indices)}")
    print(f"    低效率点占比: {len(low_eff_indices)/len(valid)*100:.2f}%")
    
    return low_eff_indices


# =============================================================================
# 第三部分：异常值检测
# =============================================================================

def detect_outliers(df):
    """基于多种方法的异常值检测"""
    
    print("\n" + "=" * 70)
    print("异常值检测分析")
    print("=" * 70)
    
    outlier_results = {}
    
    # 1. Z-Score方法
    print("\n【1. Z-Score方法 (|z| > 3)】")
    from scipy import stats
    
    for col in ['power_kw', 'ghi_wm2', 'temperature_c']:
        if col in df.columns:
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outliers = (z_scores > 3).sum()
            pct = outliers / len(df) * 100
            print(f"  {col}: {outliers} 个异常值 ({pct:.2f}%)")
            outlier_results[f'{col}_zscore'] = outliers
    
    # 2. IQR方法
    print("\n【2. IQR方法 (Q1-1.5*IQR, Q3+1.5*IQR)】")
    for col in ['power_kw', 'ghi_wm2', 'temperature_c']:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = ((df[col] < lower) | (df[col] > upper)).sum()
            pct = outliers / len(df) * 100
            print(f"  {col}: {outliers} 个异常值 ({pct:.2f}%)")
            outlier_results[f'{col}_iqr'] = outliers
    
    # 3. 功率超限检测
    print("\n【3. 功率超限检测】")
    daytime = df[df['is_daytime'] == 1]
    peak_power = 450
    max_expected_power = (daytime['ghi_wm2'] / 1000) * peak_power * 1.1
    
    over_power = daytime[daytime['power_kw'] > max_expected_power]
    if len(over_power) > 0:
        print(f"  [!] 发现 {len(over_power)} 个功率超过预期最大值的点")
        outlier_results['power_over_expected'] = len(over_power)
    else:
        print(f"  [OK] 所有功率在合理范围内")
        outlier_results['power_over_expected'] = 0
    
    # 4. 负功率检测
    print("\n【4. 负功率检测】")
    neg_flagged = df['power_kw_negative_clipped_flag'].sum()
    print(f"  负功率标记数: {neg_flagged}")
    outlier_results['negative_power'] = neg_flagged
    
    # 5. 夜间非零功率检测
    print("\n【5. 夜间非零功率检测】")
    nighttime = df[df['is_daytime'] == 0]
    nighttime_nonzero = nighttime[nighttime['power_kw'] > 0.1]
    print(f"  夜间非零功率点 (>0.1kW): {len(nighttime_nonzero)}")
    outlier_results['nighttime_nonzero'] = len(nighttime_nonzero)
    
    return outlier_results


# =============================================================================
# 第四部分：数据处理
# =============================================================================

def process_irradiance_power_anomalies(df):
    """综合处理辐照-功率一致性问题"""
    
    print("\n" + "=" * 70)
    print("辐照-功率一致性深度清理")
    print("=" * 70)
    
    df_clean = df.copy()
    processing_log = []
    
    # 1. 处理日间高辐照零功率点
    print("\n【1. 处理日间高辐照零功率点】")
    daytime_zero = df_clean[(df_clean['is_daytime'] == 1) & 
                            (df_clean['ghi_wm2'] > 100) & 
                            (df_clean['power_kw'] == 0)]
    
    if len(daytime_zero) > 0:
        df_clean.loc[daytime_zero.index, 'is_potential_shutdown'] = 1
        print(f"  [处理] 标记 {len(daytime_zero)} 个日间高辐照零功率点为潜在停机")
        processing_log.append(f"标记 {len(daytime_zero)} 个日间高辐照零功率点")
    else:
        print("  [OK] 无日间高辐照零功率点")
    
    # 2. 处理低效率点
    print("\n【2. 处理低效率点】")
    low_eff_indices = detect_low_efficiency_points(df_clean, min_efficiency_threshold=5.0)
    
    if len(low_eff_indices) > 0:
        replaced_count = 0
        for idx in low_eff_indices:
            if idx > 0 and idx < len(df_clean) - 1:
                prev_idx = df_clean.index[df_clean.index.get_loc(idx) - 1] if df_clean.index.get_loc(idx) > 0 else None
                next_idx = df_clean.index[df_clean.index.get_loc(idx) + 1] if df_clean.index.get_loc(idx) < len(df_clean) - 1 else None
                
                if prev_idx and next_idx:
                    prev_power = df_clean.loc[prev_idx, 'power_kw']
                    next_power = df_clean.loc[next_idx, 'power_kw']
                    
                    if prev_power > 0 or next_power > 0:
                        interp_power = (prev_power + next_power) / 2
                        if interp_power > 0:
                            df_clean.loc[idx, 'power_kw'] = interp_power
                            df_clean.loc[idx, 'power_pu'] = interp_power / 450
                            df_clean.loc[idx, 'power_kw_negative_clipped_flag'] = 0
                            replaced_count += 1
        
        if replaced_count > 0:
            print(f"  [处理] 插值替换 {replaced_count} 个低效率异常点")
            processing_log.append(f"插值替换 {replaced_count} 个低效率异常点")
    else:
        print("  [OK] 无低效率点")
    
    # 3. 处理功率突变
    print("\n【3. 处理功率突变异常】")
    df_clean['power_change'] = df_clean['power_kw'].diff().abs()
    threshold_3std = df_clean['power_change'].mean() + 3 * df_clean['power_change'].std()
    
    sudden_changes = df_clean[df_clean['power_change'] > threshold_3std]
    print(f"  突变阈值 (均值+3σ): {threshold_3std:.2f} kW")
    print(f"  检测到突变点: {len(sudden_changes)}")
    
    if len(sudden_changes) > 0:
        daytime_changes = sudden_changes[sudden_changes['is_daytime'] == 1]
        smoothed_count = 0
        for idx in daytime_changes.index:
            if idx > 0 and idx < len(df_clean) - 1:
                prev_power = df_clean.loc[idx - 1, 'power_kw']
                next_power = df_clean.loc[idx + 1, 'power_kw']
                
                if df_clean.loc[idx, 'is_daytime'] == 1:
                    avg_power = (prev_power + next_power) / 2
                    df_clean.loc[idx, 'power_kw'] = avg_power
                    df_clean.loc[idx, 'power_pu'] = avg_power / 450
                    smoothed_count += 1
        
        if smoothed_count > 0:
            print(f"  [处理] 平滑 {smoothed_count} 个日间功率突变点")
            processing_log.append(f"平滑 {smoothed_count} 个日间功率突变点")
    
    if 'power_change' in df_clean.columns:
        df_clean.drop('power_change', axis=1, inplace=True)
    
    # 4. 处理功率毛刺
    print("\n【4. 处理功率毛刺（瞬时尖峰）】")
    df_clean['power_diff'] = df_clean['power_kw'].diff()
    
    spikes = []
    for i in range(1, len(df_clean) - 1):
        idx = df_clean.index[i]
        d1 = df_clean.loc[df_clean.index[i-1], 'power_diff'] if i > 0 else 0
        d2 = df_clean.loc[df_clean.index[i], 'power_diff'] if i < len(df_clean) else 0
        
        if d1 != 0 and d2 != 0 and d1 * d2 < 0:
            change_magnitude = abs(d1) + abs(d2)
            if change_magnitude > 50:
                spikes.append(idx)
    
    if len(spikes) > 0:
        print(f"  检测到 {len(spikes)} 个功率毛刺点")
        for idx in spikes:
            if idx > 0 and idx < len(df_clean) - 1:
                prev_power = df_clean.loc[idx - 1, 'power_kw']
                next_power = df_clean.loc[idx + 1, 'power_kw']
                df_clean.loc[idx, 'power_kw'] = (prev_power + next_power) / 2
                df_clean.loc[idx, 'power_pu'] = df_clean.loc[idx, 'power_kw'] / 450
        print(f"  [处理] 平滑 {len(spikes)} 个功率毛刺点")
        processing_log.append(f"平滑 {len(spikes)} 个功率毛刺点")
    else:
        print("  [OK] 无功率毛刺")
    
    if 'power_diff' in df_clean.columns:
        df_clean.drop('power_diff', axis=1, inplace=True)
    
    # 5. 处理夜间非零功率
    print("\n【5. 处理夜间非零功率】")
    nighttime_nonzero = df_clean[(df_clean['is_daytime'] == 0) & (df_clean['power_kw'] > 0.1)]
    
    if len(nighttime_nonzero) > 0:
        print(f"  将 {len(nighttime_nonzero)} 个夜间非零功率设为0")
        df_clean.loc[nighttime_nonzero.index, 'power_kw'] = 0.0
        df_clean.loc[nighttime_nonzero.index, 'power_pu'] = 0.0
        processing_log.append(f"修正 {len(nighttime_nonzero)} 个夜间非零功率点")
    else:
        print("  [OK] 无夜间非零功率")
    
    # 6. 重新计算派生特征
    print("\n【6. 重新计算派生特征】")
    df_clean['power_ramp_15m_kw'] = df_clean['power_kw'].diff()
    df_clean['power_ramp_15m_pu'] = df_clean['power_ramp_15m_kw'] / 450
    df_clean['data_quality_score'] = df_clean['data_quality_score'] * 0.98
    print(f"  数据质量分数已调整: * 0.98")
    
    return df_clean, processing_log


# =============================================================================
# 第五部分：报告生成
# =============================================================================

def generate_report(df_original, df_processed, outlier_results, consistency_results, processing_log):
    """生成处理报告"""
    
    print("\n" + "=" * 70)
    print("处理报告")
    print("=" * 70)
    
    report = []
    report.append("=" * 70)
    report.append("数据质量检查与处理报告")
    report.append("=" * 70)
    report.append("")
    report.append("【处理项目】")
    for item in processing_log:
        report.append(f"  - {item}")
    
    report.append("")
    report.append("【异常值检测统计】")
    for key, val in outlier_results.items():
        report.append(f"  {key}: {val}")
    
    report.append("")
    report.append("【辐照-功率一致性检测】")
    report.append(f"  日间高辐照零功率点: {consistency_results.get('zero_power_ghi_100', {}).get('zero_count', 0)}")
    report.append(f"  低效率点数量: {consistency_results.get('low_efficiency_count', 0)}")
    report.append(f"  功率突变点数量: {consistency_results.get('sudden_change_count', 0)}")
    
    report.append("")
    report.append("【处理前后对比】")
    std_orig = df_original['power_kw'].std()
    std_proc = df_processed['power_kw'].std()
    report.append(f"  功率标准差: {std_orig:.4f} -> {std_proc:.4f}")
    
    report.append("")
    report.append("=" * 70)
    
    report_text = "\n".join(report)
    print(report_text)
    
    return report_text


# =============================================================================
# 主函数
# =============================================================================

def main(mode='full'):
    """
    主函数
    
    Args:
        mode: 运行模式
            - 'check': 仅检查数据质量
            - 'score': 检查并评分
            - 'process': 检查、处理并保存
            - 'full': 完整流程（默认）
    """
    
    print("=" * 70)
    print("数据质量检查与处理脚本")
    print("=" * 70)
    
    # 数据文件路径
    data_dir = Path(__file__).parent / "processed" / "stations"
    input_file = data_dir / "Bozhou_1_preprocessed.csv"
    output_file = data_dir / "Bozhou_1_cleaned.csv"
    report_dir = Path(__file__).parent / "data_quality"
    report_dir.mkdir(exist_ok=True)
    
    if not input_file.exists():
        print(f"错误: 文件不存在 {input_file}")
        return
    
    # 读取数据
    print(f"\n读取数据: {input_file}")
    df = pd.read_csv(input_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"数据行数: {len(df):,}")
    
    # 保存原始数据
    df_original = df.copy()
    
    # 根据模式执行
    if mode in ['check', 'score', 'full']:
        check_data_quality(input_file, df)
    
    if mode in ['score', 'full']:
        score_data_quality(df)
    
    if mode in ['process', 'full']:
        # 分析辐照-功率一致性
        consistency_results, daytime_zero_points = analyze_irradiance_power_consistency(df.copy())
        
        # 异常值检测
        outlier_results = detect_outliers(df)
        
        # 数据处理
        df_processed, processing_log = process_irradiance_power_anomalies(df)
        
        # 生成报告
        report = generate_report(df_original, df_processed, outlier_results, consistency_results, processing_log)
        
        # 保存处理后的数据
        df_processed.to_csv(output_file, index=False)
        print(f"\n处理后数据已保存: {output_file}")
        
        # 保存报告
        report_file = report_dir / "consistency_outlier_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"处理报告已保存: {report_file}")
        
        # 保存评分报告
        _, scores, grade = score_data_quality(df_processed)
        
        return df_processed, {
            'outlier_results': outlier_results,
            'consistency_results': consistency_results,
            'processing_log': processing_log,
            'scores': scores,
            'grade': grade
        }
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='数据质量检查与处理脚本')
    parser.add_argument('--mode', '-m', type=str, default='full',
                        choices=['check', 'score', 'process', 'full'],
                        help='运行模式: check(仅检查), score(检查+评分), process(检查+处理), full(完整流程)')
    
    args = parser.parse_args()
    main(mode=args.mode)
