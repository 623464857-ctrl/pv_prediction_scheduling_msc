"""
Site1 辐照度和功率数据物理一致性诊断
=====================================
检查：
1. 三类辐照度(GHI, DNI, GTI)的物理关系矛盾
2. 功率与辐照度的相位差/滞后问题

运行：
python experiments/prediction/step1_data_cleaning_alignment/run_irradiance_power_diagnostics.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime


def diagnose_irradiance_physical_consistency(df: pd.DataFrame) -> dict:
    """
    诊断三类辐照度的物理一致性
    
    物理关系：
    - GTI (总辐照度) = DNI * cos(太阳天顶角) + DHI (水平散射)
    - 或者：GHI = DNI * cos(天顶角) + DHI
    - GTI 应该 ≈ GHI
    - DNI * cos(天顶角) 应该 <= GHI
    - Direct 不应该远大于 Total
    """
    results = {
        'summary': {},
        'anomalies': [],
        'details': {}
    }
    
    # 获取辐照度列
    gti_col = 'total_irradiance_wm2'  # 或 GHI
    dni_col = 'direct_normal_irradiance_wm2'
    ghi_col = 'global_horizontal_irradiance_wm2'
    
    # 检查列是否存在
    has_gti = gti_col in df.columns
    has_dni = dni_col in df.columns
    has_ghi = ghi_col in df.columns
    
    results['columns_available'] = {
        'GTI': has_gti,
        'DNI': has_dni,
        'GHI': has_ghi
    }
    
    if not has_gti and not has_ghi:
        results['error'] = 'No irradiance columns found'
        return results
    
    n_total = len(df)
    
    # ========================================
    # 1. 检查: Direct > Total (异常)
    # ========================================
    if has_dni and (has_gti or has_ghi):
        direct_col = dni_col
        total_col = gti_col if has_gti else ghi_col
        
        # Direct > Total 是物理上不可能的
        direct_gt_total = df[direct_col] > df[total_col]
        count_direct_gt_total = direct_gt_total.sum()
        
        results['anomalies'].append({
            'type': 'Direct > Total',
            'description': '直射辐照度 > 总辐照度 (物理不可能)',
            'count': int(count_direct_gt_total),
            'percentage': float(count_direct_gt_total / n_total * 100),
            'severity': 'HIGH' if count_direct_gt_total > 100 else 'MEDIUM'
        })
        
        # 记录具体案例
        if count_direct_gt_total > 0:
            sample_cases = df[direct_gt_total][['timestamp', direct_col, total_col]].head(10)
            results['details']['direct_gt_total_cases'] = sample_cases.to_dict('records')
    
    # ========================================
    # 2. 检查: Total < Horizontal (异常)
    # ========================================
    if has_gti and has_ghi:
        total_lt_horizontal = df[gti_col] < df[ghi_col]
        count_total_lt_h = total_lt_horizontal.sum()
        
        results['anomalies'].append({
            'type': 'GTI < GHI',
            'description': '总辐照度 < 水平辐照度 (物理矛盾)',
            'count': int(count_total_lt_h),
            'percentage': float(count_total_lt_h / n_total * 100),
            'severity': 'HIGH' if count_total_lt_h > 100 else 'MEDIUM'
        })
        
        if count_total_lt_h > 0:
            sample_cases = df[total_lt_horizontal][['timestamp', gti_col, ghi_col]].head(10)
            results['details']['gti_lt_ghi_cases'] = sample_cases.to_dict('records')
    
    # ========================================
    # 3. 检查: Direct_component > Total (分离验证)
    # ========================================
    # DNI * cos(天顶角) 应该 <= GHI
    if has_dni and has_ghi and 'solar_zenith_angle_deg' in df.columns:
        zenith = df['solar_zenith_angle_deg']
        cos_zenith = np.cos(np.radians(zenith)).clip(0, 1)
        
        direct_component = df[dni_col] * cos_zenith
        direct_exceeds_ghi = direct_component > df[ghi_col]
        count_direct_exceeds = direct_exceeds_ghi.sum()
        
        results['anomalies'].append({
            'type': 'DNI*cos(zenith) > GHI',
            'description': '直射分量超过全球水平辐照',
            'count': int(count_direct_exceeds),
            'percentage': float(count_direct_exceeds / n_total * 100),
            'severity': 'HIGH' if count_direct_exceeds > 100 else 'MEDIUM'
        })
    
    # ========================================
    # 4. 检查: 辐照度为负值 (传感器错误)
    # ========================================
    for col_name, col_data in [('GTI', gti_col), ('DNI', dni_col), ('GHI', ghi_col)]:
        if col_data in df.columns:
            negative_mask = df[col_data] < 0
            count_negative = negative_mask.sum()
            
            if count_negative > 0:
                results['anomalies'].append({
                    'type': f'{col_name} < 0',
                    'description': f'{col_name}出现负值 (传感器错误)',
                    'count': int(count_negative),
                    'percentage': float(count_negative / n_total * 100),
                    'severity': 'HIGH'
                })
    
    # ========================================
    # 5. 检查: 辐照度超过理论最大值
    # ========================================
    if 'theoretical_max_gti' in df.columns and has_gti:
        over_theoretical = df[gti_col] > df['theoretical_max_gti'] * 1.05
        count_over = over_theoretical.sum()
        
        results['anomalies'].append({
            'type': 'GTI > Theoretical Max',
            'description': '辐照度超过理论最大值',
            'count': int(count_over),
            'percentage': float(count_over / n_total * 100),
            'severity': 'LOW'
        })
    
    # ========================================
    # 6. DNI/DHI比例异常
    # ========================================
    if has_dni and has_ghi:
        # 正常情况下，DNI/DHI比例在0-5之间
        dni_dhi_ratio = df[dni_col] / (df[ghi_col] + 1)
        
        # 极端情况
        extreme_high = (dni_dhi_ratio > 10).sum()
        extreme_low = (dni_dhi_ratio < 0.01).sum()
        
        results['details']['dni_dhi_ratio_stats'] = {
            'mean': float(dni_dhi_ratio.mean()),
            'median': float(dni_dhi_ratio.median()),
            'std': float(dni_dhi_ratio.std()),
            'extreme_high_count': int(extreme_high),
            'extreme_low_count': int(extreme_low)
        }
    
    # ========================================
    # 汇总
    # ========================================
    high_severity = [a for a in results['anomalies'] if a['severity'] == 'HIGH']
    results['summary']['high_severity_count'] = len(high_severity)
    results['summary']['total_anomaly_count'] = len(results['anomalies'])
    
    return results


def diagnose_power_irradiance_lag(df: pd.DataFrame, capacity_mw: float = 50.0) -> dict:
    """
    诊断功率与辐照度的相位差/滞后问题
    
    方法：
    1. 计算辐照度和功率的交叉相关性
    2. 找出最大相关性的滞后步数
    3. 分析白天时段的响应延迟
    """
    results = {
        'lag_analysis': {},
        'hourly_correlation': {},
        'rise_phase_analysis': {}
    }
    
    # 获取列
    gti_col = 'total_irradiance_wm2' if 'total_irradiance_wm2' in df.columns else 'global_horizontal_irradiance_wm2'
    power_col = 'power_mw'
    
    if gti_col not in df.columns or power_col not in df.columns:
        results['error'] = 'Required columns not found'
        return results
    
    # ========================================
    # 1. 全局交叉相关分析
    # ========================================
    gti = df[gti_col].values
    power = df[power_col].values
    
    # 只用日间数据
    if 'hour' in df.columns:
        daytime_mask = df['hour'].between(7, 17)
        gti_daytime = gti[daytime_mask.values]
        power_daytime = power[daytime_mask.values]
    else:
        gti_daytime = gti
        power_daytime = power
    
    # 计算交叉相关 (滞后从-12到+12步，15分钟间隔 = ±3小时)
    max_lag = 12
    correlations = []
    
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            corr = np.corrcoef(gti_daytime[:lag], power_daytime[-lag:])[0, 1]
        elif lag > 0:
            corr = np.corrcoef(gti_daytime[lag:], power_daytime[:-lag])[0, 1]
        else:
            corr = np.corrcoef(gti_daytime, power_daytime)[0, 1]
        correlations.append((lag * 15, corr))  # 转换为分钟
    
    # 找最大相关性对应的滞后
    best_lag = max(correlations, key=lambda x: x[1])
    
    results['lag_analysis'] = {
        'max_lag_minutes': int(best_lag[0]),
        'max_correlation': float(best_lag[1]),
        'all_correlations': correlations
    }
    
    # ========================================
    # 2. 分时段相关性分析
    # ========================================
    if 'hour' in df.columns:
        for hour in range(6, 20):
            hour_mask = df['hour'] == hour
            if hour_mask.sum() > 10:
                gti_hour = gti[hour_mask.values]
                power_hour = power[hour_mask.values]
                corr = np.corrcoef(gti_hour, power_hour)[0, 1]
                results['hourly_correlation'][hour] = float(corr)
    
    # ========================================
    # 3. 上升时段分析 (7-12点)
    # ========================================
    rise_mask = df['hour'].between(7, 12)
    df_rise = df[rise_mask].copy()
    
    if len(df_rise) > 0:
        # 计算斜率
        gti_rise = df_rise[gti_col].values
        power_rise = df_rise[power_col].values
        
        # 计算归一化斜率
        gti_range = gti_rise.max() - gti_rise.min()
        power_range = power_rise.max() - power_rise.min()
        
        if gti_range > 0 and power_range > 0:
            # 归一化后比较上升速度
            gti_normalized = (gti_rise - gti_rise.min()) / gti_range
            power_normalized = (power_rise - power_rise.min()) / power_range
            
            # 计算上升延迟
            # 找到辐照度达到50%的时间点
            gti_50_idx = np.argmax(gti_normalized >= 0.5)
            power_50_idx = np.argmax(power_normalized >= 0.5)
            
            if gti_50_idx > 0 and power_50_idx > 0:
                rise_delay = (power_50_idx - gti_50_idx) * 15  # 分钟
                
                results['rise_phase_analysis'] = {
                    'rise_delay_minutes': int(rise_delay),
                    'gti_reaches_50pct_at_min': int(gti_50_idx * 15),
                    'power_reaches_50pct_at_min': int(power_50_idx * 15),
                    'gti_normalized_slope': float(np.gradient(gti_normalized).mean()),
                    'power_normalized_slope': float(np.gradient(power_normalized).mean())
                }
    
    # ========================================
    # 4. 滞后影响评估
    # ========================================
    lag_minutes = best_lag[0]
    if abs(lag_minutes) > 30:
        results['lag_analysis']['impact'] = 'HIGH - 建议在特征中加入滞后特征'
        results['lag_analysis']['recommendation'] = '使用过去1-2个时刻的辐照度作为特征'
    elif abs(lag_minutes) > 15:
        results['lag_analysis']['impact'] = 'MEDIUM - 考虑使用滞后特征'
        results['lag_analysis']['recommendation'] = '可添加lag-1辐照度特征'
    else:
        results['lag_analysis']['impact'] = 'LOW - 滞后可忽略'
        results['lag_analysis']['recommendation'] = '当前特征设置足够'
    
    return results


def generate_diagnostic_report():
    """生成诊断报告"""
    
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    DATA_PATH = PROJECT_ROOT / "data" / "prediction" / "step1_preprocessing" / "processed" / "stations" / "Site_1_optimized.csv"
    
    print("=" * 80)
    print("Site1 辐照度和功率数据物理一致性诊断报告")
    print("=" * 80)
    print(f"生成时间: {datetime.now()}")
    print()
    
    # 读取数据
    df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
    print(f"数据量: {len(df):,} 条")
    print(f"时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
    print()
    
    capacity = 50.0
    
    # ========================================
    # 诊断1: 辐照度物理一致性
    # ========================================
    print("=" * 80)
    print("诊断1: 三类辐照度物理一致性")
    print("=" * 80)
    
    irr_results = diagnose_irradiance_physical_consistency(df)
    
    print(f"\n可用列: {irr_results.get('columns_available', {})}")
    print()
    
    if irr_results.get('anomalies'):
        print("检测到的异常:")
        for anomaly in irr_results['anomalies']:
            severity_icon = "!!!" if anomaly['severity'] == 'HIGH' else "!!" if anomaly['severity'] == 'MEDIUM' else "!"
            print(f"\n  {severity_icon} {anomaly['type']}")
            print(f"      {anomaly['description']}")
            print(f"      数量: {anomaly['count']:,} 条 ({anomaly['percentage']:.2f}%)")
    
    # 显示 DNI/DHI 比例统计
    if 'dni_dhi_ratio_stats' in irr_results.get('details', {}):
        stats = irr_results['details']['dni_dhi_ratio_stats']
        print(f"\nDNI/DHI 比例统计:")
        print(f"  平均值: {stats['mean']:.2f}")
        print(f"  中位数: {stats['median']:.2f}")
        print(f"  标准差: {stats['std']:.2f}")
        print(f"  极端高值(>10): {stats['extreme_high_count']:,} 条")
        print(f"  极端低值(<0.01): {stats['extreme_low_count']:,} 条")
    
    print()
    
    # ========================================
    # 诊断2: 功率-辐照度滞后分析
    # ========================================
    print("=" * 80)
    print("诊断2: 功率与辐照度相位差/滞后分析")
    print("=" * 80)
    
    lag_results = diagnose_power_irradiance_lag(df, capacity)
    
    if 'lag_analysis' in lag_results:
        analysis = lag_results['lag_analysis']
        print(f"\n全局交叉相关分析:")
        print(f"  最优滞后: {analysis['max_lag_minutes']} 分钟")
        print(f"  最大相关系数: {analysis['max_correlation']:.4f}")
        print(f"  影响评估: {analysis.get('impact', 'N/A')}")
        print(f"  建议: {analysis.get('recommendation', 'N/A')}")
    
    if lag_results.get('hourly_correlation'):
        print(f"\n分时段相关性:")
        print(f"  {'时段':<8} {'相关系数':<12}")
        print(f"  {'-'*25}")
        for hour, corr in sorted(lag_results['hourly_correlation'].items()):
            print(f"  {hour:>2}:00     {corr:.4f}")
    
    if lag_results.get('rise_phase_analysis'):
        rise = lag_results['rise_phase_analysis']
        print(f"\n上升时段分析 (7-12点):")
        print(f"  辐照度达到50%的时间: {rise['gti_reaches_50pct_at_min']} 分钟")
        print(f"  功率达到50%的时间: {rise['power_reaches_50pct_at_min']} 分钟")
        print(f"  上升延迟: {rise['rise_delay_minutes']} 分钟")
    
    print()
    
    # ========================================
    # 结论和建议
    # ========================================
    print("=" * 80)
    print("结论和建议")
    print("=" * 80)
    
    # 问题1总结
    high_sev_irr = irr_results['summary'].get('high_severity_count', 0)
    total_anomaly = irr_results['summary'].get('total_anomaly_count', 0)
    
    print("\n【问题1: 辐照度物理矛盾】")
    if high_sev_irr > 0:
        print(f"  发现 {high_sev_irr} 个高严重度问题")
        print("  建议:")
        print("    1. 优先使用 GHI (global_horizontal_irradiance_wm2) 作为主要特征")
        print("    2. 对 DNI 和 GTI 进行一致性校正")
        print("    3. 在模型训练时添加辐照度一致性验证特征")
        print("    4. 考虑只使用 GHI，避免多个传感器冲突")
    else:
        print("  未发现严重物理矛盾")
    
    # 问题2总结
    print("\n【问题2: 功率-辐照度滞后】")
    if 'lag_analysis' in lag_results:
        lag = lag_results['lag_analysis']['max_lag_minutes']
        impact = lag_results['lag_analysis'].get('impact', 'N/A')
        
        if 'HIGH' in impact:
            print(f"  检测到显著滞后: {lag} 分钟")
            print("  建议:")
            print("    1. 在特征中添加滞后辐照度 (lag-1, lag-2, lag-4)")
            print("    2. 添加辐照度变化率特征 (gti_change_rate)")
            print("    3. 添加功率变化率特征 (power_trend)")
        elif 'MEDIUM' in impact:
            print(f"  检测到中等滞后: {lag} 分钟")
            print("  建议: 考虑添加 lag-1 辐照度特征")
        else:
            print("  滞后影响较小，当前特征设置足够")
    
    print()
    print("=" * 80)
    print("诊断完成")
    print("=" * 80)
    
    return irr_results, lag_results


if __name__ == "__main__":
    generate_diagnostic_report()
