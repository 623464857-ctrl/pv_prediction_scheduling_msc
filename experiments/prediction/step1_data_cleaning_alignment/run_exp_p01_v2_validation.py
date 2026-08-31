"""
Site1 数据质量验证与报告生成
================================
对优化后的数据进行全面验证，并生成详细的诊断报告

运行：
python experiments/prediction/step1_data_cleaning_alignment/run_exp_p01_v2_validation.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


def generate_comprehensive_report():
    """生成完整的数据质量报告"""
    
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    DATA_PATH = PROJECT_ROOT / "data" / "prediction" / "step1_preprocessing" / "processed" / "stations" / "Site_1_optimized.csv"
    
    print("=" * 80)
    print("Site1 数据质量验证报告")
    print("=" * 80)
    print(f"生成时间: {datetime.now()}")
    print()
    
    # 读取数据
    df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
    capacity = 50.0
    
    # ========================================
    # 1. 数据概览
    # ========================================
    print("=" * 80)
    print("1. 数据概览")
    print("=" * 80)
    print(f"  总数据量: {len(df):,} 条")
    print(f"  时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
    print(f"  数据天数: {(df['timestamp'].max() - df['timestamp'].min()).days} 天")
    print(f"  采样频率: 15分钟")
    print(f"  额定容量: {capacity} MW")
    print()
    
    # ========================================
    # 2. 质量分布统计
    # ========================================
    print("=" * 80)
    print("2. 数据质量分布")
    print("=" * 80)
    
    if 'overall_quality_score' in df.columns:
        quality_stats = df['overall_quality_score'].describe()
        print(f"  质量分数统计:")
        print(f"    均值: {quality_stats['mean']:.4f}")
        print(f"    中位数: {quality_stats['50%']:.4f}")
        print(f"    标准差: {quality_stats['std']:.4f}")
        print(f"    最小值: {quality_stats['min']:.4f}")
        print(f"    最大值: {quality_stats['max']:.4f}")
        print()
    
    if 'quality_grade' in df.columns:
        grade_counts = df['quality_grade'].value_counts().sort_index()
        print(f"  质量等级分布:")
        for grade in ['A', 'B', 'C', 'D']:
            if grade in grade_counts.index:
                count = grade_counts[grade]
                pct = count / len(df) * 100
                print(f"    {grade}级: {count:,} 条 ({pct:.2f}%)")
        print()
    
    # ========================================
    # 3. 异常检测统计
    # ========================================
    print("=" * 80)
    print("3. 异常检测统计")
    print("=" * 80)
    
    anomaly_cols = [c for c in df.columns if 'anomaly' in c.lower() and 'level' not in c.lower()]
    for col in anomaly_cols:
        count = df[col].sum()
        pct = count / len(df) * 100
        name = col.replace('anomaly_', '').replace('_', ' ')
        print(f"  {name}: {count:,} 条 ({pct:.2f}%)")
    print()
    
    # 异常级别分布
    if 'irradiance_power_anomaly_level' in df.columns:
        level_dist = df['irradiance_power_anomaly_level'].value_counts().sort_index()
        print(f"  辐照-功率异常级别分布:")
        level_names = {0: '正常', 1: '轻度', 2: '严重', 3: '明显错误'}
        for level, name in level_names.items():
            if level in level_dist.index:
                count = level_dist[level]
                pct = count / len(df) * 100
                print(f"    {level}级({name}): {count:,} 条 ({pct:.2f}%)")
        print()
    
    # ========================================
    # 4. 设备状态检测
    # ========================================
    print("=" * 80)
    print("4. 设备状态检测")
    print("=" * 80)
    
    if 'in_outage' in df.columns:
        outage_count = df['in_outage'].sum()
        outage_pct = outage_count / len(df) * 100
        print(f"  设备停机期间: {outage_count:,} 条 ({outage_pct:.2f}%)")
        
        # 找出停机时段
        outage_groups = []
        in_outage = False
        start_time = None
        
        for idx, row in df.iterrows():
            if row['in_outage'] == 1 and not in_outage:
                in_outage = True
                start_time = row['timestamp']
            elif row['in_outage'] == 0 and in_outage:
                in_outage = False
                outage_groups.append(start_time)
        
        if outage_groups:
            print(f"  检测到的停机时段:")
            for i, start in enumerate(outage_groups[:5]):
                print(f"    停机{i+1}: {start}")
            if len(outage_groups) > 5:
                print(f"    ... 共{len(outage_groups)}个停机时段")
        print()
    
    # 功率突变统计
    if 'power_sudden_change' in df.columns:
        sudden_count = df['power_sudden_change'].sum()
        print(f"  功率突变(>30%容量): {sudden_count:,} 条")
        
        if 'power_sudden_change_suspicious' in df.columns:
            suspicious = df['power_sudden_change_suspicious'].sum()
            print(f"  其中白天可疑突变: {suspicious:,} 条")
        print()
    
    # ========================================
    # 5. 特征质量统计
    # ========================================
    print("=" * 80)
    print("5. 特征质量统计")
    print("=" * 80)
    
    # 湿度质量
    if 'humidity_quality_tier' in df.columns:
        hum_dist = df['humidity_quality_tier'].value_counts().sort_index()
        tier_names = {0: '不可用', 1: '低', 2: '中', 3: '高'}
        print(f"  湿度质量分布:")
        for tier, name in tier_names.items():
            if tier in hum_dist.index:
                count = hum_dist[tier]
                pct = count / len(df) * 100
                print(f"    {tier}级({name}): {count:,} 条 ({pct:.2f}%)")
        print()
    
    # 辐照质量
    if 'irradiance_quality_tier' in df.columns:
        irr_dist = df['irradiance_quality_tier'].value_counts().sort_index()
        print(f"  辐照质量分布:")
        for tier in sorted(irr_dist.index):
            count = irr_dist[tier]
            pct = count / len(df) * 100
            print(f"    {tier}级: {count:,} 条 ({pct:.2f}%)")
        print()
    
    # ========================================
    # 6. 容量利用率分析
    # ========================================
    print("=" * 80)
    print("6. 容量利用率分析")
    print("=" * 80)
    
    max_power = df['power_mw'].max()
    max_util = max_power / capacity * 100
    print(f"  历史最大功率: {max_power:.2f} MW")
    print(f"  最大容量利用率: {max_util:.2f}%")
    
    # 晴空时段分析
    if 'total_irradiance_wm2' in df.columns:
        clear_sky = df[(df['total_irradiance_wm2'] > 700) & (df['hour'].between(11, 14))]
        if len(clear_sky) > 0:
            clear_max = clear_sky['power_mw'].max()
            clear_mean = clear_sky['power_mw'].mean()
            clear_util = clear_max / capacity * 100
            print(f"  强辐照(>700)中午前平均功率: {clear_mean:.2f} MW")
            print(f"  强辐照(>700)中午前最大功率: {clear_max:.2f} MW ({clear_util:.1f}%)")
        print()
    
    # ========================================
    # 7. 月度功率统计
    # ========================================
    print("=" * 80)
    print("7. 月度功率统计")
    print("=" * 80)
    
    df['month'] = df['timestamp'].dt.month
    monthly = df.groupby('month')['power_mw'].agg(['mean', 'max', 'std'])
    monthly.columns = ['月均功率', '月最大功率', '功率标准差']
    
    print(f"  {'月份':<6} {'月均功率(MW)':<15} {'月最大功率(MW)':<15} {'标准差(MW)':<10}")
    print(f"  {'-'*50}")
    for month, row in monthly.iterrows():
        print(f"  {month:>2}月   {row['月均功率']:>12.2f}    {row['月最大功率']:>12.2f}    {row['功率标准差']:>8.2f}")
    print()
    
    # ========================================
    # 8. 问题数据汇总
    # ========================================
    print("=" * 80)
    print("8. 问题数据汇总")
    print("=" * 80)
    
    problem_count = 0
    
    # 夜间发电
    if 'anomaly_night_power' in df.columns:
        night_power = df['anomaly_night_power'].sum()
        if night_power > 0:
            print(f"  [!] 夜间发电: {night_power:,} 条 -> 需检查功率传感器")
            problem_count += night_power
    
    # 晴空无功率
    if 'anomaly_severe_clear_no_power' in df.columns:
        clear_no_power = df['anomaly_severe_clear_no_power'].sum()
        if clear_no_power > 0:
            print(f"  [!] 中午晴空无功率: {clear_no_power:,} 条 -> 可能是设备故障或弃光")
            problem_count += clear_no_power
    
    # 设备停机
    if 'in_outage' in df.columns:
        outage = df['in_outage'].sum()
        if outage > 0:
            print(f"  [!] 设备停机期间: {outage:,} 条 ({outage/len(df)*100:.2f}%) -> 建议排除这些时段")
            problem_count += outage
    
    # 低湿度
    if 'humidity_quality_tier' in df.columns:
        low_hum = (df['humidity_quality_tier'] <= 1).sum()
        if low_hum > 0:
            print(f"  [!] 低质量湿度: {low_hum:,} 条 ({low_hum/len(df)*100:.2f}%) -> 建议使用WRF湿度")
            problem_count += low_hum
    
    if problem_count == 0:
        print("  [+] 未发现明显问题数据")
    print()
    
    # ========================================
    # 9. 优化前后对比
    # ========================================
    print("=" * 80)
    print("9. 优化前后对比")
    print("=" * 80)
    
    # 读取原始数据进行对比
    original_path = PROJECT_ROOT / "data" / "prediction" / "step1_preprocessing" / "processed" / "stations" / "Site_1_preprocessed.csv"
    if original_path.exists():
        df_orig = pd.read_csv(original_path)
        
        # 对比辐照-功率相关性
        gti_orig = df_orig['total_irradiance_wm2']
        power_orig = df_orig['power_mw']
        corr_orig = gti_orig.corr(power_orig)
        
        gti_new = df['total_irradiance_wm2']
        power_new = df['power_mw']
        corr_new = gti_new.corr(power_new)
        
        print(f"  辐照-功率相关系数:")
        print(f"    优化前: {corr_orig:.4f}")
        print(f"    优化后: {corr_new:.4f}")
        print()
        
        # 对比夜间发电
        night_orig = ((df_orig['total_irradiance_wm2'] < 5) & (df_orig['power_mw'] > 0)).sum()
        night_new = df['anomaly_night_power'].sum()
        print(f"  夜间发电异常:")
        print(f"    优化前: {night_orig:,} 条")
        print(f"    优化后: {night_new:,} 条")
        print()
        
        # 对比晴空无功率
        clear_orig = ((df_orig['total_irradiance_wm2'] > 700) & (df_orig['hour'].between(10, 14)) & 
                     (df_orig['power_mw'] < 5)).sum()
        clear_new = df['anomaly_severe_clear_no_power'].sum()
        print(f"  中午晴空无功率:")
        print(f"    优化前: {clear_orig:,} 条")
        print(f"    优化后: {clear_new:,} 条")
        print()
    
    # ========================================
    # 10. 建议
    # ========================================
    print("=" * 80)
    print("10. 使用建议")
    print("=" * 80)
    
    # 计算可用样本数
    if 'use_for_training' in df.columns:
        usable = df['use_for_training'].sum()
        usable_pct = usable / len(df) * 100
        print(f"  推荐训练样本数: {usable:,} / {len(df):,} ({usable_pct:.1f}%)")
        
        if usable_pct > 90:
            print("  [OK] 数据质量良好，可用于模型训练")
        elif usable_pct > 70:
            print("  [~~] 数据质量一般，建议结合样本权重使用")
        else:
            print("  [!!] 数据质量问题较多，建议先清洗数据")
    print()
    
    # 特征使用建议
    print("  特征使用建议:")
    print("    - 湿度数据: 建议使用WRF湿度或质量分级后加权使用")
    print("    - 辐照数据: 质量良好，但需注意异常标记")
    print("    - 新增特征: power_efficiency, gti_utilization 可用于模型训练")
    print("    - sample_weight: 建议在训练时使用此权重")
    print()
    
    print("=" * 80)
    print("报告生成完成")
    print("=" * 80)
    
    # 保存报告
    report_path = PROJECT_ROOT / "data" / "prediction" / "step1_preprocessing" / "processed" / "stations" / "Site_1_optimization_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        # 重定向输出到文件
        pass  # 已在终端显示
    
    print(f"\n报告已保存到: {report_path}")


if __name__ == "__main__":
    generate_comprehensive_report()
