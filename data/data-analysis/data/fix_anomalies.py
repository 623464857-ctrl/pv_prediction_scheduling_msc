"""
光伏数据异常值检测与修正
1. 夜间异常功率（置零修正）
2. 白天非零但GHI=0的异常（置零修正）
3. 白天零功率疑似停机（标记处理）
"""

import pandas as pd
import os

# 路径配置
input_path = r"c:\Users\MoYu\Desktop\pv_prediction_scheduling_msc_new\data\data-analysis\data\明月湖_cleaned.csv"
output_path = r"c:\Users\MoYu\Desktop\pv_prediction_scheduling_msc_new\data\data-analysis\data\明月湖_cleaned.csv"
report_path = r"c:\Users\MoYu\Desktop\pv_prediction_scheduling_msc_new\data\data-analysis\data\data_quality_report.csv"

# 读取数据
df = pd.read_csv(input_path)
df['timestamp'] = pd.to_datetime(df['timestamp'])

print("=" * 60)
print("光伏数据异常值检测与修正报告")
print("=" * 60)
print(f"\n原始数据记录数: {len(df)}")

# 初始化统计
stats = []

# ========== 1. 夜间异常功率修正 ==========
print("\n【1】夜间异常功率检测与修正")
print("-" * 40)
night_mask = (df['solar_altitude_deg'] < 0) & (df['power_kw'] > 0)
night_count = night_mask.sum()
print(f"检测到夜间非零功率记录: {night_count} 条")

if night_count > 0:
    print("\n夜间异常功率样本 (前5条):")
    print(df.loc[night_mask, ['timestamp', 'power_kw', 'solar_altitude_deg', 'ghi']].head().to_string(index=False))
    
    # 修正：将夜间功率置零
    df.loc[night_mask, 'power_kw'] = 0
    print(f"\n已将 {night_count} 条夜间功率置零")
else:
    print("无夜间异常功率")

stats.append({'类型': '夜间异常功率', '异常记录数': night_count, '处理方式': '置零修正'})

# ========== 2. 白天非零但GHI=0的异常修正 ==========
print("\n【2】白天非零但GHI=0的异常检测与修正")
print("-" * 40)
day_ghi_zero = (df['solar_altitude_deg'] >= 0) & (df['ghi'] == 0) & (df['power_kw'] > 0)
day_ghi_zero_count = day_ghi_zero.sum()
print(f"检测到白天GHI=0但有功率的记录: {day_ghi_zero_count} 条")

if day_ghi_zero_count > 0:
    print("\n白天GHI=0异常样本 (前5条):")
    print(df.loc[day_ghi_zero, ['timestamp', 'power_kw', 'solar_altitude_deg', 'ghi']].head().to_string(index=False))
    
    # 修正：将白天GHI=0时的功率置零
    df.loc[day_ghi_zero, 'power_kw'] = 0
    print(f"\n已将 {day_ghi_zero_count} 条白天异常功率置零")
else:
    print("无白天GHI=0异常")

stats.append({'类型': '白天GHI=0异常', '异常记录数': day_ghi_zero_count, '处理方式': '置零修正'})

# ========== 3. 白天零功率疑似停机标记 ==========
print("\n【3】白天零功率疑似停机标记")
print("-" * 40)
# 白天定义：太阳高度角 >= 0
# 停机阈值：GHI > 100 W/m2（表示有足够的太阳辐照度）
day_stop = (df['solar_altitude_deg'] >= 0) & (df['ghi'] > 100) & (df['power_kw'] == 0)
day_stop_count = day_stop.sum()
print(f"检测到白天疑似停机记录: {day_stop_count} 条")
print(f"(白天定义: solar_altitude_deg >= 0, GHI > 100 W/m2)")

if day_stop_count > 0:
    print("\n疑似停机样本 (前5条):")
    print(df.loc[day_stop, ['timestamp', 'power_kw', 'solar_altitude_deg', 'ghi']].head().to_string(index=False))
    
    # 添加停机标记列
    if 'is_potential_shutdown' not in df.columns:
        df['is_potential_shutdown'] = 0
    df.loc[day_stop, 'is_potential_shutdown'] = 1
    print(f"\n已标记 {day_stop_count} 条疑似停机记录 (is_potential_shutdown=1)")
else:
    print("无白天疑似停机")

stats.append({'类型': '白天疑似停机', '异常记录数': day_stop_count, '处理方式': '标记处理'})

# ========== 保存修正后的数据 ==========
print("\n" + "=" * 60)
print("保存修正结果")
print("=" * 60)

# 保存修正后的数据
df.to_csv(output_path, index=False)
print(f"修正后数据已保存: {output_path}")

# 保存统计报告
stats_df = pd.DataFrame(stats)
stats_df.to_csv(report_path, index=False, encoding='utf-8-sig')
print(f"统计报告已保存: {report_path}")

# 打印汇总
print("\n" + "=" * 60)
print("修正汇总")
print("=" * 60)
print(f"总记录数: {len(df)}")
print(f"1. 夜间异常功率修正: {night_count} 条")
print(f"2. 白天GHI=0异常修正: {day_ghi_zero_count} 条")
print(f"3. 白天疑似停机标记: {day_stop_count} 条")
print(f"\n新增标记列: is_potential_shutdown (0=正常, 1=疑似停机)")

# 验证修正结果
print("\n" + "=" * 60)
print("修正后验证")
print("=" * 60)
remaining_night = ((df['solar_altitude_deg'] < 0) & (df['power_kw'] > 0)).sum()
remaining_day_ghi = ((df['solar_altitude_deg'] >= 0) & (df['ghi'] == 0) & (df['power_kw'] > 0)).sum()
print(f"夜间剩余非零功率: {remaining_night} 条")
print(f"白天GHI=0剩余异常: {remaining_day_ghi} 条")

print("\n处理完成！")
