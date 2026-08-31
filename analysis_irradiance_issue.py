"""
深入分析辐照度物理矛盾
"""

import pandas as pd
import numpy as np

df = pd.read_csv('data/prediction/step1_preprocessing/processed/stations/Site_1_optimized.csv', parse_dates=['timestamp'])

print('='*80)
print('深入分析辐照度物理矛盾')
print('='*80)

# 1. Direct > Total 案例
print('\n【Direct > Total 异常案例】')
direct_gt_total = df[df['direct_normal_irradiance_wm2'] > df['total_irradiance_wm2']]
print(f'总数: {len(direct_gt_total):,} 条')

if len(direct_gt_total) > 0:
    sample = direct_gt_total[['timestamp', 'direct_normal_irradiance_wm2', 'total_irradiance_wm2', 'hour']].head(15)
    print('\n前15个案例:')
    for _, row in sample.iterrows():
        if row['total_irradiance_wm2'] > 0:
            ratio = row['direct_normal_irradiance_wm2'] / row['total_irradiance_wm2']
        else:
            ratio = float('inf')
        print(f"  {row['timestamp']} | Direct: {row['direct_normal_irradiance_wm2']:.1f} | Total: {row['total_irradiance_wm2']:.1f} | 比例: {ratio:.2f}x")

# 2. 分析异常发生的时段
print('\n【异常发生时段分布】')
hour_dist = direct_gt_total['hour'].value_counts().sort_index()
print('Direct > Total 的小时分布:')
for hour, count in hour_dist.items():
    pct = count / len(direct_gt_total) * 100
    print(f"  {hour:>2}:00 - {count:>5} 条 ({pct:>5.1f}%)")

# 3. GTI < GHI 案例
print('\n【GTI < GHI 异常案例】')
gti_lt_ghi = df[df['total_irradiance_wm2'] < df['global_horizontal_irradiance_wm2']]
print(f'总数: {len(gti_lt_ghi):,} 条')

if len(gti_lt_ghi) > 0:
    sample = gti_lt_ghi[['timestamp', 'total_irradiance_wm2', 'global_horizontal_irradiance_wm2']].head(10)
    print('\n前10个案例:')
    for _, row in sample.iterrows():
        diff = row['global_horizontal_irradiance_wm2'] - row['total_irradiance_wm2']
        print(f"  {row['timestamp']} | GTI: {row['total_irradiance_wm2']:.1f} | GHI: {row['global_horizontal_irradiance_wm2']:.1f} | 差值: {diff:.1f}")

# 4. 辐照度相关性分析
print('\n【辐照度相关性分析】')
corr_gti_ghi = df['total_irradiance_wm2'].corr(df['global_horizontal_irradiance_wm2'])
corr_gti_dni = df['total_irradiance_wm2'].corr(df['direct_normal_irradiance_wm2'])
corr_ghi_dni = df['global_horizontal_irradiance_wm2'].corr(df['direct_normal_irradiance_wm2'])
print(f'GTI 与 GHI 相关系数: {corr_gti_ghi:.4f}')
print(f'GTI 与 DNI 相关系数: {corr_gti_dni:.4f}')
print(f'GHI 与 DNI 相关系数: {corr_ghi_dni:.4f}')

# 5. 分小时辐照度比例
print('\n【分小时 GTI/GHI 比例】')
df_daytime = df[df['hour'].between(8, 17)]
hourly_ratio = df_daytime.groupby('hour').apply(
    lambda x: (x['total_irradiance_wm2'] / (x['global_horizontal_irradiance_wm2'] + 1)).mean()
)
for hour, ratio in hourly_ratio.items():
    print(f"  {hour:>2}:00 - GTI/GHI比例: {ratio:.3f}")

# 6. 分析功率-辐照度滞后
print('\n' + '='*80)
print('功率-辐照度滞后分析')
print('='*80)

# 分时段相关性
print('\n【分时段功率-辐照度相关性】')
for hour in range(6, 20):
    mask = df['hour'] == hour
    if mask.sum() > 10:
        corr = df.loc[mask, 'power_mw'].corr(df.loc[mask, 'total_irradiance_wm2'])
        print(f"  {hour:>2}:00 - 相关系数: {corr:.4f}")

# 7. 分析中午时段滞后
print('\n【中午时段(10-14点)详细分析】')
df_noon = df[df['hour'].between(10, 14)].copy()
df_noon = df_noon.sort_values('timestamp')

# 计算斜率
gti_values = df_noon['total_irradiance_wm2'].values
power_values = df_noon['power_mw'].values

# 归一化
gti_norm = (gti_values - gti_values.min()) / (gti_values.max() - gti_values.min() + 1)
power_norm = (power_values - power_values.min()) / (power_values.max() - power_values.min() + 1)

# 找最大相关滞后
best_lag = 0
best_corr = 0
for lag in range(-12, 13):
    if lag < 0:
        c = np.corrcoef(gti_norm[:lag], power_norm[-lag:])[0, 1]
    elif lag > 0:
        c = np.corrcoef(gti_norm[lag:], power_norm[:-lag])[0, 1]
    else:
        c = np.corrcoef(gti_norm, power_norm)[0, 1]
    if abs(c) > abs(best_corr):
        best_corr = c
        best_lag = lag

print(f"  最优滞后: {best_lag * 15} 分钟")
print(f"  对应相关系数: {best_corr:.4f}")
