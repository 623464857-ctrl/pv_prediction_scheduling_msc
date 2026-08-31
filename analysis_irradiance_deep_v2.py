"""
辐照度深度分析 - 理解各列含义 (修复版)
================================
"""

import pandas as pd
import numpy as np

df = pd.read_csv('data/prediction/step1_preprocessing/processed/stations/Site_1_optimized.csv', parse_dates=['timestamp'])

TSI = 'total_irradiance_wm2'
DNI = 'direct_normal_irradiance_wm2'
GHI = 'global_horizontal_irradiance_wm2'

print("="*80)
print("辐照度列数值范围分析")
print("="*80)

print("\n全天统计:")
for col in [TSI, DNI, GHI]:
    print(f"  {col}: min={df[col].min():.1f}, max={df[col].max():.1f}, mean={df[col].mean():.1f}")

# 日间统计
daytime = df[df['hour'].between(10, 14)].copy()
print(f"\n日间(10-14h)统计:")
for col in [TSI, DNI, GHI]:
    print(f"  {col}: min={daytime[col].min():.1f}, max={daytime[col].max():.1f}, mean={daytime[col].mean():.1f}")

print("\n" + "="*80)
print("TSI 最大值分析")
print("="*80)

daytime_max_tsi = daytime[TSI].max()
print(f"日间 TSI 最大值: {daytime_max_tsi:.1f} W/m2")
print(f"理论大气层外辐照度: ~1361 W/m2")

# ========================================
# 验证 TSI 与 cos(zenith) 的关系
# ========================================
print("\n" + "="*80)
print("验证 TSI 与 DNI 的物理关系")
print("="*80)

if 'solar_zenith_angle_deg' in df.columns:
    zenith = df['solar_zenith_angle_deg'].copy()
    cos_zenith = np.cos(np.radians(zenith)).clip(0, 1)
    
    # DNI_physical = TSI * cos(zenith)
    dni_physical = df[TSI].values * cos_zenith.values
    
    # 比较 DNI_physical 与实际 DNI
    valid_mask = (df[DNI] > 10) & (dni_physical > 10)  # 只用有效值
    if valid_mask.sum() > 100:
        corr_dni_vs_physical = np.corrcoef(
            df.loc[valid_mask, DNI].values,
            dni_physical[valid_mask]
        )[0, 1]
        print(f"DNI 与 TSI*cos(zenith) 的相关性: {corr_dni_vs_physical:.4f}")
        
        # 日间相关性
        daytime_idx = daytime.index
        valid_daytime = (daytime[DNI] > 10) & (dni_physical[daytime_idx] > 10)
        if valid_daytime.sum() > 100:
            corr_daytime = np.corrcoef(
                daytime.loc[valid_daytime, DNI].values,
                dni_physical[daytime_idx][valid_daytime.values]
            )[0, 1]
            print(f"日间 DNI 与 TSI*cos(zenith) 的相关性: {corr_daytime:.4f}")
        
        if corr_daytime > 0.8 or corr_dni_vs_physical > 0.8:
            print("\n** 高度相关! TSI 很可能是 extraterrestrial irradiance **")
            print("** DNI = TSI * cos(zenith) **")

# ========================================
# GHI 与 TSI*cos(zenith) 的关系 (Clearness Index)
# ========================================
print("\n" + "="*80)
print("Clearness Index 分析 (GHI / (TSI * cos(zenith)))")
print("="*80)

if 'solar_zenith_angle_deg' in df.columns:
    cos_zenith = np.cos(np.radians(zenith)).clip(0, 1)
    
    # Clearness index = GHI / (TSI * cos(zenith))
    denominator = df[TSI].values * cos_zenith.values + 0.1
    clearness = df[GHI].values / denominator
    
    print(f"Clearness index: min={clearness.min():.2f}, max={clearness.max():.2f}, mean={clearness.mean():.2f}")
    
    # 正常 clearness index 在 0-1 之间
    invalid_clearness = (clearness > 1).sum()
    print(f"Clearness > 1 的次数: {invalid_clearness} ({invalid_clearness/len(df)*100:.1f}%)")

# ========================================
# 最终结论
# ========================================
print("\n" + "="*80)
print("最终结论")
print("="*80)

print(f"""
关键发现:
---------
TSI 列最大值: {daytime_max_tsi:.0f} W/m2
理论大气层外辐照度: 1361 W/m2

TSI 很可能是 "Extraterrestrial Irradiance" (大气层外辐照度)!

这是到达大气层顶的太阳辐照度，不是地面测量值。

数据命名问题:
- total_irradiance_wm2 实际 = extraterrestrial_irradiance (大气层外)
- direct_normal_irradiance_wm2 = DNI (法向直接辐照度) [地面测量]
- global_horizontal_irradiance_wm2 = GHI (水平全球辐照度) [地面测量]

物理关系验证:
- DNI = extraterrestrial * cos(zenith) * atmospheric_transmittance
- GHI = extraterrestrial * cos(zenith) * clearness_index

原始数据问题:
- 三类辐照度物理关系不协调是因为列名误导
- TSI 不是地面总辐照度，而是大气层外辐照度
- 需要重新理解各列的实际含义

建议:
1. 将 total_irradiance_wm2 重命名为 extraterrestrial_irradiance_wm2
2. 验证 extraterrestrial * cos(zenith) 与 DNI 的比例关系
3. 优先使用 GHI 作为建模特征 (最可靠的地面测量)
4. 考虑创建 derived features: clearness_index, atmospheric_transmittance
""")
