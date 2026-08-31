"""
辐照度深度分析 - 理解各列含义
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
    print(f"  {col}:")
    print(f"    min={df[col].min():.1f}, max={df[col].max():.1f}, mean={df[col].mean():.1f}")

# 日间统计
daytime = df[df['hour'].between(10, 14)]
print(f"\n日间(10-14h)统计:")
for col in [TSI, DNI, GHI]:
    print(f"  {col}:")
    print(f"    min={daytime[col].min():.1f}, max={daytime[col].max():.1f}, mean={daytime[col].mean():.1f}")

# 理论大气层外辐照度约为 1361 W/m2
# 如果 TSI 接近这个值，可能是 extraterrestrial irradiance
print("\n" + "="*80)
print("TSI 列可能是 extraterrestrial irradiance (大气层外辐照度)?")
print("="*80)

# 检查日间 TSI 最大值
daytime_max_tsi = daytime[TSI].max()
print(f"日间 TSI 最大值: {daytime_max_tsi:.1f} W/m2")
print(f"理论大气层外辐照度: ~1361 W/m2")
print(f"实际地面最大辐照度: ~1000 W/m2")

if daytime_max_tsi > 1300:
    print("\n** TSI 列可能存储的是大气层外辐照度 (extraterrestrial) **")
    print("** 这不是地面实际测量值 **")
elif daytime_max_tsi > 1000:
    print("\n** TSI 接近地面最大辐照度，可能存储的是理论最大值 **")

# ========================================
# 如果 TSI 是 extraterrestrial，则验证与天顶角的关系
# ========================================
print("\n" + "="*80)
print("验证 TSI 与 cos(zenith) 的关系")
print("="*80)

if 'solar_zenith_angle_deg' in df.columns:
    zenith = df['solar_zenith_angle_deg']
    cos_zenith = np.cos(np.radians(zenith)).clip(0, 1)
    
    # 如果 TSI 是 extraterrestrial，则 DNI_physical = TSI * cos(zenith)
    dni_physical = df[TSI] * cos_zenith
    
    # 比较 DNI_physical 与实际 DNI
    corr_dni_vs_physical = df[DNI].corr(dni_physical)
    print(f"DNI 与 TSI*cos(zenith) 的相关性: {corr_dni_vs_physical:.4f}")
    
    # 日间相关性
    corr_daytime = daytime[DNI].corr(dni_physical[daytime.index])
    print(f"日间 DNI 与 TSI*cos(zenith) 的相关性: {corr_daytime:.4f}")
    
    if corr_daytime > 0.8:
        print("\n** 高度相关! TSI 很可能是 extraterrestrial irradiance **")
        print("** DNI ≈ TSI * cos(zenith) **")

# ========================================
# GHI 与 TSI*cos(zenith) 的关系
# ========================================
print("\n" + "="*80)
print("验证 GHI 与 TSI*cos(zenith) 的关系")
print("="*80)

if 'solar_zenith_angle_deg' in df.columns:
    zenith = df['solar_zenith_angle_deg']
    cos_zenith = np.cos(np.radians(zenith)).clip(0, 1)
    
    # Clearness index = GHI / (TSI * cos(zenith))
    clearness = df[GHI] / (df[TSI] * cos_zenith + 0.1)
    
    print(f"Clearness index (GHI/(TSI*cos)): min={clearness.min():.2f}, max={clearness.max():.2f}, mean={clearness.mean():.2f}")
    
    # 正常 clearness index 在 0-1 之间
    invalid_clearness = (clearness > 1).sum()
    print(f"Clearness > 1 的次数: {invalid_clearness} ({invalid_clearness/len(df)*100:.1f}%)")

# ========================================
# 最终结论
# ========================================
print("\n" + "="*80)
print("最终结论")
print("="*80)

print("""
基于数据分析，TSI 列的物理含义可能是:

选项1: TSI = Extraterrestrial Irradiance (大气层外辐照度)
  - 证据: 日间最大值 > 1300 W/m2
  - 验证: DNI ≈ TSI * cos(zenith) (相关性 > 0.8)
  - 建议: 重命名为 extraterrestrial_irradiance_wm2

选项2: TSI = 理论最大直接辐照度
  - 证据: 数值介于 DNI 和 extraterrestrial 之间
  - 验证: 与 DNI 相关性高
  - 建议: 谨慎使用

推荐解决方案:
1. 将 total_irradiance_wm2 重命名为 extraterrestrial_irradiance_wm2
2. 创建实际地面辐照度估算: ghi_estimated = extraterrestrial * cos(zenith) * clearness
3. 使用 GHI 作为主要特征，因为它是最可靠的地面测量
4. 避免直接使用 TSI，除非已正确理解其来源
""")
