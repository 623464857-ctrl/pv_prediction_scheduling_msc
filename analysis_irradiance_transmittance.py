"""
验证 DNI = extraterrestrial * cos(zenith) * transmittance
"""

import pandas as pd
import numpy as np

df = pd.read_csv('data/prediction/step1_preprocessing/processed/stations/Site_1_optimized.csv', parse_dates=['timestamp'])

TSI = 'total_irradiance_wm2'
DNI = 'direct_normal_irradiance_wm2'
GHI = 'global_horizontal_irradiance_wm2'

print("="*80)
print("验证 DNI 与 TSI*cos(zenith) 的物理关系")
print("="*80)

if 'solar_zenith_angle_deg' in df.columns:
    zenith = df['solar_zenith_angle_deg'].values
    cos_zenith = np.cos(np.radians(zenith)).clip(0, 1)
    
    # 日间掩码 (天顶角 < 80度)
    daytime_mask = cos_zenith > 0.17  # cos(80°) ≈ 0.17
    
    # DNI_physical = TSI * cos(zenith) (理想情况，假设 transmittance = 1)
    dni_physical = df[TSI].values * cos_zenith
    
    # 实际 DNI
    dni_actual = df[DNI].values
    
    # 日间分析
    dni_physical_day = dni_physical[daytime_mask]
    dni_actual_day = dni_actual[daytime_mask]
    
    print(f"\n日间数据点: {daytime_mask.sum():,} 条")
    
    # 大气透射率 (transmittance)
    # DNI_actual = DNI_physical * transmittance
    # transmittance = DNI_actual / DNI_physical
    with np.errstate(divide='ignore', invalid='ignore'):
        transmittance = dni_actual_day / (dni_physical_day + 0.1)
        transmittance = np.clip(transmittance, 0, 1.5)  # 限制在合理范围
    
    valid_trans = transmittance[~np.isnan(transmittance) & ~np.isinf(transmittance)]
    
    print(f"\n大气透射率 (Transmittance) 统计:")
    print(f"  min: {valid_trans.min():.3f}")
    print(f"  max: {valid_trans.max():.3f}")
    print(f"  mean: {valid_trans.mean():.3f}")
    print(f"  median: {np.median(valid_trans):.3f}")
    
    # 正常大气透射率在 0.5-0.8 之间
    print(f"\n合理透射率 (0.3-1.0) 的比例: {((valid_trans >= 0.3) & (valid_trans <= 1.0)).sum() / len(valid_trans) * 100:.1f}%")
    
    # 检查 DNI > TSI * cos(zenith) 的情况
    dni_gt_physical = (dni_actual_day > dni_physical_day).sum()
    print(f"\nDNI > TSI*cos(zenith) 的次数: {dni_gt_physical} ({dni_gt_physical/len(dni_actual_day)*100:.1f}%)")
    
    # 相关性分析
    corr = np.corrcoef(dni_actual_day, dni_physical_day)[0, 1]
    print(f"\nDNI 与 TSI*cos(zenith) 的相关性: {corr:.4f}")
    
    # 比例分析
    ratio = dni_actual_day / (dni_physical_day + 0.1)
    valid_ratio = ratio[~np.isnan(ratio) & ~np.isinf(ratio)]
    print(f"\nDNI / (TSI*cos) 比例: min={valid_ratio.min():.3f}, max={valid_ratio.max():.3f}, mean={valid_ratio.mean():.3f}")

# ========================================
# 结论
# ========================================
print("\n" + "="*80)
print("结论")
print("="*80)

print("""
根据物理关系:
  DNI = extraterrestrial_irradiance * cos(zenith) * atmospheric_transmittance

如果 TSI = extraterrestrial_irradiance，则 DNI / (TSI * cos) = atmospheric_transmittance

大气透射率正常范围: 0.5-0.85 (晴天)

如果计算出的 transmittance 在合理范围内，说明:
  - TSI 列确实是大气层外辐照度 (extraterrestrial irradiance)
  - 数据命名有误，应该叫 extraterrestrial_irradiance_wm2
  - DNI 是法向直接辐照度的地面测量值

如果 transmittance 异常 (比如 > 1 或接近 0)，说明:
  - 可能存在传感器校准问题
  - 或者 TSI 不是 extraterrestrial irradiance
""")
