"""
验证 TSI 物理含义
"""
import pandas as pd
import numpy as np

df = pd.read_csv('data/prediction/step1_preprocessing/processed/stations/Site_1_optimized.csv', parse_dates=['timestamp'])

TSI = 'total_irradiance_wm2'
DNI = 'direct_normal_irradiance_wm2'
GHI = 'global_horizontal_irradiance_wm2'

print("=== TSI Physical Validation ===")
print()
print("TSI stats (all day):")
print(f"  min={df[TSI].min():.1f}, max={df[TSI].max():.1f}, mean={df[TSI].mean():.1f}")
print()
print("TSI stats (daytime, hour 10-14):")
daytime = df[df['hour'].between(10, 14)]
print(f"  min={daytime[TSI].min():.1f}, max={daytime[TSI].max():.1f}, mean={daytime[TSI].mean():.1f}")
print()
print("GHI stats (daytime, hour 10-14):")
print(f"  min={daytime[GHI].min():.1f}, max={daytime[GHI].max():.1f}, mean={daytime[GHI].mean():.1f}")
print()
print("DNI stats (daytime, hour 10-14):")
print(f"  min={daytime[DNI].min():.1f}, max={daytime[DNI].max():.1f}, mean={daytime[DNI].mean():.1f}")
print()

# TSI vs Power correlation
print("=== TSI vs Power Correlation ===")
full_corr = df[TSI].corr(df["power_mw"])
daytime_corr = daytime[TSI].corr(daytime["power_mw"])
print(f"Full day correlation: {full_corr:.4f}")
print(f"Daytime (10-14h) correlation: {daytime_corr:.4f}")
print()

# By hour analysis
print("By hour analysis (TSI-Power correlation):")
for hour in range(6, 20):
    mask = df['hour'] == hour
    if mask.sum() > 100:
        corr = df.loc[mask, TSI].corr(df.loc[mask, 'power_mw'])
        print(f"  Hour {hour:02d}:00 - Correlation = {corr:.4f}")

# Physical relationship check
print()
print("=== Physical Relationship Check ===")
print("Expected: GHI <= TSI (if TSI = tilted irradiance)")
print("          GHI >= DNI * cos(zenith)")
print()
n_gti_lt_ghi = (df[TSI] < df[GHI]).sum()
print(f"TSI < GHI count: {n_gti_lt_ghi} ({n_gti_lt_ghi/len(df)*100:.1f}%)")

n_dni_gt_tsi = (df[DNI] > df[TSI]).sum()
print(f"DNI > TSI count: {n_dni_gt_tsi} ({n_dni_gt_tsi/len(df)*100:.1f}%)")
