"""
分析 GHI 与功率的关系
"""
import pandas as pd
import numpy as np

df = pd.read_csv('data/prediction/step1_preprocessing/processed/stations/Site_1_optimized.csv', parse_dates=['timestamp'])

noon = df[df['hour'].between(10, 14)].copy()

print("Power by GHI bins (noon hours 10-14h):")
print("="*60)
print("GHI Range       Count    Mean Power   Max Power")
print("-"*60)

bins = [(0, 100), (100, 200), (200, 400), (400, 600), (600, 800), (800, 1000)]
for low, high in bins:
    mask = (noon['global_horizontal_irradiance_wm2'] >= low) & (noon['global_horizontal_irradiance_wm2'] < high)
    subset = noon[mask]
    if len(subset) > 0:
        print(f"{low}-{high} W/m2     {len(subset):<8} {subset.power_mw.mean():<12.1f} {subset.power_mw.max():<12.1f}")

print("\n\nPower by TSI bins (noon hours 10-14h):")
print("="*60)
print("TSI Range       Count    Mean Power   Max Power")
print("-"*60)

bins = [(0, 200), (200, 400), (400, 600), (600, 800), (800, 1000), (1000, 1200), (1200, 1400)]
for low, high in bins:
    mask = (noon['total_irradiance_wm2'] >= low) & (noon['total_irradiance_wm2'] < high)
    subset = noon[mask]
    if len(subset) > 0:
        print(f"{low}-{high} W/m2     {len(subset):<8} {subset.power_mw.mean():<12.1f} {subset.power_mw.max():<12.1f}")

# Check theoretical_power
if 'theoretical_power' in df.columns:
    print("\n\nTheoretical Power vs Actual Power correlation (noon):")
    corr = noon['power_mw'].corr(noon['theoretical_power'])
    print(f"Correlation: {corr:.4f}")

# Check DNI
print("\n\nPower by DNI bins (noon hours 10-14h):")
print("="*60)
print("DNI Range       Count    Mean Power   Max Power")
print("-"*60)

bins = [(0, 50), (50, 100), (100, 200), (200, 400), (400, 600), (600, 1000)]
for low, high in bins:
    mask = (noon['direct_normal_irradiance_wm2'] >= low) & (noon['direct_normal_irradiance_wm2'] < high)
    subset = noon[mask]
    if len(subset) > 0:
        print(f"{low}-{high} W/m2     {len(subset):<8} {subset.power_mw.mean():<12.1f} {subset.power_mw.max():<12.1f}")
