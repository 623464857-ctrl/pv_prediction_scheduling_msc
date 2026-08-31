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
print("Validation: DNI vs extraterrestrial * cos(zenith)")
print("="*80)

if 'solar_zenith_angle_deg' in df.columns:
    zenith = df['solar_zenith_angle_deg'].values
    cos_zenith = np.cos(np.radians(zenith)).clip(0, 1)
    
    # daytime mask (zenith < 80 deg)
    daytime_mask = cos_zenith > 0.17
    
    # DNI_physical = TSI * cos(zenith) (ideal, transmittance = 1)
    dni_physical = df[TSI].values * cos_zenith
    dni_actual = df[DNI].values
    
    # daytime analysis
    dni_physical_day = dni_physical[daytime_mask]
    dni_actual_day = dni_actual[daytime_mask]
    
    print(f"\nDaytime data points: {daytime_mask.sum():,}")
    
    # Transmittance = DNI_actual / DNI_physical
    with np.errstate(divide='ignore', invalid='ignore'):
        transmittance = dni_actual_day / (dni_physical_day + 0.1)
        transmittance = np.clip(transmittance, 0, 1.5)
    
    valid_trans = transmittance[~np.isnan(transmittance) & ~np.isinf(transmittance)]
    
    print(f"\nAtmospheric Transmittance stats:")
    print(f"  min: {valid_trans.min():.3f}")
    print(f"  max: {valid_trans.max():.3f}")
    print(f"  mean: {valid_trans.mean():.3f}")
    print(f"  median: {np.median(valid_trans):.3f}")
    
    # Check for anomalies
    dni_gt_physical = (dni_actual_day > dni_physical_day).sum()
    print(f"\nDNI > TSI*cos(zenith) count: {dni_gt_physical} ({dni_gt_physical/len(dni_actual_day)*100:.1f}%)")
    
    # Correlation
    corr = np.corrcoef(dni_actual_day, dni_physical_day)[0, 1]
    print(f"\nCorrelation (DNI, TSI*cos): {corr:.4f}")
    
    # Ratio
    ratio = dni_actual_day / (dni_physical_day + 0.1)
    valid_ratio = ratio[~np.isnan(ratio) & ~np.isinf(ratio)]
    print(f"DNI/(TSI*cos) ratio: min={valid_ratio.min():.3f}, max={valid_ratio.max():.3f}, mean={valid_ratio.mean():.3f}")

# ========================================
# Summary
# ========================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print("""
Physical relationship: DNI = extraterrestrial * cos(zenith) * transmittance

If TSI = extraterrestrial_irradiance, then DNI / (TSI * cos) = atmospheric_transmittance

Normal atmospheric transmittance: 0.5-0.85 (clear sky)

If computed transmittance is within range, it means:
  - TSI column is indeed extraterrestrial irradiance
  - Data should be renamed to extraterrestrial_irradiance_wm2
  - DNI is the ground measurement of direct normal irradiance
""")
