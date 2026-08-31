"""
深度分析 TSI 物理含义
"""
import pandas as pd
import numpy as np

df = pd.read_csv('data/prediction/step1_preprocessing/processed/stations/Site_1_optimized.csv', parse_dates=['timestamp'])

TSI = 'total_irradiance_wm2'
DNI = 'direct_normal_irradiance_wm2'
GHI = 'global_horizontal_irradiance_wm2'

# Find the row with max TSI
max_idx = df[TSI].idxmax()
row = df.loc[max_idx]
print('Row with max TSI (1359):')
print('  timestamp:', row['timestamp'])
print('  hour:', row['hour'], 'minute:', row['minute'])
print('  TSI: %.1f' % row[TSI])
print('  DNI: %.1f' % row[DNI])
print('  GHI: %.1f' % row[GHI])
print('  power: %.1f' % row['power_mw'])
print()

# Check daytime rows where DNI > TSI
daytime = df[df['hour'].between(10, 14)]
dni_gt_tsi = daytime[daytime[DNI] > daytime[TSI]]
print('DNI > TSI in daytime (10-14h): %d rows out of %d (%.1f%%)' % (len(dni_gt_tsi), len(daytime), len(dni_gt_tsi)/len(daytime)*100))
if len(dni_gt_tsi) > 0:
    sample = dni_gt_tsi.head(5)
    print()
    print('Sample rows where DNI > TSI:')
    print(sample[['timestamp', TSI, DNI, GHI, 'power_mw']].to_string())

# Also check TSI=1359 distribution
print()
print('TSI = 1359 count:', (df[TSI] == 1359).sum())
print('TSI >= 1300 count:', (df[TSI] >= 1300).sum())
print('Solar constant is ~1361 W/m2')

# Check relationship: TSI vs GHI+DNI
print()
print('=== TSI vs GHI + DNI ===')
daytime_nonzero = daytime[daytime[TSI] > 10]
print('Daytime non-zero rows: %d' % len(daytime_nonzero))
# Check if TSI looks like GHI + diffuse
# GHI = DNI * cos(zenith) + DHI
# But we don't have zenith directly. Let's check if TSI ~= GHI / cos(zenith) for tilted surface

# Check TSI distribution by hour
print()
print('=== TSI mean by hour ===')
for hour in range(0, 24):
    mask = df['hour'] == hour
    if mask.sum() > 100:
        tsi_mean = df.loc[mask, TSI].mean()
        ghi_mean = df.loc[mask, GHI].mean()
        dni_mean = df.loc[mask, DNI].mean()
        pwr_mean = df.loc[mask, 'power_mw'].mean()
        print('  Hour %02d: TSI=%.1f, GHI=%.1f, DNI=%.1f, Power=%.1f' % (hour, tsi_mean, ghi_mean, dni_mean, pwr_mean))

# Check source data
print()
print('=== Source file info ===')
print('source_file unique:', df['source_file'].unique()[:5])

# Check if TSI is derived from GHI using some angle correction
# Check GHI/TSI ratio
daytime_nonzero_tsi = daytime[daytime[TSI] > 10]
if len(daytime_nonzero_tsi) > 0:
    ratio = daytime_nonzero_tsi[GHI] / daytime_nonzero_tsi[TSI]
    print()
    print('=== GHI/TSI ratio (daytime non-zero) ===')
    print('  mean: %.3f' % ratio.mean())
    print('  min:  %.3f' % ratio.min())
    print('  max:  %.3f' % ratio.max())
    print('  If TSI is TOA: GHI/TSI should be ~0.45 (atmospheric transmission)')
    print('  If TSI is tilted GHI: GHI/TSI should be < 1.0 and vary with sun position')

# Check TSI vs theoretical_max_gti
print()
print('=== TSI vs theoretical_max_gti ===')
df_theo = df[df['theoretical_max_gti'].notna()]
mask = df_theo['theoretical_max_gti'] > 0
print('Rows with theoretical_max_gti > 0:', mask.sum())
if mask.sum() > 0:
    corr = df_theo.loc[mask, TSI].corr(df_theo.loc[mask, 'theoretical_max_gti'])
    print('Correlation TSI vs theoretical_max_gti: %.4f' % corr)
