"""
Verify physics-aware features
=============================

Check:
1. Irradiance physical consistency resolved
2. New features work correctly
3. Noon effect captured properly
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


def verify_irradiance_consistency():
    """Verify irradiance physical consistency"""
    
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    DATA_PATH = PROJECT_ROOT / "data" / "prediction" / "step1_preprocessing" / "processed" / "stations" / "Site_1_optimized.csv"
    
    df = pd.read_csv(DATA_PATH)
    
    print("=" * 80)
    print("Verification 1: Irradiance Physical Consistency")
    print("=" * 80)
    
    gti_col = 'total_irradiance_wm2'
    dni_col = 'direct_normal_irradiance_wm2'
    ghi_col = 'global_horizontal_irradiance_wm2'
    
    n_total = len(df)
    
    # Check DNI > GTI
    if dni_col in df.columns and gti_col in df.columns:
        dni_gt_total = (df[dni_col] > df[gti_col]).sum()
        print(f"\nDNI > Total: {dni_gt_total:,} records ({dni_gt_total/n_total*100:.2f}%)")
        
        if dni_gt_total == 0:
            print("OK: No physical contradiction")
        else:
            print("FAIL: Contradiction still exists")
            # Show some cases
            cases = df[df[dni_col] > df[gti_col]][['timestamp', dni_col, gti_col, ghi_col]].head(5)
            print(cases.to_string())
    
    # Check GTI > GHI
    if gti_col in df.columns and ghi_col in df.columns:
        gti_lt_ghi = (df[gti_col] < df[ghi_col]).sum()
        print(f"\nGTI < GHI: {gti_lt_ghi:,} records ({gti_lt_ghi/n_total*100:.2f}%)")
        
        if gti_lt_ghi < 100:
            print("OK: GTI/GHI relationship normal")
        else:
            print("WARN: More inconsistencies found")
    
    # Check DNI*cos(zenith) > GHI
    if dni_col in df.columns and ghi_col in df.columns and 'solar_zenith_angle_deg' in df.columns:
        zenith = df['solar_zenith_angle_deg'].clip(0, 89)
        cos_zenith = np.cos(np.radians(zenith))
        dni_component = df[dni_col] * cos_zenith
        exceeds = (dni_component > df[ghi_col]).sum()
        print(f"\nDNI*cos(zenith) > GHI: {exceeds:,} records ({exceeds/n_total*100:.2f}%)")
        
        if exceeds < 100:
            print("OK: DNI component normal")
        else:
            print("WARN: DNI component issue")


def verify_new_features():
    """Verify new features"""
    
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    DATA_PATH = PROJECT_ROOT / "data" / "prediction" / "step1_preprocessing" / "processed" / "stations" / "Site_1_optimized.csv"
    
    df = pd.read_csv(DATA_PATH)
    
    print("\n" + "=" * 80)
    print("Verification 2: New Feature Statistics")
    print("=" * 80)
    
    feature_groups = {
        'Irradiance Validation': ['dni_gti_ratio', 'irr_physical_score', 'irradiance_quality_score', 'irradiance_low_quality'],
        'Temperature Correction': ['estimated_cell_temp', 'temp_efficiency_factor', 'heat_stress_index', 'theoretical_power_corrected', 'high_temp_warning'],
        'Efficiency': ['instant_efficiency', 'efficiency_rolling_mean', 'low_efficiency_flag', 'power_irradiance_ratio', 'power_residual'],
        'Time Period Interaction': ['solar_period', 'is_noon_period', 'noon_temp_interaction', 'noon_efficiency_interaction']
    }
    
    for group_name, features in feature_groups.items():
        print(f"\n[{group_name}]")
        for f in features:
            if f in df.columns:
                stats = df[f].describe()
                print(f"  {f}:")
                print(f"    mean={stats['mean']:.3f}, std={stats['std']:.3f}, range=[{stats['min']:.3f}, {stats['max']:.3f}]")
            else:
                print(f"  {f}: NOT FOUND")


def verify_noon_effect():
    """Verify noon effect capture"""
    
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    DATA_PATH = PROJECT_ROOT / "data" / "prediction" / "step1_preprocessing" / "processed" / "stations" / "Site_1_optimized.csv"
    
    df = pd.read_csv(DATA_PATH)
    
    print("\n" + "=" * 80)
    print("Verification 3: Noon Effect Analysis")
    print("=" * 80)
    
    if 'hour' not in df.columns:
        print("ERROR: Missing hour column")
        return
    
    if 'instant_efficiency' not in df.columns:
        print("ERROR: Missing instant_efficiency column")
        return
    
    print("\nHourly efficiency statistics (daytime):")
    print(f"{'Hour':<10} {'Count':<10} {'Avg Efficiency':<15} {'Avg Power (MW)':<18} {'Avg GTI':<15}")
    print("-" * 70)
    
    # Daytime only
    for hour in range(6, 20):
        mask = df['hour'] == hour
        if mask.sum() > 0:
            eff = df.loc[mask, 'instant_efficiency'].mean()
            power = df.loc[mask, 'power_mw'].mean() if 'power_mw' in df.columns else 0
            gti = df.loc[mask, 'total_irradiance_wm2'].mean() if 'total_irradiance_wm2' in df.columns else 0
            
            marker = "*" if 11 <= hour <= 14 else ""
            print(f"{hour:>2}:00{marker:<4} {mask.sum():>8,} {eff:>13.3f} {power:>16.2f} {gti:>13.1f}")
    
    print("\nAnalysis:")
    
    # Calculate noon vs other time difference
    if 'is_noon_period' in df.columns:
        noon_mask = df['is_noon_period'] == 1
        non_noon_daytime = (df['hour'].between(7, 17)) & (~noon_mask)
        
        noon_eff = df.loc[noon_mask, 'instant_efficiency'].mean() if noon_mask.sum() > 0 else 0
        non_noon_eff = df.loc[non_noon_daytime, 'instant_efficiency'].mean() if non_noon_daytime.sum() > 0 else 0
        
        print(f"\nNoon period efficiency: {noon_eff:.3f}")
        print(f"Non-noon daytime efficiency: {non_noon_eff:.3f}")
        print(f"Efficiency difference: {(noon_eff - non_noon_eff):.3f}")
        
        if noon_eff < non_noon_eff:
            print("\nOK: Noon efficiency decline confirmed")
            print("   New features like noon_temp_interaction can be used for modeling")
        else:
            print("\nWARN: Expected noon efficiency decline not observed")
            print("   Possible: data was preprocessed or efficiency calculation differs")


def verify_correlations():
    """Verify correlation between new features and target"""
    
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    DATA_PATH = PROJECT_ROOT / "data" / "prediction" / "step1_preprocessing" / "processed" / "stations" / "Site_1_optimized.csv"
    
    df = pd.read_csv(DATA_PATH)
    
    print("\n" + "=" * 80)
    print("Verification 4: Feature Correlation with Power")
    print("=" * 80)
    
    if 'power_mw' not in df.columns:
        print("ERROR: Missing power_mw column")
        return
    
    # Daytime with power
    mask = df['power_mw'] > 0
    
    features_to_check = [
        'total_irradiance_wm2', 'direct_normal_irradiance_wm2',
        'estimated_cell_temp', 'temp_efficiency_factor',
        'instant_efficiency', 'power_irradiance_ratio',
        'noon_temp_interaction', 'noon_efficiency_interaction'
    ]
    
    print(f"\nCorrelation with power_mw (daytime with power):")
    print(f"{'Feature':<30} {'Correlation':<12}")
    print("-" * 45)
    
    for f in features_to_check:
        if f in df.columns:
            corr = df.loc[mask, f].corr(df.loc[mask, 'power_mw'])
            print(f"{f:<30} {corr:>10.4f}")


def main():
    """Run all verifications"""
    
    print("=" * 80)
    print("Site1 Physics-Aware Feature Verification")
    print("=" * 80)
    
    verify_irradiance_consistency()
    verify_new_features()
    verify_noon_effect()
    verify_correlations()
    
    print("\n" + "=" * 80)
    print("Verification Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
