"""
Site1 辐照度物理一致性修复
==========================
修复问题：
1. GTI命名可能错误 → 验证并重新定义
2. 夜间Direct>Total → 设置夜间为0
3. 辐照度物理关系校正

运行：
python experiments/prediction/step1_data_cleaning_alignment/run_fix_irradiance_consistency.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime


def verify_irradiance_columns(df: pd.DataFrame) -> dict:
    """
    验证辐照度列的实际含义
    
    物理关系：
    - GHI = DNI × cos(zenith) + DHI
    - GTI ≈ GHI (倾斜平面总辐照)
    """
    results = {}
    
    # 获取数据
    gti = df['total_irradiance_wm2']
    ghi = df['global_horizontal_irradiance_wm2']
    dni = df['direct_normal_irradiance_wm2']
    
    # 计算理论 DNI*cos(zenith)
    if 'solar_zenith_angle_deg' in df.columns:
        zenith = df['solar_zenith_angle_deg']
        cos_zenith = np.cos(np.radians(zenith)).clip(0, 1)
        dni_cos = dni * cos_zenith
        
        # 测试GTI是否是DNI*cos(zenith)
        corr_gti_dni_cos = np.corrcoef(gti, dni_cos)[0, 1]
        results['corr_GTI_vs_DNI_cos'] = corr_gti_dni_cos
        
        # 测试GTI是否是GHI
        corr_gti_ghi = np.corrcoef(gti, ghi)[0, 1]
        results['corr_GTI_vs_GHI'] = corr_gti_ghi
        
        # 测试GHI是否=DNI*cos+DHI
        if 'diffuse_horizontal_irradiance_wm2' in df.columns:
            dhi = df['diffuse_horizontal_irradiance_wm2']
            ghi_theoretical = dni_cos + dhi
            corr_ghi_vs_theory = np.corrcoef(ghi, ghi_theoretical)[0, 1]
            results['corr_GHI_vs_DNI_cos_plus_DHI'] = corr_ghi_vs_theory
    
    # 日间相关性
    daytime_mask = df['hour'].between(10, 14)
    if daytime_mask.sum() > 100:
        results['daytime_mean_GTI'] = gti[daytime_mask].mean()
        results['daytime_mean_GHI'] = ghi[daytime_mask].mean()
        results['daytime_ratio_GTI_GHI'] = gti[daytime_mask].mean() / (ghi[daytime_mask].mean() + 1)
    
    return results


def fix_irradiance_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """
    修复辐照度物理一致性
    """
    df = df.copy()
    
    print("=" * 80)
    print("辐照度物理一致性修复")
    print("=" * 80)
    
    # ========================================
    # 1. 夜间辐照度设为0
    # ========================================
    print("\n[Step 1] 夜间辐照度校正...")
    
    # 辐照度很低的时段视为夜间
    low_irradiance_mask = df['total_irradiance_wm2'] < 5
    
    # 夜间时所有辐照度应接近0
    nighttime_irradiance = {
        'total_irradiance_wm2': 0,
        'direct_normal_irradiance_wm2': 0,
        'global_horizontal_irradiance_wm2': 0,
        'diffuse_horizontal_irradiance_wm2': 0
    }
    
    for col, value in nighttime_irradiance.items():
        if col in df.columns:
            before_count = (df[col] > 0).sum()
            df.loc[low_irradiance_mask & (df[col] > 0), col] = 0
            after_count = (df[col] > 0).sum()
            print(f"  {col}: 清零了 {(before_count - after_count):,} 个夜间正值")
    
    # ========================================
    # 2. 验证GTI实际含义
    # ========================================
    print("\n[Step 2] 验证辐照度列含义...")
    
    verification = verify_irradiance_columns(df)
    
    print(f"  GTI 与 DNI×cos(zenith) 相关性: {verification.get('corr_GTI_vs_DNI_cos', 'N/A'):.4f}")
    print(f"  GTI 与 GHI 相关性: {verification.get('corr_GTI_vs_GHI', 'N/A'):.4f}")
    
    # 判断GTI实际代表什么
    corr_dni_cos = verification.get('corr_GTI_vs_DNI_cos', 0)
    corr_ghi = verification.get('corr_GTI_vs_GHI', 0)
    
    if corr_dni_cos > corr_ghi + 0.1:
        print(f"\n  结论: GTI 可能代表 DNI × cos(zenith) (直射分量)")
        print(f"  建议: 将 GTI 重命名为 gti_direct_component")
        df = df.rename(columns={'total_irradiance_wm2': 'gti_direct_component_wm2'})
        gti_col = 'gti_direct_component_wm2'
    else:
        print(f"\n  结论: GTI 代表总辐照度")
        gti_col = 'total_irradiance_wm2'
    
    # ========================================
    # 3. 创建物理一致性特征
    # ========================================
    print("\n[Step 3] 创建物理一致性验证特征...")
    
    if 'global_horizontal_irradiance_wm2' in df.columns:
        ghi = df['global_horizontal_irradiance_wm2']
        
        # 创建"正确"的GHI估算值
        if 'solar_zenith_angle_deg' in df.columns:
            zenith = df['solar_zenith_angle_deg']
            cos_zenith = np.cos(np.radians(zenith)).clip(0, 1)
            dni = df['direct_normal_irradiance_wm2']
            dhi = df.get('diffuse_horizontal_irradiance_wm2', ghi * 0.4)  # 估计DHI
            
            df['ghi_physical'] = (dni * cos_zenith + dhi).clip(lower=0)
            
            # 物理一致性偏差
            df['ghi_consistency_error'] = (df['ghi_physical'] - ghi).abs()
    
    # ========================================
    # 4. 添加辐照度质量标志
    # ========================================
    print("\n[Step 4] 添加辐照度质量标志...")
    
    df['irradiance_physical_valid'] = 1
    
    # Direct > Total 异常
    if 'direct_normal_irradiance_wm2' in df.columns and gti_col in df.columns:
        df.loc[df['direct_normal_irradiance_wm2'] > df[gti_col], 'irradiance_physical_valid'] = 0
    
    # GHI < 0 异常
    if 'global_horizontal_irradiance_wm2' in df.columns:
        df.loc[df['global_horizontal_irradiance_wm2'] < 0, 'irradiance_physical_valid'] = 0
    
    # DNI < 0 异常
    if 'direct_normal_irradiance_wm2' in df.columns:
        df.loc[df['direct_normal_irradiance_wm2'] < 0, 'irradiance_physical_valid'] = 0
    
    valid_count = (df['irradiance_physical_valid'] == 1).sum()
    print(f"  物理有效辐照度: {valid_count:,} / {len(df):,} ({valid_count/len(df)*100:.1f}%)")
    
    # ========================================
    # 5. 推荐使用的辐照度特征
    # ========================================
    print("\n[Step 5] 辐照度特征推荐...")
    
    # 建议：优先使用GHI，因为它是最可靠的测量
    print("  推荐使用:")
    print("    1. global_horizontal_irradiance_wm2 (GHI) - 最可靠的测量")
    print("    2. gti_direct_component_wm2 - 如果相关性高则可用")
    print("    3. 避免直接使用 DNI - 与GHI相关性极低(0.21)")
    
    return df


def add_lag_features_for_power(df: pd.DataFrame) -> pd.DataFrame:
    """
    添加滞后特征解决功率-辐照度响应延迟问题
    
    原因分析：
    - 中午时段功率-辐照度相关性下降至0.54
    - 这是由于热惯性和MPPT跟踪延迟
    """
    df = df.copy()
    
    print("\n" + "=" * 80)
    print("功率-辐照度滞后补偿")
    print("=" * 80)
    
    # 选择辐照度列（优先GHI）
    gti_col = 'global_horizontal_irradiance_wm2'
    if gti_col not in df.columns:
        gti_col = 'total_irradiance_wm2'
    
    # 添加滞后特征
    print(f"\n[Step 1] 添加辐照度滞后特征 (使用 {gti_col})...")
    
    for lag in [1, 2, 4]:  # 15min, 30min, 60min
        df[f'gti_lag_{lag}'] = df[gti_col].shift(lag)
        print(f"  添加: gti_lag_{lag} (滞后 {lag*15} 分钟)")
    
    # 添加辐照度变化率
    print(f"\n[Step 2] 添加辐照度变化率特征...")
    df['gti_change_1step'] = df[gti_col].diff()
    df['gti_change_4step'] = df[gti_col].diff(4)  # 1小时变化
    print("  添加: gti_change_1step, gti_change_4step")
    
    # 添加功率变化率
    print(f"\n[Step 3] 添加功率变化率特征...")
    df['power_change_1step'] = df['power_mw'].diff()
    df['power_change_4step'] = df['power_mw'].diff(4)
    print("  添加: power_change_1step, power_change_4step")
    
    # 添加移动平均（平滑热惯性影响）
    print(f"\n[Step 4] 添加移动平均特征...")
    df['gti_ma_4'] = df[gti_col].rolling(4, min_periods=1).mean()
    df['gti_ma_8'] = df[gti_col].rolling(8, min_periods=1).mean()
    df['power_ma_4'] = df['power_mw'].rolling(4, min_periods=1).mean()
    print("  添加: gti_ma_4, gti_ma_8, power_ma_4")
    
    # 计算辐照度-功率效率（用于检测异常）
    print(f"\n[Step 5] 添加效率特征...")
    theoretical_power = df[gti_col] / 1000 * 50 * 0.85  # 85%效率估计
    df['power_efficiency_actual'] = df['power_mw'] / (theoretical_power + 0.1)
    df['power_efficiency_actual'] = df['power_efficiency_actual'].clip(0, 1.2)
    print("  添加: power_efficiency_actual")
    
    # 中午时段特殊处理
    print(f"\n[Step 6] 中午时段特殊特征...")
    df['is_noon'] = df['hour'].between(10, 14).astype(int)
    df['noon_gti_ma'] = df['gti_ma_4'] * df['is_noon']
    print("  添加: is_noon, noon_gti_ma")
    
    return df


def main():
    """主函数"""
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    DATA_PATH = PROJECT_ROOT / "data" / "prediction" / "step1_preprocessing" / "processed" / "stations" / "Site_1_optimized.csv"
    OUTPUT_PATH = DATA_PATH
    
    print(f"\n读取数据: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
    print(f"原始数据: {len(df):,} 行, {len(df.columns)} 列")
    
    # 修复辐照度一致性
    df = fix_irradiance_consistency(df)
    
    # 添加滞后特征
    df = add_lag_features_for_power(df)
    
    # 保存
    print(f"\n保存修复后的数据...")
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"已保存到: {OUTPUT_PATH}")
    print(f"最终数据: {len(df):,} 行, {len(df.columns)} 列")
    
    # 新增列
    new_cols = [c for c in df.columns if 'lag' in c or 'change' in c or 'ma' in c or 'noon' in c or 'physical' in c or 'efficiency' in c]
    print(f"\n新增列 ({len(new_cols)}):")
    for col in new_cols:
        print(f"  - {col}")
    
    print("\n" + "=" * 80)
    print("修复完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
