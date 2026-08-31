"""
Site1 辐照度物理一致性校正与增强特征模块
========================================
修复诊断中发现的问题：
1. DNI > Total 的物理矛盾
2. 功率-辐照度中午效应
3. 添加物理一致性检查特征

运行：
python experiments/prediction/step1_data_cleaning_alignment/run_irradiance_physics_correction.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================================
# 物理一致性校正
# ============================================================================

def correct_irradiance_physical_consistency(
    df: pd.DataFrame,
    capacity_mw: float = 50.0
) -> pd.DataFrame:
    """
    对辐照度数据进行物理一致性校正
    
    物理约束：
    1. Total <= GHI + tolerance（晴天时 GTI ≈ GHI）
    2. DNI * cos(天顶角) <= GHI
    3. DNI <= GTI（直射不应超过总辐照）
    4. 所有辐照度 >= 0
    5. GTI <= 理论最大值（太阳常数 * 面积因子）
    
    校正策略：
    - 识别异常值并用物理合理值替代
    - 使用 DNI = GHI - DHI 关系重建
    """
    df = df.copy()
    
    # 太阳常数 (W/m²)
    SOLAR_CONSTANT = 1361
    
    # 列名
    gti_col = 'total_irradiance_wm2'
    dni_col = 'direct_normal_irradiance_wm2'
    ghi_col = 'global_horizontal_irradiance_wm2'
    zenith_col = 'solar_zenith_angle_deg'
    
    # 确保列存在
    required_cols = [gti_col, dni_col, ghi_col]
    if not all(c in df.columns for c in required_cols):
        print(f"警告: 缺少必要列，跳过物理校正")
        return df
    
    # ========================================
    # 1. 处理 DNI > Total 的问题
    # ========================================
    # 策略：如果 DNI > GTI，用 GTI 作为 DNI 上限
    
    dni_gt_total = df[dni_col] > df[gti_col]
    n_dni_gt_total = dni_gt_total.sum()
    
    if n_dni_gt_total > 0:
        print(f"  校正 DNI > Total: {n_dni_gt_total} 条 ({n_dni_gt_total/len(df)*100:.2f}%)")
        
        # 记录原始 DNI 超过 GTI 的比例
        df.loc[dni_gt_total, 'dni_exceeded_gti_original'] = df.loc[dni_gt_total, dni_col]
        
        # 方案A：使用 DNI = GTI * 比例因子（保留相对变化）
        # 方案B：直接用 GTI 替代（保守方案）
        # 采用混合方案： DNI = min(DNI_original, GTI * 0.9)
        df.loc[dni_gt_total, dni_col] = np.minimum(
            df.loc[dni_gt_total, dni_col],
            df.loc[dni_gt_total, gti_col] * 0.9
        )
        
        # 标记校正
        df['dni_corrected'] = dni_gt_total.astype(np.int8)
    
    # ========================================
    # 2. 处理 DNI * cos(zenith) > GHI
    # ========================================
    if zenith_col in df.columns:
        zenith = df[zenith_col].clip(0, 89)  # 避免 cos(90°) = 0 的问题
        cos_zenith = np.cos(np.radians(zenith))
        
        dni_component = df[dni_col] * cos_zenith
        dni_exceeds_ghi = dni_component > df[ghi_col]
        n_dni_exceeds = dni_exceeds_ghi.sum()
        
        if n_dni_exceeds > 0:
            print(f"  校正 DNI*cos(zenith) > GHI: {n_dni_exceeds} 条 ({n_dni_exceeds/len(df)*100:.2f}%)")
            
            # 使用物理关系重建 DNI
            # GHI = DNI * cos(zenith) + DHI
            # 假设 DHI = GHI * 0.4 (典型散射比例)
            dhi_estimate = df[ghi_col] * 0.4
            corrected_dni = (df[ghi_col] - dhi_estimate) / (cos_zenith + 0.001)
            corrected_dni = corrected_dni.clip(lower=0)
            
            df.loc[dni_exceeds_ghi, dni_col] = corrected_dni.loc[dni_exceeds_ghi]
    
    # ========================================
    # 3. 处理负值
    # ========================================
    for col in [gti_col, dni_col, ghi_col]:
        negative_mask = df[col] < 0
        n_negative = negative_mask.sum()
        if n_negative > 0:
            print(f"  校正 {col} 负值: {n_negative} 条")
            df.loc[negative_mask, col] = 0
    
    # ========================================
    # 4. 处理超过理论最大值
    # ========================================
    if zenith_col in df.columns:
        # 晴天最大辐照度 ≈ 太阳常数 * 大气透射率 * cos(zenith)
        # 假设大气透射率约 0.7
        atmosphere_transmittance = 0.7
        zenith = df[zenith_col].clip(0, 89)
        cos_zenith = np.cos(np.radians(zenith))
        theoretical_max = SOLAR_CONSTANT * atmosphere_transmittance * cos_zenith
        
        over_max = df[gti_col] > theoretical_max * 1.05  # 5%容差
        n_over_max = over_max.sum()
        
        if n_over_max > 0:
            print(f"  校正 GTI > 理论最大值: {n_over_max} 条 ({n_over_max/len(df)*100:.2f}%)")
            df.loc[over_max, gti_col] = theoretical_max.loc[over_max] * 0.95
    
    # ========================================
    # 5. 重建一致性：确保 GHI ≈ GTI
    # ========================================
    # 对于晴天（云量低），GTI 应该接近 GHI
    # 计算比值并平滑异常
    
    if 'cloud_cover_pct' in df.columns:
        clear_sky_mask = df['cloud_cover_pct'] < 20
        
        # 晴天时，GTI 应该等于或略大于 GHI
        gti_lt_ghi = (df[gti_col] < df[ghi_col]) & clear_sky_mask
        n_gti_lt = gti_lt_ghi.sum()
        
        if n_gti_lt > 0:
            print(f"  校正 GTI < GHI (晴天): {n_gti_lt} 条")
            # 用 GHI 替代 GTI
            df.loc[gti_lt_ghi, gti_col] = df.loc[gti_lt_ghi, ghi_col]
    
    return df


# ============================================================================
# 辐照度合理性检查特征
# ============================================================================

def add_irradiance_validation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    添加辐照度物理合理性检查特征
    这些特征可用于：
    1. 模型训练时的样本权重
    2. 数据筛选依据
    3. 模型输入的附加信息
    """
    df = df.copy()
    
    gti_col = 'total_irradiance_wm2'
    dni_col = 'direct_normal_irradiance_wm2'
    ghi_col = 'global_horizontal_irradiance_wm2'
    zenith_col = 'solar_zenith_angle_deg'
    
    # ========================================
    # 1. 物理一致性指标
    # ========================================
    
    # DNI / GTI 比例（正常范围 0-1）
    if dni_col in df.columns and gti_col in df.columns:
        df['dni_gti_ratio'] = df[dni_col] / (df[gti_col] + 1)
        df['dni_gti_ratio'] = df['dni_gti_ratio'].clip(0, 2)
        
        # 合理性评分（0-1，越高越合理）
        # 正常情况下 DNI <= GTI，所以比例应该 <= 1
        df['irr_physical_score'] = 1 - (df['dni_gti_ratio'] - 1).clip(lower=0)
        df['irr_physical_score'] = df['irr_physical_score'].clip(0, 1)
    
    # ========================================
    # 2. DNI 分量合理性
    # ========================================
    if dni_col in df.columns and ghi_col in df.columns and zenith_col in df.columns:
        zenith = df[zenith_col].clip(0, 89)
        cos_zenith = np.cos(np.radians(zenith))
        
        # DNI 在水平面的投影
        dni_horizontal = df[dni_col] * cos_zenith
        
        # 投影应该 <= GHI
        df['dni_horizontal_component'] = dni_horizontal
        df['dni_exceeds_ghi_flag'] = (dni_horizontal > df[ghi_col]).astype(np.int8)
        
        # 直射贡献率
        df['direct_contribution_ratio'] = dni_horizontal / (df[ghi_col] + 1)
        df['direct_contribution_ratio'] = df['direct_contribution_ratio'].clip(0, 1.5)
    
    # ========================================
    # 3. 散射辐照度估算
    # ========================================
    if ghi_col in df.columns and dni_col in df.columns and zenith_col in df.columns:
        zenith = df[zenith_col].clip(0, 89)
        cos_zenith = np.cos(np.radians(zenith))
        
        # 估算散射辐照度: DHI = GHI - DNI * cos(zenith)
        estimated_dhi = df[ghi_col] - df[dni_col] * cos_zenith
        estimated_dhi = estimated_dhi.clip(lower=0)
        
        df['estimated_dhi'] = estimated_dhi
        
        # 散射比例
        df['diffuse_ratio'] = estimated_dhi / (df[ghi_col] + 1)
        df['diffuse_ratio'] = df['diffuse_ratio'].clip(0, 2)
        
        # 散射比例异常（散射不应超过总辐照的 95%）
        df['diffuse_anomaly'] = (df['diffuse_ratio'] > 0.95).astype(np.int8)
    
    # ========================================
    # 4. 综合辐照质量评分
    # ========================================
    quality_scores = []
    
    for idx in range(len(df)):
        score = 1.0
        
        # 检查 DNI > GTI
        if dni_col in df.columns and gti_col in df.columns:
            if df.iloc[idx][dni_col] > df.iloc[idx][gti_col]:
                score *= 0.5
        
        # 检查负值
        for col in [gti_col, dni_col, ghi_col]:
            if col in df.columns and df.iloc[idx][col] < 0:
                score *= 0.3
        
        # 检查极端比例
        if 'dni_gti_ratio' in df.columns:
            ratio = df.iloc[idx]['dni_gti_ratio']
            if ratio > 1.5:
                score *= 0.6
            elif ratio > 1.2:
                score *= 0.8
        
        quality_scores.append(score)
    
    df['irradiance_quality_score'] = quality_scores
    
    # 标记低质量数据
    df['irradiance_low_quality'] = (df['irradiance_quality_score'] < 0.7).astype(np.int8)
    
    return df


# ============================================================================
# 温度修正特征
# ============================================================================

def add_temperature_correction_features(df: pd.DataFrame, capacity_mw: float = 50.0) -> pd.DataFrame:
    """
    添加温度修正特征
    
    物理背景：
    - 太阳能电池温度升高会导致效率下降
    - 典型温度系数：每升高1°C，功率下降0.4-0.5%
    - 标准测试条件(STC)：25°C, 1000 W/m²
    """
    df = df.copy()
    
    if 'air_temperature_c' not in df.columns or 'total_irradiance_wm2' not in df.columns:
        print("警告: 缺少温度或辐照度列，跳过温度修正")
        return df
    
    temp_col = 'air_temperature_c'
    gti_col = 'total_irradiance_wm2'
    
    # ========================================
    # 1. 组件温度估算
    # ========================================
    # 使用 NOCT (Nominal Operating Cell Temperature) 模型
    # T_cell = T_air + (NOCT - 20) * GTI / 800 * 0.9
    NOCT = 45  # 标称工作电池温度 (°C)
    
    irradiance_factor = df[gti_col].fillna(0) / 800
    irradiance_factor = irradiance_factor.clip(0, 1.5)
    
    df['estimated_cell_temp'] = df[temp_col] + (NOCT - 20) * irradiance_factor * 0.9
    
    # ========================================
    # 2. 温度修正因子
    # ========================================
    # 温度系数：-0.004 / °C (即每升高1°C，功率下降0.4%)
    temp_coefficient = -0.004
    
    # 相对于25°C的温差
    df['temp_diff_from_stc'] = df['estimated_cell_temp'] - 25
    
    # 温度效率修正因子
    df['temp_efficiency_factor'] = 1 + temp_coefficient * df['temp_diff_from_stc']
    df['temp_efficiency_factor'] = df['temp_efficiency_factor'].clip(0.7, 1.1)
    
    # ========================================
    # 3. 温度-辐照度交互
    # ========================================
    # 高温 + 高辐照度 = 低效率
    # 低温 + 高辐照度 = 高效率
    
    df['temp_irradiance_product'] = df[temp_col] * df[gti_col] / 1000
    
    # 效率下降指数
    df['heat_stress_index'] = df['estimated_cell_temp'] * df[gti_col] / 10000
    df['heat_stress_index'] = df['heat_stress_index'].clip(0, 10)
    
    # ========================================
    # 4. 理论最大功率（温度修正后）
    # ========================================
    if gti_col in df.columns:
        # 基础理论功率
        df['theoretical_power_stc'] = df[gti_col] / 1000 * capacity_mw * 0.8
        
        # 温度修正后的理论功率
        df['theoretical_power_corrected'] = (
            df['theoretical_power_stc'] * df['temp_efficiency_factor']
        ).clip(upper=capacity_mw)
        
        # 温度效率比
        df['power_temp_efficiency_ratio'] = df['theoretical_power_corrected'] / (df['theoretical_power_stc'] + 0.1)
    
    # ========================================
    # 5. 高温预警标志
    # ========================================
    df['high_temp_warning'] = (df['estimated_cell_temp'] > 50).astype(np.int8)
    df['extreme_temp_warning'] = (df['estimated_cell_temp'] > 60).astype(np.int8)
    
    return df


# ============================================================================
# 功率-辐照度效率特征
# ============================================================================

def add_power_irradiance_efficiency_features(df: pd.DataFrame, capacity_mw: float = 50.0) -> pd.DataFrame:
    """
    添加功率-辐照度效率相关特征
    反映转换效率的变化
    """
    df = df.copy()
    
    gti_col = 'total_irradiance_wm2'
    power_col = 'power_mw'
    
    if gti_col not in df.columns or power_col not in df.columns:
        print("警告: 缺少功率或辐照度列")
        return df
    
    # ========================================
    # 1. 实时效率
    # ========================================
    # 效率 = 实际功率 / 理论功率 (STC)
    theoretical_power = df[gti_col] / 1000 * capacity_mw * 0.8
    
    df['instant_efficiency'] = df[power_col] / (theoretical_power + 0.1)
    df['instant_efficiency'] = df['instant_efficiency'].clip(0, 1.5)
    
    # ========================================
    # 2. 效率变化趋势
    # ========================================
    df['efficiency_rolling_mean'] = df['instant_efficiency'].rolling(4, min_periods=1).mean()
    df['efficiency_rolling_std'] = df['instant_efficiency'].rolling(4, min_periods=1).std()
    
    # 效率变化
    df['efficiency_change'] = df['instant_efficiency'] - df['efficiency_rolling_mean'].shift(1)
    df['efficiency_change'] = df['efficiency_change'].fillna(0)
    
    # ========================================
    # 3. 效率异常检测
    # ========================================
    # 低效率警告（可能由于脏污、遮挡、温度过高等）
    df['low_efficiency_flag'] = (df['instant_efficiency'] < 0.6).astype(np.int8)
    
    # 高效率警告（可能是数据错误）
    df['high_efficiency_flag'] = (df['instant_efficiency'] > 0.95).astype(np.int8)
    
    # ========================================
    # 4. 归一化功率（用于跨站点比较）
    # ========================================
    df['power_pu'] = df[power_col] / capacity_mw
    
    # 辐照度归一化
    df['irradiance_pu'] = df[gti_col] / 1000
    
    # 功率-辐照度比（理想情况下应该等于效率）
    df['power_irradiance_ratio'] = df['power_pu'] / (df['irradiance_pu'] + 0.01)
    df['power_irradiance_ratio'] = df['power_irradiance_ratio'].clip(0, 1.5)
    
    # ========================================
    # 5. 功率残差（模型可学习）
    # ========================================
    # 预测功率 = f(辐照度, 温度, 效率)
    if 'theoretical_power_corrected' in df.columns:
        df['power_residual'] = df[power_col] - df['theoretical_power_corrected']
    else:
        df['power_residual'] = df[power_col] - theoretical_power
    
    return df


# ============================================================================
# 时段交互特征
# ============================================================================

def add_time_period_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    添加时段交互特征，捕捉中午效应
    """
    df = df.copy()
    
    if 'hour' not in df.columns:
        print("警告: 缺少 hour 列")
        return df
    
    # ========================================
    # 1. 太阳高度时段
    # ========================================
    # 0: 夜间/日出前 (hour < 6)
    # 1: 早晨过渡 (6 <= hour < 9)  
    # 2: 上午稳定 (9 <= hour < 11)
    # 3: 中午峰值 (11 <= hour < 14)
    # 4: 下午稳定 (14 <= hour < 16)
    # 5: 傍晚过渡 (16 <= hour < 19)
    # 6: 日落后 (hour >= 19)
    
    def get_solar_period(hour):
        if hour < 6:
            return 0  # night
        elif hour < 9:
            return 1  # morning transition
        elif hour < 11:
            return 2  # morning stable
        elif hour < 14:
            return 3  # noon peak
        elif hour < 16:
            return 4  # afternoon stable
        elif hour < 19:
            return 5  # evening transition
        else:
            return 6  # night
    
    df['solar_period'] = df['hour'].apply(get_solar_period)
    
    # ========================================
    # 2. 时段交互特征
    # ========================================
    
    # 是否中午时段
    df['is_noon_period'] = (df['solar_period'] == 3).astype(np.int8)
    
    # 是否过渡时段
    df['is_transition'] = df['solar_period'].isin([1, 5]).astype(np.int8)
    
    # 时段编码（周期性）
    hour_radians = np.radians(df['hour'] * 15)  # 15度/小时
    df['hour_sin'] = np.sin(hour_radians)
    df['hour_cos'] = np.cos(hour_radians)
    
    # ========================================
    # 3. 中午效应修正
    # ========================================
    # 中午时段：太阳高度角最高，但温度也最高
    # 添加中午-温度交互
    if 'air_temperature_c' in df.columns:
        df['noon_temp_interaction'] = df['is_noon_period'] * df['air_temperature_c']
        df['noon_temp_squared'] = df['is_noon_period'] * df['air_temperature_c'] ** 2
    
    # 中午-辐照度交互
    if 'total_irradiance_wm2' in df.columns:
        df['noon_irradiance_interaction'] = df['is_noon_period'] * df['total_irradiance_wm2']
    
    # 中午-效率交互
    if 'instant_efficiency' in df.columns:
        df['noon_efficiency_interaction'] = df['is_noon_period'] * df['instant_efficiency']
    
    # ========================================
    # 4. 太阳高度角（如果未计算）
    # ========================================
    if 'solar_elevation_angle_deg' not in df.columns and 'solar_zenith_angle_deg' in df.columns:
        df['solar_elevation_angle_deg'] = 90 - df['solar_zenith_angle_deg']
    
    # 太阳高度角时段
    if 'solar_elevation_angle_deg' in df.columns:
        df['high_sun_period'] = (df['solar_elevation_angle_deg'] > 50).astype(np.int8)
    
    return df


# ============================================================================
# 综合特征构建
# ============================================================================

def build_physics_aware_features(
    df: pd.DataFrame,
    capacity_mw: float = 50.0,
    apply_correction: bool = True
) -> pd.DataFrame:
    """
    构建完整的物理感知特征集
    
    参数：
    - df: 原始数据
    - capacity_mw: 装机容量 (MW)
    - apply_correction: 是否应用物理一致性校正
    """
    print("=" * 80)
    print("Site1 物理感知特征构建")
    print("=" * 80)
    
    print(f"\n原始数据: {len(df)} 行, {len(df.columns)} 列")
    
    # 1. 物理一致性校正
    if apply_correction:
        print("\n[1/5] 应用辐照度物理一致性校正...")
        df = correct_irradiance_physical_consistency(df, capacity_mw)
    else:
        print("\n[1/5] 跳过物理一致性校正（保留原始数据）")
    
    # 2. 辐照度合理性特征
    print("\n[2/5] 添加辐照度合理性检查特征...")
    df = add_irradiance_validation_features(df)
    
    # 3. 温度修正特征
    print("\n[3/5] 添加温度修正特征...")
    df = add_temperature_correction_features(df, capacity_mw)
    
    # 4. 效率特征
    print("\n[4/5] 添加功率-辐照度效率特征...")
    df = add_power_irradiance_efficiency_features(df, capacity_mw)
    
    # 5. 时段交互特征
    print("\n[5/5] 添加时段交互特征...")
    df = add_time_period_features(df)
    
    print(f"\n最终数据: {len(df)} 行, {len(df.columns)} 列")
    print(f"新增特征数: {len(df.columns) - 20}")  # 估计原始列数约20
    
    return df


def get_all_new_feature_names() -> list:
    """返回所有新增特征的名称"""
    return [
        # 物理校正记录
        'dni_exceeded_gti_original',
        'dni_corrected',
        
        # 辐照度合理性特征
        'dni_gti_ratio',
        'irr_physical_score',
        'dni_horizontal_component',
        'dni_exceeds_ghi_flag',
        'direct_contribution_ratio',
        'estimated_dhi',
        'diffuse_ratio',
        'diffuse_anomaly',
        'irradiance_quality_score',
        'irradiance_low_quality',
        
        # 温度修正特征
        'estimated_cell_temp',
        'temp_diff_from_stc',
        'temp_efficiency_factor',
        'temp_irradiance_product',
        'heat_stress_index',
        'theoretical_power_stc',
        'theoretical_power_corrected',
        'power_temp_efficiency_ratio',
        'high_temp_warning',
        'extreme_temp_warning',
        
        # 效率特征
        'instant_efficiency',
        'efficiency_rolling_mean',
        'efficiency_rolling_std',
        'efficiency_change',
        'low_efficiency_flag',
        'high_efficiency_flag',
        'power_pu',
        'irradiance_pu',
        'power_irradiance_ratio',
        'power_residual',
        
        # 时段特征
        'solar_period',
        'is_noon_period',
        'is_transition',
        'hour_sin',
        'hour_cos',
        'noon_temp_interaction',
        'noon_temp_squared',
        'noon_irradiance_interaction',
        'noon_efficiency_interaction',
        'high_sun_period',
        'solar_elevation_angle_deg',  # 如果新增的
    ]


def get_feature_groups() -> dict:
    """返回按类别分组的特征"""
    return {
        'irradiance_validation': [
            'dni_gti_ratio', 'irr_physical_score', 'dni_exceeds_ghi_flag',
            'direct_contribution_ratio', 'diffuse_ratio', 'diffuse_anomaly',
            'irradiance_quality_score', 'irradiance_low_quality'
        ],
        'temperature_correction': [
            'estimated_cell_temp', 'temp_diff_from_stc', 'temp_efficiency_factor',
            'heat_stress_index', 'theoretical_power_corrected', 'high_temp_warning'
        ],
        'power_efficiency': [
            'instant_efficiency', 'efficiency_rolling_mean', 'efficiency_change',
            'low_efficiency_flag', 'power_irradiance_ratio', 'power_residual'
        ],
        'time_period_interaction': [
            'solar_period', 'is_noon_period', 'noon_temp_interaction',
            'noon_irradiance_interaction', 'noon_efficiency_interaction'
        ]
    }


# ============================================================================
# 诊断和报告
# ============================================================================

def diagnose_after_correction(df_before: pd.DataFrame, df_after: pd.DataFrame) -> dict:
    """对比校正前后的数据质量"""
    
    report = {
        'corrections_applied': {},
        'improvements': {}
    }
    
    gti_col = 'total_irradiance_wm2'
    dni_col = 'direct_normal_irradiance_wm2'
    
    # DNI > GTI 问题
    if dni_col in df_before.columns and gti_col in df_before.columns:
        before_count = (df_before[dni_col] > df_before[gti_col]).sum()
        after_count = (df_after[dni_col] > df_after[gti_col]).sum()
        report['corrections_applied']['dni_gt_total'] = {
            'before': int(before_count),
            'after': int(after_count),
            'fixed': int(before_count - after_count)
        }
    
    # 负值问题
    for col in [gti_col, dni_col]:
        if col in df_before.columns:
            before_neg = (df_before[col] < 0).sum()
            after_neg = (df_after[col] < 0).sum()
            report['corrections_applied'][f'{col}_negative'] = {
                'before': int(before_neg),
                'after': int(after_neg),
                'fixed': int(before_neg - after_neg)
            }
    
    return report


def print_diagnostic_summary(df: pd.DataFrame):
    """打印诊断摘要"""
    
    print("\n" + "=" * 80)
    print("诊断摘要")
    print("=" * 80)
    
    # 辐照度质量分布
    if 'irradiance_quality_score' in df.columns:
        print("\n辐照度质量评分分布:")
        print(f"  优秀 (>=0.9): {(df['irradiance_quality_score'] >= 0.9).sum():,}")
        print(f"  良好 (0.7-0.9): {((df['irradiance_quality_score'] >= 0.7) & (df['irradiance_quality_score'] < 0.9)).sum():,}")
        print(f"  较差 (<0.7): {(df['irradiance_quality_score'] < 0.7).sum():,}")
    
    # 温度警告分布
    if 'high_temp_warning' in df.columns:
        print(f"\n高温警告 (组件温度>50°C): {df['high_temp_warning'].sum():,} 条")
    
    # 低效率分布
    if 'low_efficiency_flag' in df.columns:
        print(f"低效率警告 (效率<60%): {df['low_efficiency_flag'].sum():,} 条")
    
    # DNI 合理性
    if 'dni_gti_ratio' in df.columns:
        print(f"\nDNI/GTI 比例统计:")
        print(f"  均值: {df['dni_gti_ratio'].mean():.3f}")
        print(f"  中位数: {df['dni_gti_ratio'].median():.3f}")
        print(f"  合理 (<=1): {(df['dni_gti_ratio'] <= 1).sum():,}")
        print(f"  不合理 (>1): {(df['dni_gti_ratio'] > 1).sum():,}")
    
    # 中午效应
    if 'instant_efficiency' in df.columns and 'is_noon_period' in df.columns:
        noon_eff = df[df['is_noon_period'] == 1]['instant_efficiency'].mean()
        non_noon_eff = df[df['is_noon_period'] == 0]['instant_efficiency'].mean()
        print(f"\n中午效率 vs 非中午效率: {noon_eff:.3f} vs {non_noon_eff:.3f}")
        if noon_eff < non_noon_eff:
            print(f"  效率下降: {(1 - noon_eff/non_noon_eff)*100:.1f}%")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """运行物理感知特征构建"""
    
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    DATA_PATH = PROJECT_ROOT / "data" / "prediction" / "step1_preprocessing" / "processed" / "stations" / "Site_1_optimized.csv"
    OUTPUT_PATH = DATA_PATH
    CAPACITY_MW = 50.0
    
    print("=" * 80)
    print("Site1 辐照度物理一致性校正与增强特征")
    print("=" * 80)
    print(f"时间: {pd.Timestamp.now()}")
    print()
    
    # 读取原始数据
    print("读取数据...")
    df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'] if 'timestamp' in pd.read_csv(DATA_PATH, nrows=0).columns else None)
    print(f"数据量: {len(df):,} 条")
    print(f"时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
    
    # 保存校正前副本
    df_before = df.copy()
    
    # 应用物理感知特征
    df_after = build_physics_aware_features(df, capacity_mw=CAPACITY_MW, apply_correction=True)
    
    # 诊断对比
    report = diagnose_after_correction(df_before, df_after)
    
    print("\n" + "=" * 80)
    print("校正统计")
    print("=" * 80)
    for key, stats in report['corrections_applied'].items():
        print(f"\n{key}:")
        print(f"  校正前: {stats['before']:,}")
        print(f"  校正后: {stats['after']:,}")
        print(f"  已修复: {stats['fixed']:,}")
    
    # 打印诊断摘要
    print_diagnostic_summary(df_after)
    
    # 保存
    print("\n" + "=" * 80)
    print("保存结果")
    print("=" * 80)
    
    df_after.to_csv(OUTPUT_PATH, index=False)
    print(f"已保存到: {OUTPUT_PATH}")
    
    # 列出新增特征
    print("\n新增特征列表:")
    all_features = get_all_new_feature_names()
    for group_name, features in get_feature_groups().items():
        print(f"\n  [{group_name}]:")
        for f in features:
            if f in df_after.columns:
                print(f"    - {f}")
    
    print("\n" + "=" * 80)
    print("完成!")
    print("=" * 80)
    
    return df_after, report


if __name__ == "__main__":
    main()
