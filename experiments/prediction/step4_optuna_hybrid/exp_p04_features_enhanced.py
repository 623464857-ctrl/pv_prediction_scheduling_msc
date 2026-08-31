"""
Site1 特征工程增强模块 - EXP-P04-v2
==================================
为优化后的Site1数据添加额外的质量相关特征

运行：
python experiments/prediction/step4_optuna_hybrid/exp_p04_features_enhanced.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================================
# 特征工程增强函数
# ============================================================================

def build_enhanced_features(df: pd.DataFrame, capacity_mw: float = 50.0) -> pd.DataFrame:
    """
    为优化后的数据添加增强特征
    
    新增特征类别：
    1. 辐照质量特征
    2. 功率一致性特征  
    3. 数据质量权重
    4. 气象一致性特征
    """
    df = df.copy()
    
    # ========================================
    # 1. 辐照质量特征
    # ========================================
    
    # 辐照变化率（检测云层快速变化）
    if 'total_irradiance_wm2' in df.columns:
        df['gti_change_rate'] = df['total_irradiance_wm2'].diff().abs()
        
        # 辐照平滑度
        roll_mean = df['total_irradiance_wm2'].rolling(4, min_periods=1).mean()
        roll_std = df['total_irradiance_wm2'].rolling(4, min_periods=1).std()
        df['gti_smoothness'] = roll_std / (roll_mean + 1)
        
        # DNI/DHI比例
        if 'direct_normal_irradiance_wm2' in df.columns and 'global_horizontal_irradiance_wm2' in df.columns:
            dni = df['direct_normal_irradiance_wm2']
            ghi = df['global_horizontal_irradiance_wm2']
            df['dni_dhi_ratio'] = dni / (ghi + 1)
            
            # 比例异常检测
            df['dni_dhi_anomaly'] = ((df['dni_dhi_ratio'] > 3) | (df['dni_dhi_ratio'] < 0.1)).astype(np.int8)
    
    # ========================================
    # 2. 功率一致性特征
    # ========================================
    
    if 'total_irradiance_wm2' in df.columns and 'power_mw' in df.columns:
        # 理论功率（基于辐照估算）
        # 效率约0.8（考虑温度、灰尘等损耗）
        df['theoretical_power'] = df['total_irradiance_wm2'] / 1000 * capacity_mw * 0.8
        df['theoretical_power'] = df['theoretical_power'].clip(upper=capacity_mw)
        
        # 功率效率
        df['power_efficiency'] = df['power_mw'] / (df['theoretical_power'] + 0.1)
        df['power_efficiency'] = df['power_efficiency'].clip(lower=0, upper=1.5)
        
        # 功率-辐照一致性指标
        df['power_irradiance_consistency'] = 1 - abs(df['power_efficiency'] - 0.8) / 0.8
        df['power_irradiance_consistency'] = df['power_irradiance_consistency'].clip(lower=0, upper=1)
    
    # ========================================
    # 3. 数据质量权重（用于训练加权）
    # ========================================
    
    if 'overall_quality_score' in df.columns:
        df['sample_weight'] = df['overall_quality_score']
    
    # 训练时排除低质量样本
    if 'overall_quality_score' in df.columns:
        df['use_for_training'] = (df['overall_quality_score'] >= 0.5).astype(np.int8)
    
    # ========================================
    # 4. 气象一致性特征
    # ========================================
    
    if 'air_temperature_c' in df.columns and 'relative_humidity_pct' in df.columns:
        # 露点温度简化估算（Magon's formula简化版）
        temp = df['air_temperature_c']
        rh = df['relative_humidity_pct'].clip(lower=1, upper=99)
        df['dew_point_approx'] = temp - (100 - rh) / 5
        
        # 露点差（干湿球温差相关）
        df['dew_point_spread'] = temp - df['dew_point_approx']
        
        # 热不适指数
        df['heat_discomfort_index'] = temp + 0.5 * (rh - 40)
        
        # 温度-辐照交互
        if 'total_irradiance_wm2' in df.columns:
            df['temp_irradiance_interaction'] = temp * df['total_irradiance_wm2'] / 1000
    
    # ========================================
    # 5. 时间特征增强
    # ========================================
    
    if 'hour' in df.columns and 'month' in df.columns:
        # 是否是峰值时段
        df['is_peak_hour'] = df['hour'].between(11, 14).astype(np.int8)
        
        # 是否是早晚时段
        df['is_transition_hour'] = (df['hour'].between(6, 9) | df['hour'].between(15, 19)).astype(np.int8)
        
        # 季度
        df['quarter'] = ((df['month'] - 1) // 3) + 1
        
        # 是否是夏季峰值季节
        df['is_summer'] = df['month'].isin([6, 7, 8]).astype(np.int8)
        
        # 是否是冬季
        df['is_winter'] = df['month'].isin([12, 1, 2]).astype(np.int8)
    
    # ========================================
    # 6. 滞后特征增强
    # ========================================
    
    if 'power_mw' in df.columns:
        # 功率变化方向
        df['power_trend'] = np.sign(df['power_mw'].diff())
        
        # 连续上升/下降计数
        df['power_streak'] = 0
        current_streak = 0
        for i in range(1, len(df)):
            if df.iloc[i]['power_trend'] == df.iloc[i-1]['power_trend']:
                current_streak += 1
            else:
                current_streak = 0
            df.iloc[i, df.columns.get_loc('power_streak')] = current_streak
    
    return df


def get_enhanced_feature_columns() -> list[str]:
    """
    返回增强特征列名列表
    """
    return [
        # 辐照质量
        'gti_change_rate',
        'gti_smoothness', 
        'dni_dhi_ratio',
        'dni_dhi_anomaly',
        
        # 功率一致性
        'theoretical_power',
        'power_efficiency',
        'power_irradiance_consistency',
        
        # 质量权重
        'sample_weight',
        'use_for_training',
        
        # 气象一致性
        'dew_point_approx',
        'dew_point_spread',
        'heat_discomfort_index',
        'temp_irradiance_interaction',
        
        # 时间增强
        'is_peak_hour',
        'is_transition_hour',
        'quarter',
        'is_summer',
        'is_winter',
        
        # 滞后特征
        'power_trend',
        'power_streak',
    ]


def filter_high_quality_samples(df: pd.DataFrame, 
                               min_quality: float = 0.5,
                               exclude_anomaly_levels: list = None) -> pd.DataFrame:
    """
    过滤高质量样本
    
    参数：
    - min_quality: 最低质量分数
    - exclude_anomaly_levels: 要排除的异常级别列表
    """
    if exclude_anomaly_levels is None:
        exclude_anomaly_levels = [2, 3]  # 默认排除严重和明显异常
    
    mask = df['overall_quality_score'] >= min_quality
    
    if 'irradiance_power_anomaly_level' in df.columns:
        mask = mask & (~df['irradiance_power_anomaly_level'].isin(exclude_anomaly_levels))
    
    if 'in_outage' in df.columns:
        mask = mask & (df['in_outage'] == 0)
    
    return df[mask].copy()


# ============================================================================
# 主函数
# ============================================================================
def main():
    """
    运行特征增强
    """
    from pathlib import Path
    
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    DATA_PATH = PROJECT_ROOT / "data" / "prediction" / "step1_preprocessing" / "processed" / "stations" / "Site_1_optimized.csv"
    OUTPUT_PATH = DATA_PATH
    
    print("=" * 80)
    print("Site1 特征工程增强")
    print("=" * 80)
    
    # 读取优化后的数据
    df = pd.read_csv(DATA_PATH)
    print(f"读取数据: {len(df)} 行")
    
    # 应用特征增强
    df_enhanced = build_enhanced_features(df)
    print(f"增强后列数: {len(df.columns)} -> {len(df_enhanced.columns)}")
    
    # 统计新增特征
    new_features = get_enhanced_feature_columns()
    print(f"\n新增 {len(new_features)} 个特征:")
    for f in new_features:
        if f in df_enhanced.columns:
            print(f"  - {f}")
    
    # 统计高质量样本
    df_filtered = filter_high_quality_samples(df_enhanced)
    print(f"\n高质量样本: {len(df_filtered)} / {len(df_enhanced)} ({len(df_filtered)/len(df_enhanced)*100:.1f}%)")
    
    # 保存
    df_enhanced.to_csv(OUTPUT_PATH, index=False)
    print(f"\n已保存到: {OUTPUT_PATH}")
    
    print("=" * 80)
    print("特征工程增强完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
