"""
EXP-P08 物理约束特征工程 (精简版)

基于光伏物理原理添加约束特征：
1. 温度校正功率
2. 综合物理上限 (辐照度 × 温度校正)
3. 晴空指数

物理公式：
- P_T_corr = 1 - β × (T - T_stc)  where β ≈ 0.004/°C
- Clearness Index = GHI / ExtraTerrestrialRadiation
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 物理常数
G_REF = 1000.0  # 参考辐照度 (W/m2), STC条件
T_STC = 25.0    # STC温度 (°C)
BETA = 0.004    # 温度系数 (°C^-1), 晶硅组件典型值

# 额外天顶辐照度估算 (W/m2) - 与纬度相关，假设中纬度
EXTRA_TERRESTRIAL_RADIANCE = 1361.0  # 太阳常数


def build_physics_features(df: pd.DataFrame, capacity_mw: float = 50.0) -> pd.DataFrame:
    """
    添加物理约束特征到 DataFrame (精简版)。

    参数:
        df: 原始数据，必须包含列:
            - total_irradiance_wm2 或 total_irradiance (W/m2)
            - air_temperature_c 或 temperature
        capacity_mw: 光伏电站额定容量 (MW)

    返回:
        添加物理特征的 DataFrame
    """
    df = df.copy()

    # 确保列名一致
    irradiance_col = "total_irradiance_wm2" if "total_irradiance_wm2" in df.columns else "total_irradiance"
    temp_col = "air_temperature_c" if "air_temperature_c" in df.columns else "air_temperature_c"

    irradiance = df[irradiance_col].values
    temperature = df[temp_col].values

    # ========== 1. 温度校正因子 ==========
    # P_T = 1 - β × (T - T_stc)
    temp_correction = 1 - BETA * (temperature - T_STC)
    temp_correction = np.clip(temp_correction, 0.5, 1.2)  # 限制范围
    df["temp_correction_factor"] = temp_correction

    # ========== 2. 综合物理上限 (辐照度 × 温度校正) ==========
    irradiance_norm = np.clip(irradiance / G_REF, 0, 1.2)
    df["power_physics_bound_pu"] = np.clip(
        irradiance_norm * temp_correction,
        0,
        1.2
    )

    # ========== 3. 晴空指数 (Clearness Index) ==========
    # 基于时间估算最大可能辐照度
    ts = df.index.to_series() if df.index.name else pd.Series(df.index, name="ts")
    hour = ts.dt.hour + ts.dt.minute / 60.0
    doy = ts.dt.dayofyear

    # 简化的日变化周期
    day_factor = np.sin(2 * np.pi * (hour - 6) / 24)
    day_factor = np.clip(day_factor, 0, 1)

    # 季节因子 (夏至最大)
    season_factor = np.cos(2 * np.pi * (doy - 172) / 365)
    season_factor = np.clip(season_factor * 0.3 + 0.7, 0.3, 1.0)

    # 估算的地外辐照度分量
    extraterrestrial_estimate = EXTRA_TERRESTRIAL_RADIANCE * day_factor * season_factor

    # 晴空指数
    with np.errstate(divide='ignore', invalid='ignore'):
        clearness_index = irradiance / (extraterrestrial_estimate + 1e-6)
        clearness_index = np.clip(clearness_index, 0, 1.5)

    df["clearness_index"] = clearness_index

    # ========== 4. 晴空指数分类 ==========
    # 0-0.2: 阴天/夜间, 0.2-0.5: 多云, 0.5-0.8: 晴间多云, 0.8-1.0: 晴空
    df["clearness_category"] = pd.cut(
        clearness_index,
        bins=[-np.inf, 0.2, 0.5, 0.8, np.inf],
        labels=[0, 1, 2, 3]
    ).astype(float).fillna(0)

    # ========== 5. 辐照度变化率特征 (仅保留长时间尺度) ==========
    if irradiance_col in df.columns:
        df["irradiance_ramp_60m"] = df[irradiance_col].diff(4).fillna(0)

    return df


# 精简后的物理特征列表 (6个)
PHYSICS_FEATURE_COLUMNS = [
    # 温度校正 (1)
    "temp_correction_factor",
    # 综合物理上限 (1)
    "power_physics_bound_pu",
    # 晴空指数 (1)
    "clearness_index",
    # 晴空分类 (1)
    "clearness_category",
    # 辐照度变化率 (1)
    "irradiance_ramp_60m",
]


def get_physics_feature_columns() -> list[str]:
    """返回物理特征列名列表。"""
    return PHYSICS_FEATURE_COLUMNS.copy()


def build_combined_features(df: pd.DataFrame, capacity_mw: float = 50.0) -> pd.DataFrame:
    """
    构建完整特征集：原始特征 + 物理特征。
    复用 step4 的 build_features，然后添加物理特征。
    """
    from experiments.prediction.step4_optuna_hybrid.exp_p04_features import build_features

    # 先构建基础特征
    df_feat = build_features(df)

    # 添加物理特征
    df_feat = build_physics_features(df_feat, capacity_mw)

    return df_feat


def get_all_feature_columns() -> list[str]:
    """
    返回完整特征列：基础特征 + 物理特征。
    """
    from experiments.prediction.step4_optuna_hybrid.exp_p04_features import FEATURE_COLUMNS

    return FEATURE_COLUMNS + PHYSICS_FEATURE_COLUMNS
