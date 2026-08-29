"""EXP-P04 特征工程：lag、ramp、rolling 统计特征，无未来泄漏。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Feature engineering on raw DataFrame (no future leakage)
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    给原始 DataFrame 添加所有特征，返回新增列列表。
    假设 df 已按时间升序排列，index 对应时间顺序。
    禁止使用任何未来信息。
    """
    df = df.copy()

    # 基础气象/辐照特征（已在 df 中）
    base_features = [
        "total_irradiance_wm2", "direct_normal_irradiance_wm2",
        "global_horizontal_irradiance_wm2", "air_temperature_c",
        "atmosphere_hpa", "relative_humidity_pct", "daylight_flag",
    ]

    # 时间周期特征
    ts = df.index.to_series() if df.index.name else pd.Series(df.index, name="ts")
    hour = ts.dt.hour
    doy = ts.dt.dayofyear

    df["sin_hour"] = np.sin(2 * np.pi * hour / 24)
    df["cos_hour"] = np.cos(2 * np.pi * hour / 24)
    df["sin_dayofyear"] = np.sin(2 * np.pi * doy / 365)
    df["cos_dayofyear"] = np.cos(2 * np.pi * doy / 365)

    # data_quality_score
    if "data_quality_score" not in df.columns:
        df["data_quality_score"] = 1.0

    # ---- 功率 lag 特征 ----
    for lag in [0, 1, 2, 4, 8, 16]:
        col = f"power_pu_lag_{lag}"
        df[col] = df["power_pu"].shift(lag)
        # lag=0 本身无 shift，但保持列名一致性

    # ---- 多尺度 ramp 特征 ----
    df["power_ramp_15m_pu"] = df["power_pu"] - df["power_pu"].shift(1)
    df["power_ramp_60m_pu"] = df["power_pu"] - df["power_pu"].shift(4)
    df["power_ramp_120m_pu"] = df["power_pu"] - df["power_pu"].shift(8)

    # ---- 滚动统计特征（只用过去数据，min_periods=1 避免开头发 NaN） ----
    df["power_pu_roll_1h_mean"] = df["power_pu"].shift(1).rolling(4, min_periods=1).mean()
    df["power_pu_roll_1h_std"] = df["power_pu"].shift(1).rolling(4, min_periods=2).std().fillna(0.0)

    df["power_pu_roll_2h_mean"] = df["power_pu"].shift(1).rolling(8, min_periods=1).mean()
    df["power_pu_roll_2h_std"] = df["power_pu"].shift(1).rolling(8, min_periods=2).std().fillna(0.0)
    df["power_pu_roll_2h_max"] = df["power_pu"].shift(1).rolling(8, min_periods=1).max()
    df["power_pu_roll_2h_min"] = df["power_pu"].shift(1).rolling(8, min_periods=1).min()

    return df


FEATURE_COLUMNS = [
    # 气象/辐照（7）
    "total_irradiance_wm2", "direct_normal_irradiance_wm2",
    "global_horizontal_irradiance_wm2", "air_temperature_c",
    "atmosphere_hpa", "relative_humidity_pct", "daylight_flag",
    # 时间周期（4）
    "sin_hour", "cos_hour", "sin_dayofyear", "cos_dayofyear",
    # data quality（1）
    "data_quality_score",
    # power lag（6）
    "power_pu_lag_0", "power_pu_lag_1", "power_pu_lag_2",
    "power_pu_lag_4", "power_pu_lag_8", "power_pu_lag_16",
    # 多尺度 ramp（3）
    "power_ramp_15m_pu", "power_ramp_60m_pu", "power_ramp_120m_pu",
    # rolling 统计（7）
    "power_pu_roll_1h_mean", "power_pu_roll_1h_std",
    "power_pu_roll_2h_mean", "power_pu_roll_2h_std",
    "power_pu_roll_2h_max", "power_pu_roll_2h_min",
]


def get_feature_columns() -> list[str]:
    return FEATURE_COLUMNS.copy()


# ---------------------------------------------------------------------------
# Sequence construction (lookback / horizon)
# ---------------------------------------------------------------------------

def create_sequences(
    X_df: pd.DataFrame,
    y_series: pd.Series,
    feature_cols: list[str],
    lookback: int = 16,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    按时间顺序构造 (N, lookback, n_features) 和 (N, horizon) 样本。
    前 lookback 步作为输入，horizon 步作为标签。
    只取有效区间（前 lookback+horizon-1 行因 lag/rolling 会产生 NaN，自动跳过）。
    """
    df_feat = X_df[feature_cols].values
    y_vals = y_series.values

    n = len(df_feat)
    X_list, y_list = [], []

    for i in range(lookback, n - horizon + 1):
        x_seq = df_feat[i - lookback : i]          # [lookback, n_features]
        y_seq = y_vals[i : i + horizon]              # [horizon]

        # 跳过含 NaN 的行（仅在数据边界发生）
        if np.isnan(x_seq).any() or np.isnan(y_seq).any():
            continue

        X_list.append(x_seq)
        y_list.append(y_seq)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


# ---------------------------------------------------------------------------
# Scaler helpers
# ---------------------------------------------------------------------------

def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    """沿 axis=0 (时间维) 标准化，只用训练集。"""
    n_samples, seq_len, n_features = X_train.shape
    X_flat = X_train.reshape(-1, n_features)
    scaler = StandardScaler()
    scaler.fit(X_flat)
    return scaler


def transform_X(scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    n_samples, seq_len, n_features = X.shape
    X_flat = X.reshape(-1, n_features)
    X_scaled = scaler.transform(X_flat)
    return X_scaled.reshape(n_samples, seq_len, n_features)


def fit_y_scaler(y_train: np.ndarray) -> StandardScaler:
    """沿 axis=0 标准化标签。"""
    scaler = StandardScaler()
    scaler.fit(y_train)
    return scaler


def transform_y(scaler: StandardScaler, y: np.ndarray) -> np.ndarray:
    return scaler.transform(y)


def inverse_transform_y(scaler: StandardScaler, y: np.ndarray) -> np.ndarray:
    return scaler.inverse_transform(y)
