"""EXP-P05 特征工程：lag / rolling / ramp / daylight / 周期特征。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.prediction.step4_optuna_hybrid.exp_p04_features import (
    FEATURE_COLUMNS,
    build_features,
    create_sequences,
    fit_scaler,
    fit_y_scaler,
    get_feature_columns,
    inverse_transform_y,
    transform_X,
    transform_y,
)

FEATURE_VERSION = "p05_v1"

# step5 在 step4 特征基础上统一使用完整特征列
P05_FEATURE_COLUMNS = FEATURE_COLUMNS.copy()


def build_p05_features(df: pd.DataFrame) -> pd.DataFrame:
    """构建 EXP-P05 增强特征集（复用 EXP-P04 实现，固定版本号）。"""
    out = build_features(df)
    out.attrs["feature_version"] = FEATURE_VERSION
    return out


def get_p05_feature_columns() -> list[str]:
    return P05_FEATURE_COLUMNS.copy()


def build_windows_from_df(
    df: pd.DataFrame,
    feature_cols: list[str],
    lookback: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    从带特征的 DataFrame 构造窗口样本。
    返回: X_seq, y, y_last, daylight_flag, timestamps, valid_indices
    """
    feat = df[feature_cols].to_numpy(dtype=np.float64)
    target = df["power_pu"].to_numpy(dtype=np.float64)
    daylight = df["daylight_flag"].to_numpy(dtype=np.float64) if "daylight_flag" in df.columns else np.zeros(len(df))
    timestamps = pd.to_datetime(df["timestamp"]).to_numpy() if "timestamp" in df.columns else np.arange(len(df))

    n = len(target)
    n_samples = n - lookback - horizon + 1
    if n_samples <= 0:
        raise ValueError("时序长度不足，无法构造窗口样本")

    X_seq = np.zeros((n_samples, lookback, len(feature_cols)), dtype=np.float32)
    y = np.zeros((n_samples, horizon), dtype=np.float32)
    y_last = np.zeros((n_samples, 1), dtype=np.float32)
    day_flag = np.zeros((n_samples, 1), dtype=np.float32)
    ts_out = np.empty(n_samples, dtype="datetime64[ns]")
    valid = np.ones(n_samples, dtype=bool)

    for i in range(n_samples):
        x_win = feat[i : i + lookback]
        y_win = target[i + lookback : i + lookback + horizon]
        last_idx = i + lookback - 1
        if np.isnan(x_win).any() or np.isnan(y_win).any():
            valid[i] = False
        X_seq[i] = x_win
        y[i] = y_win
        y_last[i, 0] = target[last_idx]
        day_flag[i, 0] = daylight[i + lookback]
        ts_out[i] = timestamps[i + lookback]

    mask = valid
    return X_seq[mask], y[mask], y_last[mask], day_flag[mask], ts_out[mask], np.where(mask)[0]
