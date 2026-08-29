"""EXP-P04 特征工程：lag、ramp、rolling 统计特征，无未来泄漏。"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from experiments.prediction.step4_optuna_hybrid.exp_p04_common import (
    PROJECT_ROOT,
    STEP4_ROOT,
)

FEATURE_SELECTION_DIR = STEP4_ROOT / "feature_selection"

DEFAULT_FORCE_KEEP = [
    "power_pu_lag_0",
    "sin_hour",
    "cos_hour",
    "sin_dayofyear",
    "cos_dayofyear",
    "daylight_flag",
]


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


# ---------------------------------------------------------------------------
# Feature selection by Pearson correlation (train set only)
# ---------------------------------------------------------------------------

def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """计算 Pearson 相关系数；常数列或无效样本返回 0。"""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _feature_decision(
    name: str,
    r: float,
    threshold: float,
    force_keep: set[str],
) -> tuple[str, str]:
    if name in force_keep:
        return "keep", "强制保留"
    if r > threshold:
        return "keep", "保留"
    if r < -threshold:
        return "remove", "剔除 (负相关)"
    return "remove", f"剔除 (|r| <= {threshold})"

def filter_features_by_correlation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    correlation_threshold: float = 0.05,
    force_keep: list[str] | None = None,
) -> tuple[list[str], dict[str, float], dict[str, str]]:
    """
    基于训练集 Pearson 相关系数自动筛选特征。

    Args:
        X_train: (n_samples, n_features) 或 (n_samples, seq_len, n_features)
        y_train: (n_samples,) 或 (n_samples, horizon)
        feature_names: 特征名称列表
        correlation_threshold: 保留阈值，仅 r > threshold 或 force_keep 保留
        force_keep: 强制保留的特征名

    Returns:
        selected_features, correlation_dict, decision_dict
    """
    if force_keep is None:
        force_keep = DEFAULT_FORCE_KEEP.copy()
    force_set = set(force_keep)

    if X_train.ndim == 3:
        X_2d = X_train[:, -1, :]
    else:
        X_2d = X_train

    y_vec = np.asarray(y_train)
    if y_vec.ndim > 1 and y_vec.shape[1] > 1:
        y_vec = y_vec[:, 0]
    y_vec = y_vec.ravel()

    if X_2d.shape[1] != len(feature_names):
        raise ValueError(
            f"特征数不匹配: X 有 {X_2d.shape[1]} 列, feature_names 有 {len(feature_names)} 项"
        )

    correlation_dict: dict[str, float] = {}
    decision_dict: dict[str, str] = {}
    selected: list[str] = []

    for j, name in enumerate(feature_names):
        r = _pearson_r(X_2d[:, j], y_vec)
        correlation_dict[name] = round(r, 6)
        action, decision = _feature_decision(name, r, correlation_threshold, force_set)
        decision_dict[name] = decision
        if action == "keep":
            selected.append(name)

    return selected, correlation_dict, decision_dict


def subset_sequence_features(
    X: np.ndarray,
    feature_names: list[str],
    selected_features: list[str],
) -> np.ndarray:
    """按特征名子集筛选 3D 序列张量的最后一维。"""
    indices = [feature_names.index(f) for f in selected_features]
    return X[:, :, indices]


def log_feature_selection_report(
    logger: logging.Logger,
    *,
    correlation_threshold: float,
    force_keep: list[str],
    feature_names: list[str],
    selected_features: list[str],
    correlation_dict: dict[str, float],
    decision_dict: dict[str, str],
    save_path: Path | None = None,
) -> None:
    """输出特征筛选日志到 logger。"""
    removed = [f for f in feature_names if f not in selected_features]
    force_in_data = [f for f in force_keep if f in feature_names]

    logger.info("=== 特征相关性自动筛选 ===")
    logger.info("筛选阈值: r > %.4f (|r| <= %.4f 或 r < -%.4f 剔除)",
                correlation_threshold, correlation_threshold, correlation_threshold)
    logger.info("强制保留特征: %s", ", ".join(force_in_data) if force_in_data else "(无)")
    logger.info("")
    logger.info("特征相关性评估结果:")
    logger.info("  %-28s  %10s  %s", "特征名称", "相关系数", "决策")
    for name in feature_names:
        r = correlation_dict[name]
        decision = decision_dict[name]
        mark = "✅" if name in selected_features else "❌"
        logger.info("  %-28s  %10.4f  %s %s", name, r, mark, decision)
    logger.info("")
    logger.info(
        "筛选结果: 原始 %d 个特征 → 保留 %d 个特征，剔除 %d 个特征",
        len(feature_names), len(selected_features), len(removed),
    )
    if save_path is not None:
        logger.info("已保存特征列表到: %s", save_path.relative_to(PROJECT_ROOT))


def save_feature_selection_result(
    horizon: int,
    *,
    correlation_threshold: float,
    original_features: list[str],
    selected_features: list[str],
    correlation_dict: dict[str, float],
    decision_dict: dict[str, str] | None = None,
    target_mode: str = "residual",
    method: str = "pearson_correlation",
) -> Path:
    """保存特征筛选 JSON。"""
    FEATURE_SELECTION_DIR.mkdir(parents=True, exist_ok=True)
    removed = [f for f in original_features if f not in selected_features]
    payload = {
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "horizon": horizon,
        "target_mode": target_mode,
        "method": method,
        "correlation_threshold": correlation_threshold,
        "original_features": original_features,
        "selected_features": selected_features,
        "removed_features": removed,
        "correlation_dict": correlation_dict,
        "decisions": decision_dict or {},
    }
    path = FEATURE_SELECTION_DIR / f"selected_features_h{horizon}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_feature_selection_result(horizon: int) -> dict[str, Any] | None:
    path = FEATURE_SELECTION_DIR / f"selected_features_h{horizon}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def get_feature_selection_config(base_cfg: dict) -> dict[str, Any]:
    """读取 feature_selection 配置并填充默认值。"""
    fs = base_cfg.get("feature_selection", {})
    return {
        "enabled": fs.get("enabled", False),
        "method": fs.get("method", "pearson_correlation"),
        "correlation_threshold": float(fs.get("correlation_threshold", 0.05)),
        "force_keep": list(fs.get("force_keep", DEFAULT_FORCE_KEEP)),
    }


def apply_feature_selection_to_sequences(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    fs_cfg: dict[str, Any],
    logger: logging.Logger,
    horizon: int,
    *,
    target_mode: str = "residual",
    persist: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    """
    在训练集上拟合相关性筛选，并应用到 train/val/test 序列。
    若 enabled=False，原样返回。
    """
    if not fs_cfg.get("enabled", False):
        return X_train, X_val, X_test, feature_names, {}

    threshold = fs_cfg["correlation_threshold"]
    force_keep = fs_cfg["force_keep"]

    cached = load_feature_selection_result(horizon)
    if (
        cached
        and cached.get("original_features") == feature_names
        and cached.get("correlation_threshold") == threshold
        and cached.get("target_mode") == target_mode
    ):
        selected = cached["selected_features"]
        correlation_dict = cached["correlation_dict"]
        decision_dict = cached.get("decisions", {})
        logger.info("加载已有特征筛选结果: selected_features_h%d.json (%d → %d 特征)",
                    horizon, len(feature_names), len(selected))
    else:
        selected, correlation_dict, decision_dict = filter_features_by_correlation(
            X_train, y_train, feature_names, threshold, force_keep,
        )
        save_path = None
        if persist:
            save_path = save_feature_selection_result(
                horizon,
                correlation_threshold=threshold,
                original_features=feature_names,
                selected_features=selected,
                correlation_dict=correlation_dict,
                decision_dict=decision_dict,
                target_mode=target_mode,
                method=fs_cfg.get("method", "pearson_correlation"),
            )
        log_feature_selection_report(
            logger,
            correlation_threshold=threshold,
            force_keep=force_keep,
            feature_names=feature_names,
            selected_features=selected,
            correlation_dict=correlation_dict,
            decision_dict=decision_dict,
            save_path=save_path,
        )

    if len(selected) == len(feature_names):
        result = load_feature_selection_result(horizon) or {
            "selected_features": selected,
            "correlation_dict": correlation_dict,
        }
        return X_train, X_val, X_test, feature_names, result

    X_train = subset_sequence_features(X_train, feature_names, selected)
    X_val = subset_sequence_features(X_val, feature_names, selected)
    X_test = subset_sequence_features(X_test, feature_names, selected)

    result = load_feature_selection_result(horizon) or {
        "original_features": feature_names,
        "selected_features": selected,
        "correlation_dict": correlation_dict,
        "removed_features": [f for f in feature_names if f not in selected],
    }
    return X_train, X_val, X_test, selected, result


def load_sample_arrays_with_feature_selection(
    hdir: Path,
    base_cfg: dict,
    logger: logging.Logger,
    horizon: int,
    *,
    target_mode: str = "residual",
) -> dict[str, Any]:
    """加载样本并在启用时应用特征筛选（仅基于训练集决策）。"""
    meta = json.loads((hdir / "meta.json").read_text(encoding="utf-8"))
    feature_names = list(meta["feature_cols"])

    out: dict[str, Any] = {
        "X_train": np.load(hdir / "X_train_seq.npy"),
        "X_val": np.load(hdir / "X_val_seq.npy"),
        "X_test": np.load(hdir / "X_test_seq.npy"),
        "y_train": np.load(hdir / "y_train.npy"),
        "y_val": np.load(hdir / "y_val.npy"),
        "y_test": np.load(hdir / "y_test.npy"),
        "meta": meta,
    }
    for key, fname in (
        ("y_anchor_train", "y_anchor_train.npy"),
        ("y_anchor_val", "y_anchor_val.npy"),
        ("y_anchor_test", "y_anchor_test.npy"),
    ):
        p = hdir / fname
        out[key] = np.load(p) if p.exists() else None

    fs_cfg = get_feature_selection_config(base_cfg)

    if meta.get("feature_selection_applied") and fs_cfg["enabled"]:
        logger.info(
            "样本已含特征筛选 (原始 %d → 当前 %d 特征)",
            meta.get("n_features_original", meta["n_features"]),
            meta["n_features"],
        )
        out["feature_names"] = feature_names
        out["selection_result"] = load_feature_selection_result(horizon)
        return out

    if not fs_cfg["enabled"]:
        out["feature_names"] = feature_names
        out["selection_result"] = None
        return out

    X_train, X_val, X_test, selected, result = apply_feature_selection_to_sequences(
        out["X_train"], out["X_val"], out["X_test"],
        out["y_train"], feature_names, fs_cfg, logger, horizon,
        target_mode=target_mode, persist=True,
    )
    out["X_train"] = X_train
    out["X_val"] = X_val
    out["X_test"] = X_test
    out["feature_names"] = selected
    out["selection_result"] = result
    return out
