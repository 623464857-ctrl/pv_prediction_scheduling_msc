"""
python -m experiments.prediction.step4_optuna_hybrid.run_exp_p04_prepare_samples --horizon 1
python -m experiments.prediction.step4_optuna_hybrid.run_exp_p04_prepare_samples --horizon 4 --lookback 48
python -m experiments.prediction.step4_optuna_hybrid.run_exp_p04_prepare_samples --horizon 16 --lookback 96
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.prediction.step4_optuna_hybrid.exp_p04_common import (
    SAMPLES_DIR,
    load_config,
    setup_logger,
)
from experiments.prediction.step4_optuna_hybrid.exp_p04_step_audit import (
    record_step_failure,
    record_step_result,
)

# =============================================================================
# 功率历史特征配置
# =============================================================================
# 电站容量 (MW) - 用于计算归一化功率
CAPACITY_MW = None  # 动态从数据中获取

# 滞后特征列表 (历史时刻数)
POWER_LAG_COLS = [
    "power_pu_lag_0",   # t-0 (当前时刻)
    "power_pu_lag_1",   # t-1 (15分钟前)
    "power_pu_lag_2",   # t-2 (30分钟前)
    "power_pu_lag_4",   # t-4 (1小时前)
    "power_pu_lag_8",   # t-8 (2小时前)
    "power_pu_lag_16",  # t-16 (4小时前)
]

# 多尺度Ramp特征列表
POWER_RAMP_COLS = [
    "power_ramp_15m_pu",  # 15分钟变化率
    "power_ramp_60m_pu",  # 60分钟变化率
    "power_ramp_120m_pu", # 120分钟变化率
]

# 滚动统计特征列表
POWER_ROLL_COLS = [
    "power_pu_roll_1h_mean",   # 1小时滚动均值
    "power_pu_roll_1h_std",    # 1小时滚动标准差
    "power_pu_roll_2h_mean",   # 2小时滚动均值
    "power_pu_roll_2h_std",    # 2小时滚动标准差
    "power_pu_roll_2h_max",    # 2小时滚动最大值
    "power_pu_roll_2h_min",    # 2小时滚动最小值
]

# 模型输入特征列表 (15 + 3 + 3 + 4 + 2 + 1 = 28个)
# 功率历史(15) + 辐照(3) + 气象(3) + 时间(4) + 状态(2) + 质量(1)
FEATURE_COLUMNS = [
    # === 功率历史特征 (15个) ===
    # 滞后特征
    "power_pu_lag_0",
    "power_pu_lag_1",
    "power_pu_lag_2",
    "power_pu_lag_4",
    "power_pu_lag_8",
    "power_pu_lag_16",
    # 多尺度Ramp
    "power_ramp_15m_pu",
    "power_ramp_60m_pu",
    "power_ramp_120m_pu",
    # 滚动统计
    "power_pu_roll_1h_mean",
    "power_pu_roll_1h_std",
    "power_pu_roll_2h_mean",
    "power_pu_roll_2h_std",
    "power_pu_roll_2h_max",
    "power_pu_roll_2h_min",

    # === 辐照特征 (3个) ===
    "total_irradiance_wm2",
    "direct_normal_irradiance_wm2",
    "global_horizontal_irradiance_wm2",

    # === 气象特征 (3个) ===
    "air_temperature_c",
    "atmosphere_hpa",
    "relative_humidity_pct",

    # === 时间特征 (4个) ===
    "hour_sin",
    "hour_cos",
    "sin_dayofyear",
    "cos_dayofyear",

    # === 状态标志 (2个) ===
    "is_peak_hour",    # 峰值时段标志(11:00-14:00)
    "daylight_flag",   # 日间标志(GTI > 5 W/m² 为白天)

    # === 质量评分 (1个) ===
    "data_quality_score",
]


def compute_power_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算功率相关特征：滞后、ramp、滚动统计

    所有特征只使用当前时刻及历史时刻，无未来泄漏。
    """
    df = df.copy()
    capacity = df["capacity_mw"].iloc[0] if "capacity_mw" in df.columns else 50.0

    # -------------------------------------------------------------------------
    # 1. 基础功率归一化
    # -------------------------------------------------------------------------
    if "power_pu" not in df.columns:
        df["power_pu"] = df["power_mw"] / capacity

    # -------------------------------------------------------------------------
    # 2. 滞后特征 (只使用历史值，无未来泄漏)
    # -------------------------------------------------------------------------
    lag_steps = [0, 1, 2, 4, 8, 16]
    for lag in lag_steps:
        col_name = f"power_pu_lag_{lag}"
        if lag == 0:
            df[col_name] = df["power_pu"]
        else:
            df[col_name] = df["power_pu"].shift(lag)

    # -------------------------------------------------------------------------
    # 3. 多尺度Ramp特征 (功率变化率)
    # -------------------------------------------------------------------------
    # 15分钟变化率 (1步 = 15分钟)
    df["power_ramp_15m_pu"] = df["power_pu"].diff(1)

    # 60分钟变化率 (4步 = 60分钟)
    df["power_ramp_60m_pu"] = df["power_pu"].diff(4)

    # 120分钟变化率 (8步 = 120分钟)
    df["power_ramp_120m_pu"] = df["power_pu"].diff(8)

    # -------------------------------------------------------------------------
    # 4. 滚动统计特征
    # -------------------------------------------------------------------------
    # 1小时滚动窗口 (4步 = 60分钟)
    df["power_pu_roll_1h_mean"] = df["power_pu"].shift(1).rolling(window=4, min_periods=1).mean()
    df["power_pu_roll_1h_std"] = df["power_pu"].shift(1).rolling(window=4, min_periods=1).std()

    # 2小时滚动窗口 (8步 = 120分钟)
    df["power_pu_roll_2h_mean"] = df["power_pu"].shift(1).rolling(window=8, min_periods=1).mean()
    df["power_pu_roll_2h_std"] = df["power_pu"].shift(1).rolling(window=8, min_periods=1).std()
    df["power_pu_roll_2h_max"] = df["power_pu"].shift(1).rolling(window=8, min_periods=1).max()
    df["power_pu_roll_2h_min"] = df["power_pu"].shift(1).rolling(window=8, min_periods=1).min()

    # -------------------------------------------------------------------------
    # 5. 日间标志 (基于辐照度阈值)
    # -------------------------------------------------------------------------
    # GTI > 5 W/m² 认为处于日间
    df["daylight_flag"] = (df["total_irradiance_wm2"] > 5).astype(float)

    return df


def build_windows(
    features: np.ndarray,
    target: np.ndarray,
    lookback: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """构造滑动窗口样本，返回 (X_seq, y_residual, y_last, valid_idx).

    残差预测: Δy = y_future - y_last
    """
    n = len(target)
    n_samples = n - lookback - horizon + 1
    if n_samples <= 0:
        raise ValueError("时序长度不足，无法构造窗口样本")

    X_seq = np.zeros((n_samples, lookback, features.shape[1]), dtype=np.float32)
    y_residual = np.zeros((n_samples, horizon), dtype=np.float32)
    y_last = np.zeros((n_samples, horizon), dtype=np.float32)
    valid = np.ones(n_samples, dtype=bool)

    for i in range(n_samples):
        x_win = features[i : i + lookback]
        y_last_val = target[i + lookback - 1]
        y_future = target[i + lookback : i + lookback + horizon]
        y_delta = y_future - y_last_val

        if np.isnan(x_win).any() or np.isnan(y_delta).any() or np.isnan(y_last_val):
            valid[i] = False
            continue

        X_seq[i] = x_win
        y_residual[i] = y_delta
        y_last[i] = y_last_val

    return X_seq[valid], y_residual[valid], y_last[valid], valid


def main():
    parser = argparse.ArgumentParser(description="构造多 horizon 样本并保存")
    parser.add_argument("--horizon", type=int, choices=[1, 4, 16], required=True)
    parser.add_argument("--lookback", type=int, default=None,
                        help="lookback 步数 (默认: base.json 中的值)")
    args = parser.parse_args()

    horizon = args.horizon
    base_cfg = load_config("exp_p04_base.json")
    lookback = args.lookback if args.lookback is not None else base_cfg["lookback"]

    hdir = SAMPLES_DIR / f"h{horizon}_lb{lookback}"
    hdir.mkdir(parents=True, exist_ok=True)

    log_file = f"EXP-P04_h{horizon}_lb{lookback}_prepare_samples.log"
    logger = setup_logger("prepare_samples", log_file)
    logger.info("=" * 60)
    logger.info("开始构造 horizon=%d 样本  lookback=%d", horizon, lookback)
    t0 = time.time()

    data_path = PROJECT_ROOT / base_cfg["data_raw_path"]
    logger.info("加载数据: %s", data_path)

    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info("原始行数: %d  |  时间范围: %s ~ %s",
                len(df), df["timestamp"].iloc[0], df["timestamp"].iloc[-1])

    # 计算功率历史特征 (滞后、ramp、滚动统计)
    logger.info("计算功率历史特征...")
    df = compute_power_features(df)
    logger.info("功率特征计算完成，新增 %d 个特征", len(df.columns))

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        logger.error("缺少特征列: %s", missing)
        raise ValueError(f"缺少特征列: {missing}")

    feature_arr = df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    target = df["power_pu"].to_numpy(dtype=np.float64)

    logger.info("构造序列: lookback=%d, horizon=%d, n_features=%d",
                lookback, horizon, len(FEATURE_COLUMNS))

    X_all, y_residual_all, y_anchor_all, _ = build_windows(
        feature_arr, target, lookback, horizon
    )
    n_total_features = X_all.shape[2]

    logger.info("总样本数: %d  |  X shape: %s  |  残差 y shape: %s",
                len(X_all), X_all.shape, y_residual_all.shape)

    n = len(X_all)
    n_train_val = int(n * (base_cfg["train_frac"] + base_cfg["val_frac"]))
    n_train = int(n_train_val * base_cfg["train_frac"] /
                  (base_cfg["train_frac"] + base_cfg["val_frac"]))
    n_val = n_train_val - n_train
    n_test = n - n_train_val

    X_train, X_val, X_test = X_all[:n_train], X_all[n_train:n_train_val], X_all[n_train_val:]
    y_residual_train = y_residual_all[:n_train]
    y_residual_val = y_residual_all[n_train:n_train_val]
    y_residual_test = y_residual_all[n_train_val:]
    y_anchor_train = y_anchor_all[:n_train]
    y_anchor_val = y_anchor_all[n_train:n_train_val]
    y_anchor_test = y_anchor_all[n_train_val:]

    logger.info("训练集: %d  验证集: %d  测试集: %d", n_train, n_val, n_test)

    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, n_total_features))

    def transform_X(X):
        shape = X.shape
        return scaler.transform(X.reshape(-1, shape[-1])).reshape(shape).astype(np.float32)

    X_train_s = transform_X(X_train)
    X_val_s = transform_X(X_val)
    X_test_s = transform_X(X_test)

    y_scaler = StandardScaler()
    y_scaler.fit(y_residual_train)
    y_residual_train_s = y_scaler.transform(y_residual_train).astype(np.float32)
    y_residual_val_s = y_scaler.transform(y_residual_val).astype(np.float32)
    y_residual_test_s = y_scaler.transform(y_residual_test).astype(np.float32)

    def save_npy(arr, name):
        p = hdir / name
        np.save(p, arr)
        logger.info("  保存 %s  ->  shape=%s", name, arr.shape)

    save_npy(X_train_s, "X_train_seq.npy")
    save_npy(X_val_s, "X_val_seq.npy")
    save_npy(X_test_s, "X_test_seq.npy")
    save_npy(y_residual_train_s, "y_train.npy")
    save_npy(y_residual_val_s, "y_val.npy")
    save_npy(y_residual_test_s, "y_test.npy")
    save_npy(y_anchor_train.astype(np.float32), "y_anchor_train.npy")
    save_npy(y_anchor_val.astype(np.float32), "y_anchor_val.npy")
    save_npy(y_anchor_test.astype(np.float32), "y_anchor_test.npy")
    save_npy(y_residual_train.astype(np.float32), "y_residual_train_raw.npy")
    save_npy(y_residual_val.astype(np.float32), "y_residual_val_raw.npy")
    save_npy(y_residual_test.astype(np.float32), "y_residual_test_raw.npy")

    scaler_params = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "feature_cols": FEATURE_COLUMNS,
        "n_total_features": n_total_features,
        "y_mean": y_scaler.mean_.tolist(),
        "y_scale": y_scaler.scale_.tolist(),
    }
    with open(hdir / "scaler_params.json", "w", encoding="utf-8") as f:
        json.dump(scaler_params, f, indent=2, ensure_ascii=False)
    logger.info("  保存 scaler_params.json")

    test_ts_start = n_train_val + lookback + horizon - 1
    test_timestamps = df["timestamp"].iloc[test_ts_start : test_ts_start + n_test].reset_index(drop=True)
    pd.DataFrame({"timestamp": test_timestamps}).to_csv(hdir / "test_timestamps.csv", index=False)
    logger.info("  保存 test_timestamps.csv (%d 行)", n_test)

    meta = {
        "lookback": lookback,
        "horizon": horizon,
        "n_features": n_total_features,
        "feature_cols": FEATURE_COLUMNS,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "train_frac": base_cfg["train_frac"],
        "val_frac": base_cfg["val_frac"],
        "test_frac": base_cfg["test_frac"],
        "source_csv": str(data_path.relative_to(PROJECT_ROOT)),
        "total_windows": int(n),
        "prediction_mode": "residual",
        "residual_formula": "Delta_y = y_future - y_anchor (y_anchor = power at t_lookback-1)",
    }
    with open(hdir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    logger.info("  保存 meta.json")

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("样本构造完成！耗时: %.1f 秒", elapsed)
    logger.info("保存目录: %s", hdir.relative_to(PROJECT_ROOT))

    record_step_result(
        horizon, "prepare_samples", "success", log_file,
        summary={
            "lookback": lookback,
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
            "X_shape": list(X_train_s.shape),
            "sample_dir": str(hdir.relative_to(PROJECT_ROOT)),
            "elapsed_sec": round(elapsed, 1),
        },
        duration_sec=elapsed,
        artifacts=[
            str((hdir / "meta.json").relative_to(PROJECT_ROOT)),
            str((hdir / "X_train_seq.npy").relative_to(PROJECT_ROOT)),
        ],
    )
    return horizon, log_file


if __name__ == "__main__":
    t0 = time.time()
    try:
        main()
    except Exception as e:
        record_step_failure("prepare_samples", t0, e)
        raise
