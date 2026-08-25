"""
python -m experiments.prediction.step4_optuna_hybrid.run_exp_p04_prepare_samples --horizon 1
python -m experiments.prediction.step4_optuna_hybrid.run_exp_p04_prepare_samples --horizon 4 --lookback 48 --wrf_version physical
python -m experiments.prediction.step4_optuna_hybrid.run_exp_p04_prepare_samples --horizon 16 --lookback 96 --wrf_version minimal
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

# 与 step3 相同的 13 个特征（由 step1 预计算）
STEP1_FEATURES = [
    "power_ramp_15m_pu",
    "total_irradiance_wm2",
    "direct_normal_irradiance_wm2",
    "global_horizontal_irradiance_wm2",
    "air_temperature_c",
    "atmosphere_hpa",
    "relative_humidity_pct",
    "daylight_flag",
    "sin_hour",
    "cos_hour",
    "sin_dayofyear",
    "cos_dayofyear",
    "data_quality_score",
]

# WRF 天气预报特征全集（9个，plan.md Section 3 定义）
WRF_FORECAST_FEATURES = [
    "wrf_gti_wm2",
    "wrf_tsi_wm2",
    "wrf_clearness_index",
    "wrf_temperature_c",
    "wrf_relative_humidity_pct",
    "wrf_weather_code",
    "wrf_dew_point_c",
    "wrf_cloud_cover_ratio",
]

# WRF 特征子集（plan.md Section 3）
# - full:  全部 8 个特征
# - physical: 辐照度 + 晴空指数 + 云量 + 温度（5个）
# - minimal: 仅辐照度 + 云量（2个）
WRF_FEATURE_SUBSETS = {
    "full": WRF_FORECAST_FEATURES,
    "physical": [
        "wrf_gti_wm2",
        "wrf_tsi_wm2",
        "wrf_clearness_index",
        "wrf_cloud_cover_ratio",
        "wrf_temperature_c",
    ],
    "minimal": [
        "wrf_gti_wm2",
        "wrf_cloud_cover_ratio",
    ],
}

# Open-Meteo forecast_hourly 数据说明：
# 每个时间戳 T 的值 = 在 T-4h 发出的预报，内容是 T 时刻的天气预报
# 即：wrf[T] 是预报发出时间 = T-4h，预报覆盖时间 = T
# 这意味着 wrf[t+h] 在预测时刻 t 时已存在（因为它在 t+h-4h 就已发出）
#
# 对齐策略：对每个 horizon 步 h，用 wrf[t+h] 作为该步的 WRF 核心特征
# 这样模型在预测 power[t+h] 时，直接看到的是"在 t 时刻已知的、对 t+h 时刻天气的预报"
# 而 lookback 窗口中的 WRF 特征则提供历史趋势信息


def build_windows(
    features: np.ndarray,
    target: np.ndarray,
    lookback: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """构造滑动窗口样本，返回 (X_seq, y_residual, y_last, valid_idx).

    残差预测: Δy = y_future - y_last
    - y_future: 预测窗口的实际功率值
    - y_last: 预测窗口开始前最后一个已知功率值 (t_lookback-1)
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


def build_windows_with_forecast_aligned(
    lookback_features: np.ndarray,
    wrf_features: np.ndarray,
    target: np.ndarray,
    lookback: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """构造带 per-horizon aligned WRF forecast features 的滑动窗口样本.

    核心设计 (每个 lookback 步同时包含 step1 + WRF aligned forecast):
    - X_seq[t, step, 0:n_step1]       = step1 历史特征 (辐照度/功率等)
    - X_seq[t, step, n_step1:]         = wrf[t + step] = 对 t+step 时刻天气的预报
                                           wrf[T] 在 T-4h 发出，T 时刻可见
                                           所以 step=15 (最后一步) 的 WRF = wrf[t+15]
                                           覆盖 t+15min 的天气 -> 直接预测 power[t+15min]

    残差预测: delta_y = y_future - y_last
    - y_future: 预测窗口的功率值 (len = horizon)
    - y_last: 锚点功率值 (t_lookback-1)
    """
    n = len(target)
    n_samples = n - lookback - horizon + 1
    if n_samples <= 0:
        raise ValueError("时序长度不足，无法构造窗口样本")

    n_step1 = lookback_features.shape[1]
    n_wrf = wrf_features.shape[1]
    total_features = n_step1 + n_wrf

    X_seq = np.zeros((n_samples, lookback, total_features), dtype=np.float32)
    y_residual = np.zeros((n_samples, horizon), dtype=np.float32)
    y_last = np.zeros((n_samples, horizon), dtype=np.float32)
    valid = np.ones(n_samples, dtype=bool)

    for i in range(n_samples):
        t_pred = i + lookback  # 预测时刻索引

        # 构建 lookback 窗口: 每步包含 step1[step] + wrf[t_pred + step]
        # step 0: step1[t_pred-16], wrf[t_pred]      (4h old forecast for t_pred)
        # step 15: step1[t_pred-1], wrf[t_pred+15]   (0h old forecast for t_pred+15)
        x_combined = np.zeros((lookback, total_features), dtype=np.float32)
        for step in range(lookback):
            step1_idx = t_pred - lookback + step  # step1 index for this lookback step
            wrf_idx = t_pred + step               # aligned WRF index

            x_combined[step, 0:n_step1] = lookback_features[step1_idx]
            if wrf_idx < len(wrf_features):
                x_combined[step, n_step1:] = wrf_features[wrf_idx]
            else:
                x_combined[step, n_step1:] = wrf_features[-1]

        y_last_val = target[t_pred - 1]
        y_future = target[t_pred : t_pred + horizon]
        y_delta = y_future - y_last_val

        if (np.isnan(x_combined).any() or np.isnan(y_delta).any()
                or np.isnan(y_last_val)):
            valid[i] = False
            continue

        X_seq[i] = x_combined
        y_residual[i] = y_delta
        y_last[i] = y_last_val

    return X_seq[valid], y_residual[valid], y_last[valid], valid


def main():
    parser = argparse.ArgumentParser(description="构造多 horizon 样本并保存")
    parser.add_argument("--horizon", type=int, choices=[1, 4, 16], required=True)
    parser.add_argument("--lookback", type=int, default=None,
                        help="lookback 步数 (默认: base.json 中的值). "
                             "计划实验: 16, 32, 48, 96")
    parser.add_argument("--wrf_version", type=str, default=None,
                        choices=["full", "physical", "minimal"],
                        help="WRF 特征版本 (默认: full). "
                             "plan.md Phase 2 WRF 消融实验")
    args = parser.parse_args()

    horizon = args.horizon
    base_cfg = load_config("exp_p04_base.json")

    # lookback: 默认取 base.json 中的值
    lookback = args.lookback if args.lookback is not None else base_cfg["lookback"]

    # wrf_version: 默认取 "full"
    wrf_version = args.wrf_version if args.wrf_version is not None else "full"
    wrf_feature_list = WRF_FEATURE_SUBSETS[wrf_version]

    # 输出目录结构: h{horizon}_lb{lookback}_wrf_{version}
    # 同一 horizon 不同 lookback/wrf_version 的样本分开存储
    hdir = SAMPLES_DIR / f"h{horizon}_lb{lookback}_wrf_{wrf_version}"
    hdir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(
        "prepare_samples",
        f"EXP-P04_h{horizon}_lb{lookback}_wrf_{wrf_version}_prepare_samples.log"
    )
    logger.info("=" * 60)
    logger.info("开始构造 horizon=%d 样本  lookback=%d  wrf_version=%s",
                horizon, lookback, wrf_version)
    t0 = time.time()

    # ── 1. 加载原始数据（与 step3 相同数据源） ─────────────────────────────
    data_path = PROJECT_ROOT / base_cfg["data_raw_path"]
    logger.info("加载数据: %s", data_path)

    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info("原始行数: %d  |  时间范围: %s ~ %s",
                 len(df), df["timestamp"].iloc[0], df["timestamp"].iloc[-1])

    # ── 2. 验证特征列 ─────────────────────────────────────────────────────
    missing_step1 = [c for c in STEP1_FEATURES if c not in df.columns]
    missing_wrf = [c for c in wrf_feature_list if c not in df.columns]
    if missing_step1:
        logger.error("缺少 step1 列: %s", missing_step1)
        sys.exit(1)
    if missing_wrf:
        logger.error("缺少 WRF 列 (%s): %s", wrf_version, missing_wrf)
        sys.exit(1)

    # ── 3. 构造序列 ───────────────────────────────────────────────────────
    # 分离 step1 特征和 WRF 特征
    step1_arr = df[STEP1_FEATURES].to_numpy(dtype=np.float64)       # (N, 13)
    wrf_arr = df[wrf_feature_list].to_numpy(dtype=np.float64)      # (N, n_wrf)
    target = df["power_pu"].to_numpy(dtype=np.float64)             # (N,)

    logger.info("构造序列: lookback=%d, horizon=%d", lookback, horizon)
    logger.info("  step1 特征: %d 维 (历史 lookback 窗口)", len(STEP1_FEATURES))
    logger.info("  WRF forecast 特征: %d 维 (%s, per-horizon aligned)", len(wrf_feature_list), wrf_version)
    logger.info("  WRF aligned: lookback 步 step 的 WRF = wrf[t_pred + step]")
    logger.info("  即: step=0 时 WRF=wrf[t_pred] (覆盖 t_pred 天气, 4h old forecast)")
    logger.info("       step=last 时 WRF=wrf[t_pred+lookback-1] (覆盖 t_pred+lookback-1 天气, 0h old forecast)")
    logger.info("  核心: 最后一步 WRF 直接覆盖首个预测目标功率的天气条件")

    X_all, y_residual_all, y_anchor_all, valid_mask = build_windows_with_forecast_aligned(
        step1_arr, wrf_arr, target, lookback, horizon
    )
    # X_all shape: (n_samples, lookback, n_step1+n_wrf)
    n_total_features = X_all.shape[2]

    logger.info("总样本数: %d  |  X shape: %s  |  残差 y shape: %s",
                len(X_all), X_all.shape, y_residual_all.shape)

    # ── 4. 时序划分（70/15/15） ───────────────────────────────────────────
    n = len(X_all)
    n_train_val = int(n * (base_cfg["train_frac"] + base_cfg["val_frac"]))
    n_train = int(n_train_val * base_cfg["train_frac"] /
                  (base_cfg["train_frac"] + base_cfg["val_frac"]))
    n_val = n_train_val - n_train
    n_test = n - n_train_val

    X_train, X_val, X_test = X_all[:n_train], X_all[n_train:n_train_val], X_all[n_train_val:]
    # 残差目标
    y_residual_train, y_residual_val, y_residual_test = (
        y_residual_all[:n_train], y_residual_all[n_train:n_train_val], y_residual_all[n_train_val:]
    )
    # 锚点值 (用于最终重构)
    y_anchor_train, y_anchor_val, y_anchor_test = (
        y_anchor_all[:n_train], y_anchor_all[n_train:n_train_val], y_anchor_all[n_train_val:]
    )

    logger.info("训练集: %d  验证集: %d  测试集: %d", n_train, n_val, n_test)

    # ── 5. 标准化 ───────────────────────────────────────────────────────
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, n_total_features))

    def transform_X(X):
        shape = X.shape
        return scaler.transform(X.reshape(-1, shape[-1])).reshape(shape).astype(np.float32)

    X_train_s = transform_X(X_train)
    X_val_s = transform_X(X_val)
    X_test_s = transform_X(X_test)

    # 残差目标标准化
    y_scaler = StandardScaler()
    y_scaler.fit(y_residual_train)
    y_residual_train_s = y_scaler.transform(y_residual_train).astype(np.float32)
    y_residual_val_s = y_scaler.transform(y_residual_val).astype(np.float32)
    y_residual_test_s = y_scaler.transform(y_residual_test).astype(np.float32)

    # ── 6. 保存 ───────────────────────────────────────────────────────────
    def save_npy(arr, name):
        p = hdir / name
        np.save(p, arr)
        logger.info("  保存 %s  ->  shape=%s", name, arr.shape)

    save_npy(X_train_s, "X_train_seq.npy")
    save_npy(X_val_s, "X_val_seq.npy")
    save_npy(X_test_s, "X_test_seq.npy")
    # 残差目标 (标准化后，用于训练)
    save_npy(y_residual_train_s, "y_train.npy")
    save_npy(y_residual_val_s, "y_val.npy")
    save_npy(y_residual_test_s, "y_test.npy")
    # 锚点值 (用于预测后重构: y_hat = y_anchor + y_residual_pred)
    save_npy(y_anchor_train.astype(np.float32), "y_anchor_train.npy")
    save_npy(y_anchor_val.astype(np.float32), "y_anchor_val.npy")
    save_npy(y_anchor_test.astype(np.float32), "y_anchor_test.npy")
    # 原始残差值 (用于分析)
    save_npy(y_residual_train.astype(np.float32), "y_residual_train_raw.npy")
    save_npy(y_residual_val.astype(np.float32), "y_residual_val_raw.npy")
    save_npy(y_residual_test.astype(np.float32), "y_residual_test_raw.npy")

    # 保存 scaler 参数
    scaler_params = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "feature_cols": f"step1({len(STEP1_FEATURES)})+wrf_aligned_{wrf_version}({len(wrf_feature_list)})",
        "step1_feature_cols": STEP1_FEATURES,
        "wrf_forecast_feature_cols": wrf_feature_list,
        "n_total_features": n_total_features,
        "y_mean": y_scaler.mean_.tolist(),
        "y_scale": y_scaler.scale_.tolist(),
    }
    scaler_path = hdir / "scaler_params.json"
    with open(scaler_path, "w", encoding="utf-8") as f:
        json.dump(scaler_params, f, indent=2, ensure_ascii=False)
    logger.info("  保存 scaler_params.json")

    # 保存测试时间戳
    test_ts_start = n_train_val + lookback + horizon - 1
    test_timestamps = df["timestamp"].iloc[test_ts_start : test_ts_start + n_test].reset_index(drop=True)
    ts_path = hdir / "test_timestamps.csv"
    pd.DataFrame({"timestamp": test_timestamps}).to_csv(ts_path, index=False)
    logger.info("  保存 test_timestamps.csv (%d 行)", n_test)

    # 保存 meta
    meta = {
        "lookback": lookback,
        "horizon": horizon,
        "wrf_version": wrf_version,
        "n_features": n_total_features,
        "feature_cols": f"step1({len(STEP1_FEATURES)})+wrf_aligned_{wrf_version}({len(wrf_feature_list)})",
        "step1_feature_cols": STEP1_FEATURES,
        "wrf_forecast_feature_cols": wrf_feature_list,
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
        "feature_alignment": "per-horizon WRF forecast (wrf[t+h] at prediction step h, issue_lag=4h)",
    }
    meta_path = hdir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    logger.info("  保存 meta.json")

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("样本构造完成！耗时: %.1f 秒", elapsed)
    logger.info("保存目录: %s", hdir.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
