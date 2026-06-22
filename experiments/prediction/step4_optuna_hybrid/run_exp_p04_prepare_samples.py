"""
python -m experiments.prediction.step4_optuna_hybrid.run_exp_p04_prepare_samples --horizon 1
python -m experiments.prediction.step4_optuna_hybrid.run_exp_p04_prepare_samples --horizon 4
python -m experiments.prediction.step4_optuna_hybrid.run_exp_p04_prepare_samples --horizon 16
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


def build_windows(
    features: np.ndarray,
    target: np.ndarray,
    lookback: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构造滑动窗口样本，返回 (X_seq, y, valid_idx)。"""
    n = len(target)
    n_samples = n - lookback - horizon + 1
    if n_samples <= 0:
        raise ValueError("时序长度不足，无法构造窗口样本")

    X_seq = np.zeros((n_samples, lookback, features.shape[1]), dtype=np.float32)
    y = np.zeros((n_samples, horizon), dtype=np.float32)
    valid = np.ones(n_samples, dtype=bool)

    for i in range(n_samples):
        x_win = features[i : i + lookback]
        y_win = target[i + lookback : i + lookback + horizon]
        if np.isnan(x_win).any() or np.isnan(y_win).any():
            valid[i] = False
        X_seq[i] = x_win
        y[i] = y_win

    return X_seq[valid], y[valid]


def main():
    parser = argparse.ArgumentParser(description="构造多 horizon 样本并保存")
    parser.add_argument("--horizon", type=int, choices=[1, 4, 16], required=True)
    args = parser.parse_args()

    horizon = args.horizon
    hdir = SAMPLES_DIR / f"h{horizon}"
    hdir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("prepare_samples", f"EXP-P04_h{horizon}_prepare_samples.log")
    logger.info("=" * 60)
    logger.info("开始构造 horizon=%d 样本", horizon)
    t0 = time.time()

    # ── 1. 加载原始数据（与 step3 相同数据源） ─────────────────────────────
    base_cfg = load_config("exp_p04_base.json")
    data_path = PROJECT_ROOT / base_cfg["data_raw_path"]
    logger.info("加载数据: %s", data_path)

    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info("原始行数: %d  |  时间范围: %s ~ %s",
                 len(df), df["timestamp"].iloc[0], df["timestamp"].iloc[-1])

    # ── 2. 验证特征列 ─────────────────────────────────────────────────────
    missing = [c for c in STEP1_FEATURES + ["power_pu"] if c not in df.columns]
    if missing:
        logger.error("缺少列: %s", missing)
        sys.exit(1)

    # ── 3. 构造序列 ────────────────────────────────────────────────────────
    lookback = base_cfg["lookback"]
    features = df[STEP1_FEATURES].to_numpy(dtype=np.float64)
    target = df["power_pu"].to_numpy(dtype=np.float64)

    logger.info("构造序列: lookback=%d, horizon=%d, 特征数=%d", lookback, horizon, len(STEP1_FEATURES))
    X_all, y_all = build_windows(features, target, lookback, horizon)
    logger.info("总样本数: %d  |  X shape: %s  |  y shape: %s",
                 len(X_all), X_all.shape, y_all.shape)

    # ── 4. 时序划分（70/15/15） ───────────────────────────────────────────
    n = len(X_all)
    n_train_val = int(n * (base_cfg["train_frac"] + base_cfg["val_frac"]))
    n_train = int(n_train_val * base_cfg["train_frac"] /
                  (base_cfg["train_frac"] + base_cfg["val_frac"]))
    n_val = n_train_val - n_train
    n_test = n - n_train_val

    X_train, X_val, X_test = X_all[:n_train], X_all[n_train:n_train_val], X_all[n_train_val:]
    y_train, y_val, y_test = y_all[:n_train], y_all[n_train:n_train_val], y_all[n_train_val:]

    logger.info("训练集: %d  验证集: %d  测试集: %d", n_train, n_val, n_test)

    # ── 5. 标准化 ────────────────────────────────────────────────────────
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, len(STEP1_FEATURES)))

    def transform_X(X):
        shape = X.shape
        return scaler.transform(X.reshape(-1, shape[-1])).reshape(shape).astype(np.float32)

    X_train_s = transform_X(X_train)
    X_val_s = transform_X(X_val)
    X_test_s = transform_X(X_test)

    # y 标准化
    y_scaler = StandardScaler()
    y_scaler.fit(y_train)
    y_train_s = y_scaler.transform(y_train).astype(np.float32)
    y_val_s = y_scaler.transform(y_val).astype(np.float32)
    y_test_s = y_scaler.transform(y_test).astype(np.float32)

    # ── 6. 保存 ───────────────────────────────────────────────────────────
    def save_npy(arr, name):
        p = hdir / name
        np.save(p, arr)
        logger.info("  保存 %s  ->  shape=%s", name, arr.shape)

    save_npy(X_train_s, "X_train_seq.npy")
    save_npy(X_val_s, "X_val_seq.npy")
    save_npy(X_test_s, "X_test_seq.npy")
    save_npy(y_train_s, "y_train.npy")
    save_npy(y_val_s, "y_val.npy")
    save_npy(y_test_s, "y_test.npy")
    # 原始 y（用于最终评估指标计算）
    save_npy(y_train.astype(np.float32), "y_train_raw.npy")
    save_npy(y_val.astype(np.float32), "y_val_raw.npy")
    save_npy(y_test.astype(np.float32), "y_test_raw.npy")

    # 保存 scaler 参数
    scaler_params = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "feature_cols": STEP1_FEATURES,
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
        "n_features": len(STEP1_FEATURES),
        "feature_cols": STEP1_FEATURES,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "train_frac": base_cfg["train_frac"],
        "val_frac": base_cfg["val_frac"],
        "test_frac": base_cfg["test_frac"],
        "source_csv": str(data_path.relative_to(PROJECT_ROOT)),
        "total_windows": int(n),
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
