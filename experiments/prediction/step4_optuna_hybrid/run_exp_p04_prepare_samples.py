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

FEATURE_COLUMNS = [
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
