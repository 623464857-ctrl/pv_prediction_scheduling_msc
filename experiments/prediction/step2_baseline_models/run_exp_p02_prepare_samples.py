"""
实验编号: EXP-P02-prepare
实验名称: 基础预测模型对比 — 样本构造与数据划分
实验目的: 从 Site_1 清洗长表构造滑动窗口样本，按时序 70/15/15 划分，并用训练集拟合标准化器
所属方向: prediction / step2_baseline_models
输入路径: data/prediction/step1_preprocessing/processed/stations/Site_1_preprocessed.csv
输出路径: data/prediction/step2_baseline_models/samples/
运行方式: python experiments/prediction/step2_baseline_models/run_exp_p02_prepare_samples.py
统一约束: lookback=16, horizon=1, 13 特征, 目标 power_pu, 禁止 shuffle
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from exp_p02_common import (  # noqa: E402
    PROJECT_ROOT,
    SAMPLES_DIR,
    append_log_summary,
    load_config,
    setup_logger,
)

LOG_NAME = "EXP-P02_prepare.log"


def build_windows(
    features: np.ndarray,
    target: np.ndarray,
    lookback: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(target)
    n_samples = n - lookback - horizon + 1
    if n_samples <= 0:
        raise ValueError("时序长度不足以构造窗口样本")

    X_seq = np.zeros((n_samples, lookback, features.shape[1]), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.float32)
    valid = np.ones(n_samples, dtype=bool)

    for i in range(n_samples):
        x_win = features[i : i + lookback]
        y_val = target[i + lookback + horizon - 1]
        if np.isnan(x_win).any() or np.isnan(y_val):
            valid[i] = False
        X_seq[i] = x_win
        y[i] = y_val

    timestamps_idx = np.arange(lookback + horizon - 1, lookback + horizon - 1 + n_samples)
    return X_seq[valid], y[valid], timestamps_idx[valid]


def main() -> None:
    logger = setup_logger("EXP-P02-prepare", LOG_NAME)
    cfg = load_config()

    src = PROJECT_ROOT / cfg["source_csv"]
    if not src.exists():
        logger.error("未找到输入文件: %s", src)
        sys.exit(1)

    feature_cols = cfg["features"]
    target_col = cfg["target"]
    lookback = cfg["lookback"]
    horizon = cfg["horizon"]

    logger.info("读取 Site_%s: %s", cfg["site_id"], src.name)
    df = pd.read_csv(src, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    missing_cols = [c for c in feature_cols + [target_col] if c not in df.columns]
    if missing_cols:
        logger.error("缺少列: %s", missing_cols)
        sys.exit(1)

    features = df[feature_cols].to_numpy(dtype=np.float64)
    target = df[target_col].to_numpy(dtype=np.float64)
    timestamps = df["timestamp"].to_numpy()

    X_seq, y, sample_ts_idx = build_windows(features, target, lookback, horizon)
    sample_timestamps = timestamps[sample_ts_idx]

    n = len(y)
    n_train = int(n * cfg["train_ratio"])
    n_val = int(n * cfg["val_ratio"])
    n_test = n - n_train - n_val

    splits = {
        "train": (0, n_train),
        "val": (n_train, n_train + n_val),
        "test": (n_train + n_val, n),
    }

    X_train = X_seq[splits["train"][0] : splits["train"][1]]
    X_val = X_seq[splits["val"][0] : splits["val"][1]]
    X_test = X_seq[splits["test"][0] : splits["test"][1]]

    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, len(feature_cols)))

    def transform(X: np.ndarray) -> np.ndarray:
        shape = X.shape
        flat = scaler.transform(X.reshape(-1, shape[-1]))
        return flat.reshape(shape).astype(np.float32)

    X_train_s = transform(X_train)
    X_val_s = transform(X_val)
    X_test_s = transform(X_test)

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    np.save(SAMPLES_DIR / "X_train_seq.npy", X_train_s)
    np.save(SAMPLES_DIR / "X_val_seq.npy", X_val_s)
    np.save(SAMPLES_DIR / "X_test_seq.npy", X_test_s)
    np.save(SAMPLES_DIR / "X_train_flat.npy", X_train_s.reshape(len(X_train_s), -1))
    np.save(SAMPLES_DIR / "X_val_flat.npy", X_val_s.reshape(len(X_val_s), -1))
    np.save(SAMPLES_DIR / "X_test_flat.npy", X_test_s.reshape(len(X_test_s), -1))
    np.save(SAMPLES_DIR / "y_train.npy", y[splits["train"][0] : splits["train"][1]].astype(np.float32))
    np.save(SAMPLES_DIR / "y_val.npy", y[splits["val"][0] : splits["val"][1]].astype(np.float32))
    np.save(SAMPLES_DIR / "y_test.npy", y[splits["test"][0] : splits["test"][1]].astype(np.float32))

    test_ts = sample_timestamps[splits["test"][0] : splits["test"][1]]
    pd.DataFrame({"timestamp": test_ts}).to_csv(SAMPLES_DIR / "test_timestamps.csv", index=False)

    scaler_params = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "feature_cols": feature_cols,
        "fitted_on": "train_set_only",
        "n_train_windows": len(X_train_s),
    }
    with open(SAMPLES_DIR / "scaler_params.json", "w", encoding="utf-8") as f:
        json.dump(scaler_params, f, indent=2, ensure_ascii=False)

    meta = {
        "experiment_id": cfg["experiment_id"],
        "site_id": cfg["site_id"],
        "target": target_col,
        "features": feature_cols,
        "n_features": len(feature_cols),
        "lookback": lookback,
        "horizon": horizon,
        "flat_input_dim": lookback * len(feature_cols),
        "n_samples": n,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "split": {k: list(v) for k, v in splits.items()},
        "train_ratio": cfg["train_ratio"],
        "val_ratio": cfg["val_ratio"],
        "test_ratio": cfg["test_ratio"],
        "random_seed": cfg["random_seed"],
        "time_split_only": True,
        "train_time_start": str(sample_timestamps[splits["train"][0]]),
        "train_time_end": str(sample_timestamps[splits["train"][1] - 1]),
        "val_time_start": str(sample_timestamps[splits["val"][0]]),
        "val_time_end": str(sample_timestamps[splits["val"][1] - 1]),
        "test_time_start": str(sample_timestamps[splits["test"][0]]),
        "test_time_end": str(sample_timestamps[splits["test"][1] - 1]),
        "source_csv": str(src.relative_to(PROJECT_ROOT)),
    }
    with open(SAMPLES_DIR / "site1_window_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    logger.info("样本总数=%d | 训练=%d 验证=%d 测试=%d", n, n_train, n_val, n_test)
    logger.info("展平输入维度=%d | 序列输入=[%d, %d]", meta["flat_input_dim"], lookback, len(feature_cols))
    logger.info("测试集时间: %s ~ %s", meta["test_time_start"], meta["test_time_end"])
    logger.info("样本已写入 %s", SAMPLES_DIR.relative_to(PROJECT_ROOT))

    append_log_summary(
        LOG_NAME,
        [
            "=" * 60,
            "【EXP-P02-prepare 摘要】",
            f"- Site_1 窗口样本: {n} 条 (lookback={lookback}, horizon={horizon})",
            f"- 特征数: {len(feature_cols)}, 展平维度: {meta['flat_input_dim']}",
            f"- 划分: train={n_train}, val={n_val}, test={n_test} (时序顺序, 无 shuffle)",
            "- 标准化: StandardScaler, 仅训练集拟合",
            "- 产出: samples/*.npy, scaler_params.json, site1_window_meta.json",
            "=" * 60,
        ],
    )
    logger.info("EXP-P02-prepare 结束")


if __name__ == "__main__":
    main()
