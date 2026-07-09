"""
Step 2: 特征工程增强 + 样本构造
python experiments/prediction/step5_new_experiments/run_exp_p05_prepare_samples.py --horizon 1
"""

from __future__ import annotations

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

from experiments.prediction.step5_new_experiments.exp_p05_common import (
    PROJECT_ROOT as ROOT,
    SAMPLES_DIR,
    ensure_dirs,
    load_config,
    setup_logger,
)
from experiments.prediction.step5_new_experiments.exp_p05_features import (
    FEATURE_VERSION,
    build_p05_features,
    build_windows_from_df,
    get_p05_feature_columns,
)


def main():
    parser = argparse.ArgumentParser(description="EXP-P05 增强特征样本构造")
    parser.add_argument("--horizon", type=int, choices=[1, 4, 16], required=True)
    args = parser.parse_args()

    horizon = args.horizon
    hdir = SAMPLES_DIR / f"h{horizon}"
    ensure_dirs(hdir)

    logger = setup_logger("prepare_samples", f"EXP-P05_h{horizon}_prepare_samples.log")
    logger.info("=" * 60)
    logger.info("EXP-P05 Step2 特征增强样本构造 horizon=%d", horizon)
    t0 = time.time()

    cfg = load_config()
    data_path = ROOT / cfg["data_raw_path"]
    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.set_index("timestamp")

    df_feat = build_p05_features(df)
    feature_cols = get_p05_feature_columns()
    missing = [c for c in feature_cols + ["power_pu"] if c not in df_feat.columns]
    if missing:
        logger.error("缺少列: %s", missing)
        sys.exit(1)

    df_feat = df_feat.dropna(subset=feature_cols + ["power_pu"]).reset_index()
    lookback = cfg["lookback"]

    X_all, y_all, y_last_all, day_all, ts_all, _ = build_windows_from_df(
        df_feat, feature_cols, lookback, horizon
    )
    logger.info("总样本: %d  X=%s  y=%s", len(X_all), X_all.shape, y_all.shape)

    n = len(X_all)
    n_train_val = int(n * (cfg["train_frac"] + cfg["val_frac"]))
    n_train = int(n_train_val * cfg["train_frac"] / (cfg["train_frac"] + cfg["val_frac"]))
    n_val = n_train_val - n_train

    splits = {
        "train": slice(0, n_train),
        "val": slice(n_train, n_train_val),
        "test": slice(n_train_val, n),
    }

    X_train, X_val, X_test = X_all[splits["train"]], X_all[splits["val"]], X_all[splits["test"]]
    y_train, y_val, y_test = y_all[splits["train"]], y_all[splits["val"]], y_all[splits["test"]]
    y_last_train = y_last_all[splits["train"]]
    y_last_val = y_last_all[splits["val"]]
    y_last_test = y_last_all[splits["test"]]
    day_test = day_all[splits["test"]]
    ts_test = ts_all[splits["test"]]

    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, len(feature_cols)))

    def transform_x(x):
        s = x.shape
        return scaler.transform(x.reshape(-1, s[-1])).reshape(s).astype(np.float32)

    y_scaler = StandardScaler()
    y_scaler.fit(y_train)

    def save(name, arr):
        np.save(hdir / name, arr)
        logger.info("保存 %s shape=%s", name, arr.shape)

    save("X_train_seq.npy", transform_x(X_train))
    save("X_val_seq.npy", transform_x(X_val))
    save("X_test_seq.npy", transform_x(X_test))
    save("y_train.npy", y_scaler.transform(y_train).astype(np.float32))
    save("y_val.npy", y_scaler.transform(y_val).astype(np.float32))
    save("y_test.npy", y_scaler.transform(y_test).astype(np.float32))
    save("y_train_raw.npy", y_train.astype(np.float32))
    save("y_val_raw.npy", y_val.astype(np.float32))
    save("y_test_raw.npy", y_test.astype(np.float32))
    save("y_last_train.npy", y_last_train.astype(np.float32))
    save("y_last_val.npy", y_last_val.astype(np.float32))
    save("y_last_test.npy", y_last_test.astype(np.float32))
    save("daylight_flag_test.npy", day_test.astype(np.float32))

    scaler_params = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "feature_cols": feature_cols,
        "y_mean": y_scaler.mean_.tolist(),
        "y_scale": y_scaler.scale_.tolist(),
        "feature_version": FEATURE_VERSION,
    }
    (hdir / "scaler_params.json").write_text(json.dumps(scaler_params, indent=2, ensure_ascii=False), encoding="utf-8")

    pd.DataFrame({"timestamp": pd.to_datetime(ts_test)}).to_csv(hdir / "test_timestamps.csv", index=False)

    meta = {
        "lookback": lookback,
        "horizon": horizon,
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "feature_version": FEATURE_VERSION,
        "n_train": int(n_train),
        "n_val": int(n_val),
        "n_test": int(len(y_test)),
        "source_csv": str(data_path.relative_to(ROOT)),
    }
    (hdir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("完成，耗时 %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
