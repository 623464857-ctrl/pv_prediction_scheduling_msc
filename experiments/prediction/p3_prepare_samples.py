"""
预测 P3: 构造模型输入特征和滑动窗口样本

输入: data/processed/Bozhou_1_cleaned.csv
输出: result/prediction/samples/h{H}_lb{L}/
        X_{train,val,test}.npy   (样本数, L, 12)  已用训练集均值/标准差标准化
        y_{train,val,test}.npy   (样本数, H)      未来 H 步 power_pu, 未缩放
        {val,test}_timestamps.csv                 每个样本第 1 个预测点的时刻
        meta.json, scaler_params.json

划分方式: 按时间顺序 70% / 15% / 15%

python -m experiments.prediction.p3_prepare_samples            # 默认 H=1
python -m experiments.prediction.p3_prepare_samples --horizon 1 4 16
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.common import CAPACITY_KW, CLEANED_CSV, LOOKBACK, sample_dir

# 13 个输入特征 (精简版: 去除 lag_16, ramp_60m, roll_2h_mean, roll_2h_std)
INPUT_FEATURES = [
    # 功率滞后 (power_pu = power_kw / CAPACITY_KW)
    "power_pu_lag_0", "power_pu_lag_4", "power_pu_lag_8",
    # 功率爬坡 (15 min 变化量)
    "power_ramp_15m_pu",
    # 滚动统计 (1h)
    "power_pu_roll_1h_mean", "power_pu_roll_1h_std",
    # 天气
    "ghi_wm2", "temperature_c", "relative_humidity_pct", "cloud_cover_pct",
    # 时间
    "sin_hour", "cos_hour",
]
SPLIT = (0.70, 0.15, 0.15)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """在清洗后的数据上构造功率衍生特征 (只用当前及过去的数据)。

    精简版: 去除 lag_16, ramp_60m, roll_2h_mean, roll_2h_std
    """
    p = df["power_kw"] / CAPACITY_KW
    df["power_pu"] = p
    for k in (0, 4, 8):  # 精简: 去掉 lag_16
        df[f"power_pu_lag_{k}"] = p.shift(k)
    # 精简: 去掉 ramp_60m，只保留 ramp_15m
    for name, w in (("1h", 4),):  # 精简: 去掉 roll_2h
        roll = p.rolling(w, min_periods=1)
        df[f"power_pu_roll_{name}_mean"] = roll.mean()
        df[f"power_pu_roll_{name}_std"] = roll.std()
    return df.fillna(0)


def make_windows(df: pd.DataFrame, lookback: int, horizon: int):
    """滑动窗口: 用第 i..i+L-1 步的特征预测第 i+L..i+L+H-1 步的 power_pu。"""
    values = df[INPUT_FEATURES].to_numpy(np.float32)
    power = df["power_pu"].to_numpy(np.float32)
    n = len(df) - lookback - horizon + 1
    X = sliding_window_view(values, lookback, axis=0)[:n].transpose(0, 2, 1)   # (n, L, F)
    y = np.stack([power[lookback + k: lookback + k + n] for k in range(horizon)], axis=1)  # (n, H)
    first_target_ts = df["timestamp"].iloc[lookback: lookback + n].to_numpy()
    return np.ascontiguousarray(X), y, first_target_ts


def prepare(horizon: int) -> dict:
    lookback = LOOKBACK[horizon]
    df = build_features(pd.read_csv(CLEANED_CSV, parse_dates=["timestamp"]))
    X, y, ts = make_windows(df, lookback, horizon)

    n = len(X)
    n_train, n_val = int(n * SPLIT[0]), int(n * SPLIT[1])
    idx = {"train": slice(0, n_train), "val": slice(n_train, n_train + n_val), "test": slice(n_train + n_val, n)}

    # 输入标准化: 每个特征用训练集的均值/标准差 (GHI ~1000, 功率 ~0.4, 量级差异很大)
    flat = X[idx["train"]].reshape(-1, X.shape[-1]).astype(np.float64)
    x_mean, x_std = flat.mean(axis=0), flat.std(axis=0)
    x_std[x_std < 1e-6] = 1.0  # 防止标准差为 0 的特征导致除零

    out = sample_dir(horizon)
    out.mkdir(parents=True, exist_ok=True)
    for name, s in idx.items():
        np.save(out / f"X_{name}.npy", ((X[s] - x_mean) / x_std).astype(np.float32))
        np.save(out / f"y_{name}.npy", y[s])
        if name != "train":
            pd.DataFrame({"timestamp": ts[s]}).to_csv(out / f"{name}_timestamps.csv", index=False)

    meta = {
        "horizon": horizon, "lookback": lookback, "n_features": len(INPUT_FEATURES),
        "feature_cols": INPUT_FEATURES, "capacity_kw": CAPACITY_KW, "split": SPLIT,
        "n_train": n_train, "n_val": n_val, "n_test": n - n_train - n_val,
        "test_start": str(ts[idx["test"]][0]), "source_csv": CLEANED_CSV.name,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "scaler_params.json").write_text(json.dumps(
        {"feature_cols": INPUT_FEATURES, "x_mean": x_mean.tolist(), "x_std": x_std.tolist()}, indent=2),
        encoding="utf-8")
    return meta


def main():
    parser = argparse.ArgumentParser(description="构造滑动窗口样本")
    parser.add_argument("--horizon", type=int, nargs="+", default=[1], choices=list(LOOKBACK))
    args = parser.parse_args()
    for h in args.horizon:
        m = prepare(h)
        print(f"H={h} L={m['lookback']}: train={m['n_train']} val={m['n_val']} test={m['n_test']} "
              f"(测试集从 {m['test_start']} 开始) -> {sample_dir(h)}")


if __name__ == "__main__":
    main()
