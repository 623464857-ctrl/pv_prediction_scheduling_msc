"""EXP-P04 共享路径、配置、日志与指标工具。"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STEP4_ROOT = PROJECT_ROOT / "data" / "prediction" / "step4_optuna_hybrid"
CONFIG_DIR = STEP4_ROOT / "config"
SAMPLES_DIR = STEP4_ROOT / "samples"
MODELS_DIR = STEP4_ROOT / "models"
PRED_DIR = STEP4_ROOT / "predictions"
METRICS_DIR = STEP4_ROOT / "metrics"
FIGURES_DIR = STEP4_ROOT / "figures"
REPORTS_DIR = STEP4_ROOT / "reports"
LOG_DIR = PROJECT_ROOT / "logs" / "prediction" / "step4_optuna_hybrid"

MODEL_DISPLAY_NAMES = {
    "lstm": "LSTM",
    "bilstm": "BiLSTM",
    "cnn_lstm": "CNN-LSTM",
    "cnn_bilstm": "CNN-BiLSTM",
    "minipatchtst": "MiniPatchTST",
    "patchtst": "PatchTST",
    "afsa_patchtst": "AFSA-PatchTST (old)",
}

MODEL_ORDER = ["lstm", "bilstm", "cnn_lstm", "cnn_bilstm", "minipatchtst", "afsa_patchtst"]


# ---------------------------------------------------------------------------
# Config / Logger
# ---------------------------------------------------------------------------

def load_config(name: str) -> dict:
    path = CONFIG_DIR / name
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def setup_logger(name: str, log_file: str) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / log_file
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8", mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    logger = logging.getLogger(name)
    logger.info("日志文件: %s", log_path.relative_to(PROJECT_ROOT))
    return logger


def append_log_summary(log_file: str, lines: list[str]) -> None:
    log_path = LOG_DIR / log_file
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n" + "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Dirs / Seed
# ---------------------------------------------------------------------------

def ensure_dirs(horizon: str) -> None:
    for d in (MODELS_DIR / horizon, PRED_DIR / horizon, METRICS_DIR / horizon,
              FIGURES_DIR / horizon):
        d.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------

def load_samples(horizon: str) -> dict:
    base = SAMPLES_DIR / horizon
    return {
        "X_train_seq": np.load(base / "X_train_seq.npy"),
        "X_val_seq": np.load(base / "X_val_seq.npy"),
        "X_test_seq": np.load(base / "X_test_seq.npy"),
        "y_train": np.load(base / "y_train.npy"),
        "y_val": np.load(base / "y_val.npy"),
        "y_test": np.load(base / "y_test.npy"),
    }


def load_test_timestamps(horizon: str) -> pd.Series:
    return pd.read_csv(SAMPLES_DIR / horizon / "test_timestamps.csv", parse_dates=["timestamp"])["timestamp"]


def load_y_scaler_from_json(horizon: str):
    """从 JSON 参数文件重建 y 的 StandardScaler。"""
    from sklearn.preprocessing import StandardScaler
    params = json.loads((SAMPLES_DIR / horizon / "scaler_params.json").read_text(encoding="utf-8"))
    scaler = StandardScaler()
    n_y = len(params["y_mean"])
    scaler.mean_ = np.array(params["y_mean"])
    scaler.scale_ = np.array(params["y_scale"])
    scaler.n_features_in_ = n_y
    scaler.n_samples_seen_ = None
    return scaler


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_predictions(horizon: str, model_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> Path:
    ts = load_test_timestamps(horizon)
    y_true_flat = np.asarray(y_true).ravel()
    y_pred_flat = np.asarray(y_pred).ravel()
    n_flat = len(y_true_flat)
    n_ts = len(ts)
    horizon_int = int(horizon.replace("h", "")) if isinstance(horizon, str) else horizon
    if n_flat != n_ts:
        # horizon > 1: repeat each timestamp for each horizon step
        ts_vals = np.repeat(ts.values, horizon_int)
        ts_vals = ts_vals[:n_flat]
    else:
        ts_vals = ts.values
    out = pd.DataFrame(
        {
            "timestamp": ts_vals,
            "y_true": y_true_flat,
            "y_pred": y_pred_flat,
            "model_name": model_name,
        }
    )
    path = PRED_DIR / horizon / f"{model_name}_test.csv"
    out.to_csv(path, index=False)
    return path


def save_train_history(horizon: str, model_key: str, history: list[dict]) -> Path:
    path = METRICS_DIR / horizon / f"{model_key}_train_history.csv"
    pd.DataFrame(history).to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mask = np.abs(y_true) > 0.01
    if mask.any():
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    else:
        mape = float("nan")
    r2 = float(r2_score(y_true, y_pred))
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


def compute_metrics_multi_step(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """y_true, y_pred shape: (N, horizon)，返回每步均值指标。"""
    horizon = y_true.shape[1]
    rows = []
    for h in range(horizon):
        m = compute_all_metrics(y_true[:, h], y_pred[:, h])
        m["horizon_step"] = h + 1
        rows.append(m)
    return rows


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_loss_curve(history_path, title: str, out_path) -> None:
    if not history_path.exists():
        return
    df = pd.read_csv(history_path)
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))
    plt.plot(df["epoch"], df["train_loss"], label="train_loss")
    plt.plot(df["epoch"], df["val_loss"], label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("MSE loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_pred_curve(df: pd.DataFrame, title: str, out_path, n_points: int = 500) -> None:
    import matplotlib.pyplot as plt

    sub = df.iloc[:n_points]
    plt.figure(figsize=(12, 4))
    plt.plot(sub["timestamp"], sub["y_true"], label="actual", linewidth=1.2)
    plt.plot(sub["timestamp"], sub["y_pred"], label="predicted", linewidth=1.0, alpha=0.85)
    plt.xlabel("timestamp")
    plt.ylabel("power_pu")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_overlay(preds: dict, n_points: int, out_path, horizon: str) -> None:
    import matplotlib.pyplot as plt

    first_key = next(iter(preds))
    first = preds[first_key]
    sub_ts = first["timestamp"].iloc[:n_points]
    y_true = first["y_true"].iloc[:n_points]

    plt.figure(figsize=(14, 5))
    plt.plot(sub_ts, y_true, label="actual", color="black", linewidth=1.5)
    colors = ["C0", "C1", "C2", "C3", "C4", "C5"]
    for (key, label), color in zip(
        [(k, MODEL_DISPLAY_NAMES.get(k, k)) for k in MODEL_ORDER if k in preds], colors
    ):
        sub = preds[key].iloc[:n_points]
        plt.plot(sub_ts, sub["y_pred"], label=label, alpha=0.8, color=color)
    plt.xlabel("timestamp")
    plt.ylabel("power_pu")
    plt.title(f"Test set predictions — all models ({horizon})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_metrics_bar(metrics_df: pd.DataFrame, out_path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    x = np.arange(len(metrics_df))
    for ax, col in zip(axes, ["MAE", "RMSE", "R2"]):
        ax.bar(x, metrics_df[col])
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_df["display_name"], rotation=20, ha="right")
        ax.set_title(col)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_training_time_comparison(metrics_df: pd.DataFrame, out_path) -> None:
    import matplotlib.pyplot as plt

    sub = metrics_df.dropna(subset=["training_time_sec"]).copy()
    if sub.empty:
        return
    sub = sub.sort_values("training_time_sec")
    plt.figure(figsize=(8, 4))
    bars = plt.barh(sub["display_name"], sub["training_time_sec"], color="steelblue")
    plt.xlabel("Training time (seconds)")
    plt.title("Training Time Comparison")
    for bar, val in zip(bars, sub["training_time_sec"]):
        plt.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}s",
            va="center",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
