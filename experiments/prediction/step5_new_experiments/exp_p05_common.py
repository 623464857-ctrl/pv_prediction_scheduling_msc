"""EXP-P05 共享路径、配置、日志、指标与样本加载工具。"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STEP5_ROOT = PROJECT_ROOT / "data" / "prediction" / "step5_new_experiments"
STEP4_ROOT = PROJECT_ROOT / "data" / "prediction" / "step4_optuna_hybrid"
CONFIG_DIR = STEP5_ROOT / "config"
SAMPLES_DIR = STEP5_ROOT / "samples"
MODELS_DIR = STEP5_ROOT / "models"
PRED_DIR = STEP5_ROOT / "predictions"
METRICS_DIR = STEP5_ROOT / "metrics"
FIGURES_DIR = STEP5_ROOT / "figures"
BENCHMARK_DIR = STEP5_ROOT / "benchmark"
REPORTS_DIR = STEP5_ROOT / "reports"
LOG_DIR = PROJECT_ROOT / "logs" / "prediction" / "step5_new_experiments"

MODEL_DISPLAY_NAMES = {
    "persistence": "Persistence",
    "moving_average": "Moving Average",
    "ridge": "Ridge Regression",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "lstm": "LSTM",
    "bilstm": "BiLSTM",
    "cnn_lstm": "CNN-LSTM",
    "cnn_bilstm": "CNN-BiLSTM",
    "patchtst": "PatchTST",
    "lstm_residual": "LSTM (Residual)",
    "bilstm_residual": "BiLSTM (Residual)",
    "cnn_lstm_residual": "CNN-LSTM (Residual)",
    "cnn_bilstm_residual": "CNN-BiLSTM (Residual)",
    "patchtst_residual": "PatchTST (Residual)",
    "proposed": "Proposed",
}

RESIDUAL_MODELS = ["lstm", "bilstm", "cnn_lstm", "cnn_bilstm", "patchtst"]


def load_config(name: str = "exp_p05_base.json") -> dict:
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


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def horizon_dir(horizon: int) -> str:
    return f"h{horizon}"


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    capacity: float = 1.0,
) -> dict:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mask = np.abs(y_true) > 0.01
    if mask.any():
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    else:
        mape = float("nan")
    r2 = float(r2_score(y_true, y_pred))
    nrmse = float(rmse / capacity) if capacity > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2, "nRMSE": nrmse}


def build_daytime_mask(
    y_true: np.ndarray,
    daylight_flag: np.ndarray | None = None,
    capacity: float = 1.0,
    threshold_ratio: float = 0.01,
) -> np.ndarray:
    y_flat = np.asarray(y_true).ravel()
    power_mask = y_flat > threshold_ratio * capacity
    if daylight_flag is None:
        return power_mask
    daylight_flat = np.asarray(daylight_flag).ravel()
    # 多步预测时 y_true 展平为 N*H，daylight_flag 仍为 N，需对齐
    if len(daylight_flat) != len(y_flat):
        if len(y_flat) % len(daylight_flat) == 0:
            repeat = len(y_flat) // len(daylight_flat)
            daylight_flat = np.repeat(daylight_flat, repeat)
        else:
            return power_mask
    daylight_mask = daylight_flat == 1
    return daylight_mask | power_mask


def compute_segmented_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    daylight_flag: np.ndarray | None = None,
    capacity: float = 1.0,
    threshold_ratio: float = 0.01,
) -> dict:
    all_day = compute_all_metrics(y_true, y_pred, capacity=capacity)
    mask = build_daytime_mask(y_true, daylight_flag, capacity, threshold_ratio)
    if mask.any():
        daytime = compute_all_metrics(y_true[mask], y_pred[mask], capacity=capacity)
    else:
        daytime = {k: float("nan") for k in all_day}
    return {"all_day": all_day, "daytime_only": daytime}


def load_samples(horizon: int, use_step5: bool = True) -> dict:
    base = (SAMPLES_DIR if use_step5 else STEP4_ROOT / "samples") / horizon_dir(horizon)
    data = {
        "X_train_seq": np.load(base / "X_train_seq.npy"),
        "X_val_seq": np.load(base / "X_val_seq.npy"),
        "X_test_seq": np.load(base / "X_test_seq.npy"),
        "y_train": np.load(base / "y_train.npy"),
        "y_val": np.load(base / "y_val.npy"),
        "y_test": np.load(base / "y_test.npy"),
    }
    for split in ("train", "val", "test"):
        raw_path = base / f"y_{split}_raw.npy"
        if raw_path.exists():
            data[f"y_{split}_raw"] = np.load(raw_path)
    last_path = base / "y_last_test.npy"
    if last_path.exists():
        data["y_last_test"] = np.load(last_path)
    for split in ("train", "val"):
        p = base / f"y_last_{split}.npy"
        if p.exists():
            data[f"y_last_{split}"] = np.load(p)
    daylight_path = base / "daylight_flag_test.npy"
    if daylight_path.exists():
        data["daylight_flag_test"] = np.load(daylight_path)
    return data


def load_step4_samples(horizon: int) -> dict:
    return load_samples(horizon, use_step5=False)


def load_meta(horizon: int, use_step5: bool = True) -> dict:
    base = (SAMPLES_DIR if use_step5 else STEP4_ROOT / "samples") / horizon_dir(horizon)
    with open(base / "meta.json", encoding="utf-8") as f:
        return json.load(f)


def load_step4_meta(horizon: int) -> dict:
    return load_meta(horizon, use_step5=False)


def load_test_timestamps(horizon: int, use_step5: bool = True) -> pd.Series:
    base = (SAMPLES_DIR if use_step5 else STEP4_ROOT / "samples") / horizon_dir(horizon)
    return pd.read_csv(base / "test_timestamps.csv", parse_dates=["timestamp"])["timestamp"]


def load_y_scaler(horizon: int, use_step5: bool = True):
    from sklearn.preprocessing import StandardScaler

    base = (SAMPLES_DIR if use_step5 else STEP4_ROOT / "samples") / horizon_dir(horizon)
    params = json.loads((base / "scaler_params.json").read_text(encoding="utf-8"))
    scaler = StandardScaler()
    scaler.mean_ = np.array(params["y_mean"])
    scaler.scale_ = np.array(params["y_scale"])
    scaler.n_features_in_ = len(params["y_mean"])
    return scaler


def load_step4_scaler(horizon: int):
    return load_y_scaler(horizon, use_step5=False)


def load_residual_scaler(horizon: int):
    from sklearn.preprocessing import StandardScaler

    path = SAMPLES_DIR / horizon_dir(horizon) / "residual_scaler_params.json"
    params = json.loads(path.read_text(encoding="utf-8"))
    scaler = StandardScaler()
    scaler.mean_ = np.array(params["mean"])
    scaler.scale_ = np.array(params["scale"])
    scaler.n_features_in_ = len(params["mean"])
    return scaler


def save_predictions(
    horizon: int,
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    suffix: str = "",
) -> Path:
    ts = load_test_timestamps(horizon, use_step5=True)
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    # 对齐多步预测形状：y_true (N,H) vs y_pred (N,) 时扩展预测
    if y_true_arr.ndim > 1 and y_pred_arr.ndim == 1 and len(y_pred_arr) == y_true_arr.shape[0]:
        y_pred_arr = np.repeat(y_pred_arr.reshape(-1, 1), y_true_arr.shape[1], axis=1)
    elif y_true_arr.ndim > 1 and y_pred_arr.ndim > 1 and y_pred_arr.shape != y_true_arr.shape:
        raise ValueError(
            f"预测形状不匹配: y_true={y_true_arr.shape}, y_pred={y_pred_arr.shape}, model={model_name}"
        )

    y_true_flat = y_true_arr.ravel()
    y_pred_flat = y_pred_arr.ravel()
    if len(y_true_flat) != len(y_pred_flat):
        raise ValueError(
            f"保存预测失败，长度不一致: y_true={len(y_true_flat)}, y_pred={len(y_pred_flat)}, model={model_name}"
        )

    n_flat = len(y_true_flat)
    n_ts = len(ts)
    if n_flat != n_ts:
        ts_vals = np.repeat(ts.values, max(1, n_flat // max(n_ts, 1)))[:n_flat]
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
    fname = f"{model_name}{suffix}_test.csv"
    path = PRED_DIR / horizon_dir(horizon) / fname
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return path


def flatten_sequences(X: np.ndarray) -> np.ndarray:
    n_samples = X.shape[0]
    return X.reshape(n_samples, -1)


def sort_results_by_rmse(results: dict[str, dict]) -> list[tuple[str, dict]]:
    return sorted(results.items(), key=lambda kv: (kv[1].get("RMSE", float("inf")), kv[1].get("MAE", float("inf"))))
