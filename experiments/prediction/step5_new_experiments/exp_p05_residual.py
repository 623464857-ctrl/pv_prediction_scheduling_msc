"""EXP-P05 残差预测建模工具。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler


def compute_residual_targets(y: np.ndarray, y_last: np.ndarray) -> np.ndarray:
    """Δy = y_future - y_last。"""
    y = np.asarray(y, dtype=np.float32)
    y_last = np.asarray(y_last, dtype=np.float32)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if y_last.ndim == 1:
        y_last = y_last.reshape(-1, 1)
    if y.shape[1] == 1:
        return (y - y_last).astype(np.float32)
    # 多步预测：每一步相对同一 y_last
    return (y - y_last).astype(np.float32)


def reconstruct_from_residual(y_last: np.ndarray, delta_pred: np.ndarray) -> np.ndarray:
    """y_hat_future = y_last + Δy_hat。"""
    y_last = np.asarray(y_last, dtype=np.float32)
    delta_pred = np.asarray(delta_pred, dtype=np.float32)
    if y_last.ndim == 1:
        y_last = y_last.reshape(-1, 1)
    if delta_pred.ndim == 1:
        delta_pred = delta_pred.reshape(-1, 1)
    return (y_last + delta_pred).astype(np.float32)


def fit_residual_scaler(y_residual_train: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(y_residual_train)
    return scaler


def save_residual_scaler(scaler: StandardScaler, path: Path) -> None:
    payload = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def transform_residual(scaler: StandardScaler, y_residual: np.ndarray) -> np.ndarray:
    return scaler.transform(y_residual).astype(np.float32)


def inverse_transform_residual(scaler: StandardScaler, y_residual_scaled: np.ndarray) -> np.ndarray:
    return scaler.inverse_transform(y_residual_scaled).astype(np.float32)
