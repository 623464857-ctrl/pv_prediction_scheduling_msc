"""EXP-P05 传统强基线：Persistence / Moving Average / Ridge / XGBoost / LightGBM。"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from experiments.prediction.step5_new_experiments.exp_p05_common import compute_all_metrics, flatten_sequences


def _target_matrix(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim == 1:
        return y.reshape(-1, 1)
    return y


def _metrics_vs_true(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return compute_all_metrics(np.asarray(y_true).ravel(), np.asarray(y_pred).ravel())


class PersistenceBaseline:
    def predict(self, X: np.ndarray | None, y_last: np.ndarray, horizon: int = 1) -> np.ndarray:
        y_last = np.asarray(y_last).reshape(-1, 1)
        if horizon == 1:
            return y_last.ravel()
        return np.repeat(y_last, horizon, axis=1)


class MovingAverageBaseline:
    def __init__(self, window: int = 4):
        self.window = window

    def predict(self, X: np.ndarray, horizon: int = 1) -> np.ndarray:
        seq = X[:, -self.window :, 0]
        ma = seq.mean(axis=1, keepdims=True)
        if horizon == 1:
            return ma.ravel()
        return np.repeat(ma, horizon, axis=1)


class RidgeBaseline:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)
        self.scaler = StandardScaler()
        self.horizon = 1

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeBaseline":
        X_flat = flatten_sequences(X)
        y_target = _target_matrix(y)
        self.horizon = y_target.shape[1]
        X_scaled = self.scaler.fit_transform(X_flat)
        self.model.fit(X_scaled, y_target)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_flat = flatten_sequences(X)
        X_scaled = self.scaler.transform(X_flat)
        pred = self.model.predict(X_scaled)
        if self.horizon == 1:
            return np.asarray(pred).ravel()
        return np.asarray(pred)


class XGBoostBaseline:
    def __init__(self, **params):
        self.params = params
        self.model = None
        self.scaler = StandardScaler()
        self.horizon = 1

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostBaseline":
        from xgboost import XGBRegressor

        X_flat = flatten_sequences(X)
        y_target = _target_matrix(y)
        self.horizon = y_target.shape[1]
        X_scaled = self.scaler.fit_transform(X_flat)
        base = XGBRegressor(**self.params)
        if self.horizon == 1:
            self.model = base
            self.model.fit(X_scaled, y_target.ravel())
        else:
            self.model = MultiOutputRegressor(base)
            self.model.fit(X_scaled, y_target)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_flat = flatten_sequences(X)
        X_scaled = self.scaler.transform(X_flat)
        pred = self.model.predict(X_scaled)
        if self.horizon == 1:
            return np.asarray(pred).ravel()
        return np.asarray(pred)


class LightGBMBaseline:
    def __init__(self, **params):
        self.params = params
        self.model = None
        self.scaler = StandardScaler()
        self.horizon = 1

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LightGBMBaseline":
        import lightgbm as lgb

        X_flat = flatten_sequences(X)
        y_target = _target_matrix(y)
        self.horizon = y_target.shape[1]
        X_scaled = self.scaler.fit_transform(X_flat)
        base = lgb.LGBMRegressor(**self.params)
        if self.horizon == 1:
            self.model = base
            self.model.fit(X_scaled, y_target.ravel())
        else:
            self.model = MultiOutputRegressor(base)
            self.model.fit(X_scaled, y_target)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_flat = flatten_sequences(X)
        X_scaled = self.scaler.transform(X_flat)
        pred = self.model.predict(X_scaled)
        if self.horizon == 1:
            return np.asarray(pred).ravel()
        return np.asarray(pred)


def evaluate_persistence(y_true: np.ndarray, y_last: np.ndarray, horizon: int = 1) -> dict:
    y_pred = PersistenceBaseline().predict(None, y_last, horizon=horizon)
    return _metrics_vs_true(y_true, y_pred)


def evaluate_moving_average(X_test: np.ndarray, y_true: np.ndarray, window: int = 4, horizon: int = 1) -> dict:
    y_pred = MovingAverageBaseline(window=window).predict(X_test, horizon=horizon)
    return _metrics_vs_true(y_true, y_pred)


def evaluate_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float = 1.0,
) -> dict:
    model = RidgeBaseline(alpha=alpha)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return _metrics_vs_true(y_test, y_pred)


def evaluate_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    params: dict | None = None,
) -> dict:
    model = XGBoostBaseline(**(params or {}))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return _metrics_vs_true(y_test, y_pred)


def evaluate_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    params: dict | None = None,
) -> dict:
    model = LightGBMBaseline(**(params or {}))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return _metrics_vs_true(y_test, y_pred)
