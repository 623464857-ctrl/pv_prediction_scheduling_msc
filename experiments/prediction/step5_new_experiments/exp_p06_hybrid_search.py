"""EXP-P06 Optuna-AFSA 混合搜索（S1-S6 消融，对齐 Step4 搜索策略）。

与 Step4 一致的核心逻辑：
- Step 1: 用训练集后 1/3 做 quick subset 快速筛选参数（trial 级别）
- Step 2: 用最优参数在完整 3-fold CV 上评估，得到稳健的 val_RMSE
- 保留 S1-S6 混合消融实验

与 Step5 的区别：
- 不使用 full 50 epoch 训练，使用 quick subset + 3-fold 评估
- 支持快速模式（quick subset）和完整模式（完整训练数据 + 3-fold）
"""

from __future__ import annotations

import json
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.prediction.step4_optuna_hybrid.exp_p04_models import build_model
from experiments.prediction.step4_optuna_hybrid.exp_p04_torch_utils import (
    eval_loss,
    get_device,
    make_loader,
    train_with_early_stop,
)
from experiments.prediction.step5_new_experiments.exp_p05_benchmark import benchmark_forward


# ---------------------------------------------------------------------------
# 搜索空间
# ---------------------------------------------------------------------------
DEFAULT_SEARCH_SPACE = {
    "hidden": [32, 64, 128],
    "layers": [1, 2],
    "dropout": [0.1, 0.2, 0.3],
    "lr": [0.0005, 0.001, 0.002],
    "batch_size": [128, 256],
}


def _sample_from_space(rng: random.Random, space: dict) -> dict:
    return {k: rng.choice(v) for k, v in space.items()}


def _normalize_model_params(params: dict, model_name: str) -> dict:
    """将通用搜索空间中的 hidden/layers 映射到各模型构造器的实际参数名。"""
    name = model_name.lower().replace("-", "_")
    out = dict(params)
    hidden = out.pop("hidden", None)
    layers = out.pop("layers", None)

    if name in ("lstm", "bilstm"):
        if hidden is not None:
            out["hidden"] = hidden
        if layers is not None:
            out["layers"] = layers
    elif name == "cnn_lstm":
        if hidden is not None:
            out["lstm_hidden"] = hidden
        if layers is not None:
            out["lstm_layers"] = layers
    elif name == "cnn_bilstm":
        if hidden is not None:
            out["bilstm_hidden"] = hidden
        if layers is not None:
            out["bilstm_layers"] = layers
    elif name in ("minipatchtst", "patchtst"):
        if hidden is not None:
            out["d_model"] = hidden
        if layers is not None:
            out["num_layers"] = layers
    return out


# ---------------------------------------------------------------------------
# 3-fold 滚动窗口切分（与 Step4 一致）
# ---------------------------------------------------------------------------

def create_rolling_folds(n_total: int, n_folds: int = 3, train_frac: float = 0.7) -> list[tuple[np.ndarray, np.ndarray]]:
    """与 exp_p04_cv.create_rolling_folds 完全一致的切分逻辑。"""
    all_idx = np.arange(n_total)
    step = max(1, n_total // (n_folds + 1))
    folds = []
    for i in range(n_folds):
        tr_end = (i + 1) * step
        tr_start = max(0, int(tr_end - train_frac * n_total))
        tr_idx = all_idx[tr_start:tr_end]
        va_idx = all_idx[tr_end:min(tr_end + step, n_total)]
        folds.append((tr_idx, va_idx))
    return folds


# ---------------------------------------------------------------------------
# 单 trial 训练（quick subset 模式）
# ---------------------------------------------------------------------------

def _train_trial(
    params: dict,
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    meta: dict,
    horizon: int,
    max_epochs: int,
    patience: int,
    device: torch.device,
) -> tuple[float, torch.nn.Module]:
    """单次训练，返回 (val_rmse, model)。"""
    params = deepcopy(params)
    batch_size = int(params.pop("batch_size"))
    lr = float(params.pop("lr"))
    model_kwargs = _normalize_model_params(params, model_name)

    model = build_model(
        model_name,
        n_features=meta["n_features"],
        seq_len=meta["lookback"],
        horizon=horizon,
        **model_kwargs,
    ).to(device)

    train_loader = make_loader(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, batch_size=batch_size, shuffle=False)

    model, _ = train_with_early_stop(
        model, train_loader, val_loader,
        lr=lr, max_epochs=max_epochs, patience=patience, device=device,
    )
    val_rmse = float(np.sqrt(eval_loss(model, val_loader, nn.MSELoss(), device)))
    return val_rmse, model


# ---------------------------------------------------------------------------
# 推理基准（benchmark）
# ---------------------------------------------------------------------------

def _benchmark_model(model, X_sample: np.ndarray, device: torch.device) -> dict:
    sample = torch.from_numpy(X_sample.astype(np.float32))
    try:
        bench = benchmark_forward(model, sample, device=device, warmup_iters=3, repeat_iters=10)
        return {"latency_ms": bench["ms_per_sample"], "params": float(bench["params"])}
    except Exception:
        return {"latency_ms": -1.0, "params": -1.0}


# ---------------------------------------------------------------------------
# 三种搜索策略（与 Step4 策略一致：quick subset 训练）
# ---------------------------------------------------------------------------

def run_random_search(
    n_trials: int,
    model_name: str,
    X_quick: np.ndarray,
    y_quick: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    meta: dict,
    horizon: int,
    max_epochs: int,
    patience: int,
    seed: int,
    *,
    logger=None,
) -> list[dict]:
    """
    Random Search on quick subset. 每个 trial 返回 trial-level 结果。
    返回: [{"RMSE": ..., "train_params": ..., "model_state": ...}, ...]
    """
    rng = random.Random(seed)
    device = get_device()
    rows = []

    for i in range(n_trials):
        torch.manual_seed(42)
        np.random.seed(42)
        params = _sample_from_space(rng, DEFAULT_SEARCH_SPACE)
        val_rmse, model = _train_trial(
            deepcopy(params), model_name,
            X_quick, y_quick, X_val, y_val,
            meta, horizon, max_epochs, patience, device,
        )
        bench = _benchmark_model(model, X_quick[:512], device)
        row = {
            "RMSE": val_rmse,
            "MAE": val_rmse,
            "latency_ms": bench["latency_ms"],
            "params": bench["params"],
            "model_state": deepcopy(model.state_dict()),
            "train_params": {"batch_size": int(params["batch_size"]), "lr": float(params["lr"]),
                             **{k: v for k, v in params.items() if k not in ("batch_size", "lr")}},
        }
        rows.append(row)
        _log_trial(logger, i + 1, n_trials, row)
    return rows


def run_optuna_search(
    n_trials: int,
    model_name: str,
    X_quick: np.ndarray,
    y_quick: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    meta: dict,
    horizon: int,
    max_epochs: int,
    patience: int,
    seed: int,
    *,
    logger=None,
) -> list[dict]:
    """
    Optuna TPE Search on quick subset. 每个 trial 返回 trial-level 结果。
    """
    device = get_device()
    rows: list[dict] = []

    def objective(trial: optuna.Trial) -> float:
        torch.manual_seed(42)
        np.random.seed(42)
        params = {k: trial.suggest_categorical(k, v) for k, v in DEFAULT_SEARCH_SPACE.items()}
        val_rmse, model = _train_trial(
            deepcopy(params), model_name,
            X_quick, y_quick, X_val, y_val,
            meta, horizon, max_epochs, patience, device,
        )
        bench = _benchmark_model(model, X_quick[:512], device)
        row = {
            "RMSE": val_rmse,
            "MAE": val_rmse,
            "latency_ms": bench["latency_ms"],
            "params": bench["params"],
            "model_state": deepcopy(model.state_dict()),
            "train_params": {"batch_size": int(params["batch_size"]), "lr": float(params["lr"]),
                             **{k: v for k, v in params.items() if k not in ("batch_size", "lr")}},
        }
        rows.append(row)
        _log_trial(logger, len(rows), n_trials, row)
        return val_rmse

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return rows


def run_afsa_search(
    n_trials: int,
    model_name: str,
    X_quick: np.ndarray,
    y_quick: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    meta: dict,
    horizon: int,
    max_epochs: int,
    patience: int,
    seed: int,
    init_params: dict | None = None,
    *,
    logger=None,
) -> list[dict]:
    """
    简化 AFSA on quick subset: 随机邻域 + 向最优移动。
    """
    rng = random.Random(seed)
    device = get_device()
    rows = []
    best = None

    if init_params:
        p0 = deepcopy(init_params)
    else:
        p0 = _sample_from_space(rng, DEFAULT_SEARCH_SPACE)

    torch.manual_seed(42)
    np.random.seed(42)
    val_rmse, model = _train_trial(p0, model_name, X_quick, y_quick, X_val, y_val, meta, horizon, max_epochs, patience, device)
    bench = _benchmark_model(model, X_quick[:512], device)
    best = {
        "RMSE": val_rmse, "MAE": val_rmse, "latency_ms": bench["latency_ms"], "params": bench["params"],
        "model_state": deepcopy(model.state_dict()),
        "train_params": {"batch_size": int(p0["batch_size"]), "lr": float(p0["lr"]),
                         **{k: v for k, v in p0.items() if k not in ("batch_size", "lr")}},
    }
    rows.append(best)
    _log_trial(logger, 1, n_trials, best)

    for i in range(1, n_trials):
        torch.manual_seed(42)
        np.random.seed(42)
        cur = _sample_from_space(rng, DEFAULT_SEARCH_SPACE)
        if best and rng.random() < 0.5:
            merged = deepcopy(cur)
            for k in DEFAULT_SEARCH_SPACE:
                if rng.random() < 0.5:
                    merged[k] = best["train_params"].get(k, cur[k])
            cur = merged
        val_rmse, model = _train_trial(cur, model_name, X_quick, y_quick, X_val, y_val, meta, horizon, max_epochs, patience, device)
        bench = _benchmark_model(model, X_quick[:512], device)
        row = {
            "RMSE": val_rmse, "MAE": val_rmse, "latency_ms": bench["latency_ms"], "params": bench["params"],
            "model_state": deepcopy(model.state_dict()),
            "train_params": {"batch_size": int(cur["batch_size"]), "lr": float(cur["lr"]),
                             **{k: v for k, v in cur.items() if k not in ("batch_size", "lr")}},
        }
        rows.append(row)
        _log_trial(logger, i + 1, n_trials, row)
        if val_rmse < best["RMSE"]:
            best = row
    return rows


def _log_trial(logger, trial_idx: int, total: int, result: dict):
    """打印单个 trial 的结果到日志。"""
    if logger is None:
        return
    logger.info(
        "  Trial %d/%d: quick_val_RMSE=%.6f  latency=%.4fms  params=%.0f",
        trial_idx, total,
        result.get("RMSE", -1),
        result.get("latency_ms", -1),
        result.get("params", -1),
    )


# ---------------------------------------------------------------------------
# 3-fold 全量评估（与 Step4 一致）
# ---------------------------------------------------------------------------

def evaluate_on_full_folds(
    best_train_params: dict,
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    meta: dict,
    horizon: int,
    n_folds: int,
    train_frac: float,
    max_epochs: int,
    patience: int,
    device: torch.device,
    logger,
) -> dict:
    """
    用最优参数在完整 3-fold 滚动窗口上评估，得到稳健的 val_RMSE 和 TEST_RMSE。
    与 Step4 的 3-fold 评估逻辑完全一致。
    """
    folds = create_rolling_folds(len(X_train), n_folds=n_folds, train_frac=train_frac)
    logger.info("    完整 3-fold 评估: n_folds=%d  train_frac=%.3f", n_folds, train_frac)

    fold_val_rmses = []
    fold_test_rmses = []
    best_model = None
    best_val = float("inf")

    for fold_idx, (tr_idx, va_idx) in enumerate(folds):
        torch.manual_seed(42)
        np.random.seed(42)

        X_tr = X_train[tr_idx]
        y_tr = y_train[tr_idx]
        X_va = X_train[va_idx]
        y_va = y_train[va_idx]

        val_rmse, model = _train_trial(
            deepcopy(best_train_params), model_name,
            X_tr, y_tr, X_va, y_va,
            meta, horizon, max_epochs, patience, device,
        )
        fold_val_rmses.append(val_rmse)

        test_loader = make_loader(X_test, y_test, batch_size=best_train_params.get("batch_size", 256), shuffle=False)
        test_rmse = float(np.sqrt(eval_loss(model, test_loader, nn.MSELoss(), device)))
        fold_test_rmses.append(test_rmse)

        logger.info("      fold %d: val_RMSE=%.6f  TEST_RMSE=%.6f", fold_idx, val_rmse, test_rmse)

        if val_rmse < best_val:
            best_val = val_rmse
            best_model = deepcopy(model.state_dict())

    avg_val = float(np.mean(fold_val_rmses))
    avg_test = float(np.mean(fold_test_rmses))
    logger.info("    3-fold 平均: val_RMSE=%.6f  TEST_RMSE=%.6f", avg_val, avg_test)

    return {
        "fold_val_rmses": fold_val_rmses,
        "fold_test_rmses": fold_test_rmses,
        "avg_val_RMSE": avg_val,
        "avg_test_RMSE": avg_test,
        "best_model_state": best_model,
    }


# ---------------------------------------------------------------------------
# 评分归一化与排名（trial-level）
# ---------------------------------------------------------------------------

def normalize_scores(rows: list[dict], weights: dict[str, float]) -> list[dict]:
    """多目标综合评分归一化与排序。"""
    keys = ["RMSE", "MAE", "latency_ms", "params"]
    mins = {k: min(r[k] for r in rows) for k in keys}
    maxs = {k: max(r[k] for r in rows) for k in keys}

    def norm(k: str, v: float) -> float:
        if maxs[k] == mins[k]:
            return 0.0
        return (v - mins[k]) / (maxs[k] - mins[k])

    out = []
    for r in rows:
        score = (
            weights.get("rmse", 0.5) * norm("RMSE", r["RMSE"])
            + weights.get("mae", 0.25) * norm("MAE", r["MAE"])
            + weights.get("latency", 0.15) * norm("latency_ms", r["latency_ms"])
            + weights.get("params", 0.10) * norm("params", r["params"])
        )
        item = dict(r)
        item["composite_score"] = float(score)
        out.append(item)
    return sorted(out, key=lambda x: x["composite_score"])


# ---------------------------------------------------------------------------
# 混合搜索主函数（与 Step4 策略一致）
# ---------------------------------------------------------------------------

def run_hybrid_ablation(
    strategy: str,
    model_name: str,
    data: dict,
    meta: dict,
    horizon: int,
    cfg: dict,
    *,
    use_quick_subset: bool = True,
    n_folds: int = 3,
    train_frac: float = 0.7,
    logger=None,
) -> dict[str, Any]:
    """
    运行指定策略的混合搜索，对齐 Step4 策略。

    Step 1: 用 quick subset（训练集后 1/3）做 trial 搜索
    Step 2: 用最优参数在完整 3-fold CV 上评估

    Parameters
    ----------
    strategy : S1 / S2 / S3 / S4 / S5 / S6
    model_name : 模型名（lstm / bilstm / cnn_lstm / cnn_bilstm / patchtst）
    data : 含 X_train_seq, y_train, X_val_seq, y_val, X_test_seq, y_test, y_test_raw
    meta : 含 n_features, lookback
    horizon : 预测步长
    cfg : 配置字典
    use_quick_subset : True → 后 1/3 训练数据做 trial；False → 完整训练数据
    n_folds : 3-fold CV 的 fold 数
    train_frac : 滚动窗口训练集比例
    logger : 日志记录器

    Returns
    -------
    {
        "strategy": str,
        "trials": int,
        "quick_best": dict,        # quick subset 上的最优 trial
        "full_eval": dict,         # 3-fold 完整评估结果
        "all_ranked": list[dict],  # 所有 trial 排序结果
    }
    """
    hs = cfg["hybrid_search"]
    n_trials = hs["n_trials"]
    search_max_epochs = cfg["search_epochs"]["n_epochs_trial"]   # 搜索阶段 epoch
    search_patience = cfg["search_epochs"]["patience_trial"]    # 搜索阶段 patience
    final_max_epochs = cfg["residual_train"]["max_epochs"]      # 3-fold 评估 epoch
    final_patience = cfg["residual_train"]["patience"]
    seed = cfg["residual_train"]["seed"]
    weights = hs["score_weights"]

    X_train = data["X_train_seq"]
    y_train = data["y_train"]
    X_val = data["X_val_seq"]
    y_val = data["y_val"]
    X_test = data["X_test_seq"]
    y_test = data["y_test"]

    # --- Step 1: Quick subset 准备 ---
    if use_quick_subset:
        n_total = len(X_train)
        tr_end = int(n_total * 2 / 3)
        X_quick = X_train[tr_end:]
        y_quick = y_train[tr_end:]
        logger.info("  [Quick] train=%d  quick_train=%d  val=%d", n_total, len(X_quick), len(X_val))
    else:
        X_quick = X_train
        y_quick = y_train
        logger.info("  [Full]  train=%d  val=%d", len(X_train), len(X_val))

    device = get_device()

    # --- Step 1: 运行搜索（trial 阶段用 search_epochs 配置：与 P04 对齐 epoch=12 patience=3）---
    if strategy == "S1":
        rows = run_random_search(n_trials, model_name, X_quick, y_quick, X_val, y_val, meta, horizon, search_max_epochs, search_patience, seed, logger=logger)
    elif strategy == "S2":
        rows = run_optuna_search(n_trials, model_name, X_quick, y_quick, X_val, y_val, meta, horizon, search_max_epochs, search_patience, seed, logger=logger)
    elif strategy == "S3":
        rows = run_afsa_search(n_trials, model_name, X_quick, y_quick, X_val, y_val, meta, horizon, search_max_epochs, search_patience, seed, logger=logger)
    elif strategy == "S4":
        half = max(3, n_trials // 2)
        optuna_rows = run_optuna_search(half, model_name, X_quick, y_quick, X_val, y_val, meta, horizon, search_max_epochs, search_patience, seed, logger=logger)
        init = optuna_rows[0]["train_params"] if optuna_rows else None
        afsa_rows = run_afsa_search(n_trials, model_name, X_quick, y_quick, X_val, y_val, meta, horizon, search_max_epochs, search_patience, seed, init_params=init, logger=logger)
        rows = optuna_rows + afsa_rows
    elif strategy == "S5":
        half = max(3, n_trials // 2)
        afsa_rows = run_afsa_search(half, model_name, X_quick, y_quick, X_val, y_val, meta, horizon, search_max_epochs, search_patience, seed, logger=logger)
        optuna_rows = run_optuna_search(n_trials, model_name, X_quick, y_quick, X_val, y_val, meta, horizon, search_max_epochs, search_patience, seed, logger=logger)
        rows = afsa_rows + optuna_rows
    elif strategy == "S6":
        third = n_trials // 3
        rand_rows = run_random_search(max(2, third), model_name, X_quick, y_quick, X_val, y_val, meta, horizon, search_max_epochs, search_patience, seed, logger=logger)
        optuna_rows = run_optuna_search(max(3, third), model_name, X_quick, y_quick, X_val, y_val, meta, horizon, search_max_epochs, search_patience, seed, logger=logger)
        init = optuna_rows[0]["train_params"] if optuna_rows else None
        afsa_rows = run_afsa_search(max(3, third), model_name, X_quick, y_quick, X_val, y_val, meta, horizon, search_max_epochs, search_patience, seed, init_params=init, logger=logger)
        rows = rand_rows + optuna_rows + afsa_rows
    else:
        raise ValueError(f"未知策略: {strategy}")

    # Trial-level 排名
    ranked = normalize_scores(rows, weights)
    quick_best = ranked[0]
    logger.info("  Quick 最优: RMSE=%.6f  composite=%.6f", quick_best["RMSE"], quick_best["composite_score"])

    # --- Step 2: 完整 3-fold 评估（用最终训练配置：epoch=50 patience=8）---
    logger.info("  开始 3-fold 完整评估（max_epochs=%d  patience=%d）...", final_max_epochs, final_patience)
    full_eval = evaluate_on_full_folds(
        best_train_params=quick_best["train_params"],
        model_name=model_name,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        meta=meta,
        horizon=horizon,
        n_folds=n_folds,
        train_frac=train_frac,
        max_epochs=final_max_epochs,
        patience=final_patience,
        device=device,
        logger=logger,
    )

    return {
        "strategy": strategy,
        "trials": len(rows),
        "quick_best": {k: v for k, v in quick_best.items() if k != "model_state"},
        "full_eval": {
            "avg_val_RMSE": full_eval["avg_val_RMSE"],
            "avg_test_RMSE": full_eval["avg_test_RMSE"],
            "fold_val_rmses": full_eval["fold_val_rmses"],
            "fold_test_rmses": full_eval["fold_test_rmses"],
            "best_model_state": full_eval["best_model_state"],
        },
        "all_ranked": [{k: v for k, v in r.items() if k != "model_state"} for r in ranked],
    }
