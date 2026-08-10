"""EXP-P05 Optuna-AFSA 混合搜索（S1-S6 消融）。"""

from __future__ import annotations

import random
from copy import deepcopy
from typing import Any, Callable

import numpy as np
import optuna
import torch

from experiments.prediction.step4_optuna_hybrid.exp_p04_models import build_model
from experiments.prediction.step4_optuna_hybrid.exp_p04_torch_utils import (
    eval_loss,
    get_device,
    make_loader,
    train_with_early_stop,
)
from experiments.prediction.step5_new_experiments.exp_p05_benchmark import benchmark_forward, count_parameters


# ========== 残差预测辅助函数 ==========

def _compute_residual_targets(y_raw: np.ndarray, y_last: np.ndarray) -> np.ndarray:
    """计算残差目标: residual = y_raw - y_last
    
    支持两种情况:
    - y_raw: (N, horizon), y_last: (N,) -> 输出 (N, horizon)
    - y_raw: (N, 1), y_last: (N, 1) -> 输出 (N, 1)
    """
    if y_last.ndim == 1:
        # y_last: (N,) -> reshape for broadcasting
        return y_raw - y_last[:, np.newaxis]
    else:
        # y_last: (N, 1) -> already 2D, direct subtraction
        return y_raw - y_last


def _fit_residual_scaler(y_res: np.ndarray) -> dict:
    """拟合残差的标准化参数"""
    mean = float(np.mean(y_res))
    std = float(np.std(y_res))
    if std < 1e-6:
        std = 1.0
    return {"mean": mean, "std": std}


def _transform_residual(scaler: dict, y_res: np.ndarray) -> np.ndarray:
    """标准化残差"""
    return (y_res - scaler["mean"]) / scaler["std"]


def _inverse_transform_residual(scaler: dict, y_res_scaled: np.ndarray) -> np.ndarray:
    """反标准化残差"""
    return y_res_scaled * scaler["std"] + scaler["mean"]


def _reconstruct_from_residual(y_last: np.ndarray, delta_pred: np.ndarray) -> np.ndarray:
    """从残差重构预测: y_pred = y_last + delta_pred"""
    if delta_pred.ndim == 2:
        return y_last[:, np.newaxis] + delta_pred
    return y_last + delta_pred


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
    """把通用搜索空间中的 hidden/layers 映射到各模型构造器的实际参数名。"""
    name = model_name.lower().replace("-", "_")
    out = dict(params)
    hidden = out.pop("hidden", None)
    layers = out.pop("layers", None)

    if name == "cnn_bilstm":
        if hidden is not None:
            out["bilstm_hidden"] = hidden
        if layers is not None:
            out["bilstm_layers"] = layers
    return out


def _train_and_score(
    params: dict,
    model_name: str,
    data: dict,
    meta: dict,
    horizon: int,
    max_epochs: int,
    patience: int,
    device: torch.device,
) -> dict[str, float]:
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

    # ========== 残差预测处理 ==========
    # 计算残差目标: y_res = y_raw - y_last
    y_last_train = data["y_last_train"]
    y_last_val = data["y_last_val"]

    y_res_train = _compute_residual_targets(data["y_train_raw"], y_last_train)
    y_res_val = _compute_residual_targets(data["y_val_raw"], y_last_val)

    # 标准化残差
    res_scaler = _fit_residual_scaler(y_res_train)
    y_train_scaled = _transform_residual(res_scaler, y_res_train)
    y_val_scaled = _transform_residual(res_scaler, y_res_val)

    train_loader = make_loader(data["X_train_seq"], y_train_scaled, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(data["X_val_seq"], y_val_scaled, batch_size=batch_size, shuffle=False)

    model, _ = train_with_early_stop(
        model, train_loader, val_loader, lr=lr, max_epochs=max_epochs, patience=patience, device=device
    )

    # 验证集评估（残差空间）
    val_rmse = float(np.sqrt(eval_loss(model, val_loader, torch.nn.MSELoss(), device)))

    # 推理计时
    sample = torch.from_numpy(data["X_test_seq"][:512].astype(np.float32))
    bench = benchmark_forward(model, sample, device=device, warmup_iters=3, repeat_iters=10)

    return {
        "RMSE": val_rmse,
        "MAE": val_rmse,  # 快速搜索阶段以 RMSE 为主，MAE 近似占位
        "latency_ms": bench["ms_per_sample"],
        "params": float(bench["params"]),
        "model_state": deepcopy(model.state_dict()),
        "train_params": {"batch_size": batch_size, "lr": lr, **params},
        "_res_scaler": res_scaler,  # 保存 scaler 用于最终评估
        "_y_last_val": y_last_val,
    }


def normalize_scores(rows: list[dict], weights: dict[str, float]) -> list[dict]:
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


def run_random_search(
    n_trials: int,
    model_name: str,
    data: dict,
    meta: dict,
    horizon: int,
    search_space: dict,
    max_epochs: int,
    patience: int,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    device = get_device()
    rows = []
    for _ in range(n_trials):
        params = _sample_from_space(rng, search_space)
        rows.append(
            _train_and_score(deepcopy(params), model_name, data, meta, horizon, max_epochs, patience, device)
        )
    return rows


def run_optuna_search(
    n_trials: int,
    model_name: str,
    data: dict,
    meta: dict,
    horizon: int,
    search_space: dict,
    max_epochs: int,
    patience: int,
    seed: int,
) -> list[dict]:
    device = get_device()
    rows: list[dict] = []

    def objective(trial: optuna.Trial) -> float:
        params = {k: trial.suggest_categorical(k, v) for k, v in search_space.items()}
        result = _train_and_score(deepcopy(params), model_name, data, meta, horizon, max_epochs, patience, device)
        rows.append(result)
        return result["RMSE"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return rows


def run_afsa_search(
    n_trials: int,
    model_name: str,
    data: dict,
    meta: dict,
    horizon: int,
    search_space: dict,
    max_epochs: int,
    patience: int,
    seed: int,
    init_params: dict | None = None,
) -> list[dict]:
    """简化 AFSA：随机邻域 + 向最优移动。"""
    rng = random.Random(seed)
    device = get_device()
    rows = []
    best = None
    if init_params:
        p0 = deepcopy(init_params)
    else:
        p0 = _sample_from_space(rng, search_space)
    best = _train_and_score(p0, model_name, data, meta, horizon, max_epochs, patience, device)
    rows.append(best)

    for _ in range(n_trials - 1):
        cur = _sample_from_space(rng, search_space)
        # 50% 概率向当前最优参数靠拢（离散空间：逐键替换）
        if best and rng.random() < 0.5:
            merged = deepcopy(cur)
            for k in search_space:
                if rng.random() < 0.5:
                    merged[k] = best["train_params"].get(k, cur[k])
            cur = merged
        rows.append(_train_and_score(cur, model_name, data, meta, horizon, max_epochs, patience, device))
        if rows[-1]["RMSE"] < best["RMSE"]:
            best = rows[-1]
    return rows


def run_hybrid_ablation(
    strategy: str,
    model_name: str,
    data: dict,
    meta: dict,
    horizon: int,
    cfg: dict,
) -> dict[str, Any]:
    """
    strategy: S2..S6
    S2 Optuna, S3 AFSA, S4 Optuna→AFSA, S5 AFSA→Optuna, S6 Optuna+AFSA 混合
    """
    hs = cfg["hybrid_search"]
    n_trials = hs["n_trials"]
    max_epochs = min(15, cfg["residual_train"]["max_epochs"])
    patience = cfg["residual_train"]["patience"]
    seed = cfg["residual_train"]["seed"]
    space = DEFAULT_SEARCH_SPACE
    weights = hs["score_weights"]

    if strategy == "S2":
        rows = run_optuna_search(n_trials, model_name, data, meta, horizon, space, max_epochs, patience, seed)
    elif strategy == "S3":
        rows = run_afsa_search(n_trials, model_name, data, meta, horizon, space, max_epochs, patience, seed)
    elif strategy == "S4":
        optuna_rows = run_optuna_search(max(3, n_trials // 2), model_name, data, meta, horizon, space, max_epochs, patience, seed)
        init = optuna_rows[0]["train_params"] if optuna_rows else None
        rows = run_afsa_search(n_trials, model_name, data, meta, horizon, space, max_epochs, patience, seed, init_params=init)
        rows = optuna_rows + rows
    elif strategy == "S5":
        afsa_rows = run_afsa_search(max(3, n_trials // 2), model_name, data, meta, horizon, space, max_epochs, patience, seed)
        optuna_rows = run_optuna_search(n_trials, model_name, data, meta, horizon, space, max_epochs, patience, seed)
        rows = afsa_rows + optuna_rows
    elif strategy == "S6":
        # Optuna + AFSA 混合 (移除 Random Search)
        optuna_rows = run_optuna_search(max(4, n_trials // 2), model_name, data, meta, horizon, space, max_epochs, patience, seed)
        init = optuna_rows[0]["train_params"] if optuna_rows else None
        afsa_rows = run_afsa_search(max(4, n_trials // 2), model_name, data, meta, horizon, space, max_epochs, patience, seed, init_params=init)
        rows = optuna_rows + afsa_rows
    else:
        raise ValueError(f"未知策略: {strategy}")

    ranked = normalize_scores(rows, weights)
    best = ranked[0]
    return {"strategy": strategy, "trials": len(rows), "best": best, "all_ranked": ranked}


def retrain_with_best_params(
    best_params: dict,
    model_name: str,
    data: dict,
    meta: dict,
    horizon: int,
    cfg: dict,
) -> dict:
    """
    使用最优参数重新训练模型并保存。
    返回训练好的模型指标。
    """
    import time
    from experiments.prediction.step4_optuna_hybrid.exp_p04_torch_utils import (
        eval_loss,
        get_device,
        make_loader,
        train_with_early_stop,
    )
    from experiments.prediction.step5_new_experiments.exp_p05_common import (
        MODELS_DIR,
        PRED_DIR,
        compute_all_metrics,
        set_seed,
        save_predictions,
    )
    from experiments.prediction.step5_new_experiments.exp_p05_residual import (
        compute_residual_targets,
        fit_residual_scaler,
        inverse_transform_residual,
        reconstruct_from_residual,
        transform_residual,
    )

    rt = cfg["residual_train"]
    set_seed(rt["seed"])
    device = get_device()

    # 准备残差数据
    y_last_train = data["y_last_train"]
    y_last_val = data["y_last_val"]
    y_last_test = data["y_last_test"]

    y_res_train = compute_residual_targets(data["y_train_raw"], y_last_train)
    y_res_val = compute_residual_targets(data["y_val_raw"], y_last_val)
    res_scaler = fit_residual_scaler(y_res_train)

    y_train = transform_residual(res_scaler, y_res_train)
    y_val = transform_residual(res_scaler, y_res_val)

    # 构建模型，使用最优参数
    train_params = best_params.get("train_params", {})
    batch_size = int(train_params.get("batch_size", rt["batch_size"]))
    lr = float(train_params.get("lr", rt["lr"]))

    # 处理模型特定参数
    normalized = _normalize_model_params(dict(train_params), model_name)

    model = build_model(
        model_name,
        n_features=meta["n_features"],
        seq_len=meta["lookback"],
        horizon=horizon,
        **normalized,
    ).to(device)

    train_loader = make_loader(data["X_train_seq"], y_train, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(data["X_val_seq"], y_val, batch_size=batch_size, shuffle=False)

    # 训练
    t0 = time.time()
    model, history = train_with_early_stop(
        model,
        train_loader,
        val_loader,
        lr=lr,
        max_epochs=rt["max_epochs"],
        patience=rt["patience"],
        device=device,
    )
    train_time = time.time() - t0

    # 保存模型
    model_path = MODELS_DIR / f"h{horizon}" / f"{model_name}_hybrid_best.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)

    # 预测
    from experiments.prediction.step4_optuna_hybrid.exp_p04_torch_utils import predict

    delta_scaled = predict(model, data["X_test_seq"], device, batch_size=batch_size)
    delta_pred = inverse_transform_residual(res_scaler, delta_scaled)
    y_pred = reconstruct_from_residual(y_last_test, delta_pred)

    # 计算指标
    y_true = data["y_test_raw"]
    if horizon == 1:
        y_true_eval = y_true[:, 0]
        y_pred_eval = y_pred[:, 0]
    else:
        y_true_eval = y_true.ravel()
        y_pred_eval = y_pred.ravel()

    metrics = compute_all_metrics(y_true_eval, y_pred_eval)
    metrics["training_time_sec"] = train_time

    # 保存预测结果
    save_predictions(horizon, f"{model_name}_hybrid_best", y_true_eval, y_pred_eval)

    return {
        "metrics": metrics,
        "model_path": str(model_path),
        "train_params": train_params,
        "train_history": history,
    }
