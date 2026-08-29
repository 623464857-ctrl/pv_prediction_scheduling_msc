"""EXP-P04 Optuna-AFSA 混合超参搜索（S2-S6，不含 S1 Random）。"""

from __future__ import annotations

import random
import time
from copy import deepcopy
from typing import Any

import numpy as np
import optuna
import torch
import torch.nn as nn

from experiments.prediction.step4_optuna_hybrid.exp_p04_models import build_model
from experiments.prediction.step4_optuna_hybrid.exp_p04_torch_utils import (
    eval_loss,
    get_device,
    make_loader,
    train_with_early_stop,
)

STRATEGIES = ["S2", "S3", "S4", "S5", "S6"]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def benchmark_forward(
    model: nn.Module,
    sample_input: torch.Tensor,
    *,
    warmup_iters: int = 3,
    repeat_iters: int = 10,
    device: torch.device | None = None,
) -> dict[str, float]:
    device = device or torch.device("cpu")
    model = model.to(device)
    model.eval()
    x = sample_input.to(device)

    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()

        times = []
        n_samples = x.shape[0]
        for _ in range(repeat_iters):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    mean_sec = float(np.mean(times))
    return {
        "ms_per_sample": mean_sec / max(n_samples, 1) * 1000.0,
        "params": float(count_parameters(model)),
    }


def _sample_from_space(rng: random.Random, space: dict) -> dict:
    return {k: rng.choice(v) for k, v in space.items()}


def _train_and_score(
    params: dict,
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_bench: np.ndarray,
    seq_len: int,
    n_features: int,
    horizon: int,
    max_epochs: int,
    patience: int,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    trial_params = deepcopy(params)
    batch_size = int(trial_params.pop("batch_size"))
    lr = float(trial_params.pop("lr"))

    model = build_model(
        model_name,
        n_features=n_features,
        seq_len=seq_len,
        horizon=horizon,
        **trial_params,
    ).to(device)

    train_loader = make_loader(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, batch_size=batch_size, shuffle=False)
    model, _ = train_with_early_stop(
        model, train_loader, val_loader,
        lr=lr, max_epochs=max_epochs, patience=patience, device=device,
    )
    val_loss = eval_loss(model, val_loader, nn.MSELoss(), device)
    val_rmse = float(np.sqrt(val_loss))

    bench_n = min(512, len(X_bench))
    sample = torch.from_numpy(X_bench[:bench_n].astype(np.float32))
    bench = benchmark_forward(model, sample, device=device)

    return {
        "RMSE": val_rmse,
        "MAE": val_rmse,
        "latency_ms": bench["ms_per_sample"],
        "params": bench["params"],
        "train_params": {"batch_size": batch_size, "lr": lr, **trial_params},
    }


def normalize_scores(rows: list[dict], weights: dict[str, float]) -> list[dict]:
    keys = ["RMSE", "MAE", "latency_ms", "params"]
    mins = {k: min(r[k] for r in rows) for k in keys}
    maxs = {k: max(r[k] for r in rows) for k in keys}

    def norm(key: str, value: float) -> float:
        if maxs[key] == mins[key]:
            return 0.0
        return (value - mins[key]) / (maxs[key] - mins[key])

    ranked = []
    for row in rows:
        score = (
            weights.get("rmse", 0.5) * norm("RMSE", row["RMSE"])
            + weights.get("mae", 0.25) * norm("MAE", row["MAE"])
            + weights.get("latency", 0.15) * norm("latency_ms", row["latency_ms"])
            + weights.get("params", 0.10) * norm("params", row["params"])
        )
        item = dict(row)
        item["composite_score"] = float(score)
        ranked.append(item)
    return sorted(ranked, key=lambda x: x["composite_score"])


def run_random_search(*args, **kwargs):
    raise NotImplementedError("S1 Random Search 已从 P04 混合搜索中移除")


def run_optuna_search(
    n_trials: int,
    model_name: str,
    search_space: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_bench: np.ndarray,
    seq_len: int,
    n_features: int,
    horizon: int,
    max_epochs: int,
    patience: int,
    seed: int,
) -> list[dict]:
    device = get_device()
    rows: list[dict] = []

    def objective(trial: optuna.Trial) -> float:
        params = {}
        for key, space in search_space.items():
            vals_str = [str(v) for v in space]
            chosen = trial.suggest_categorical(key, vals_str)
            try:
                params[key] = int(chosen)
            except ValueError:
                try:
                    params[key] = float(chosen)
                except ValueError:
                    params[key] = chosen
        result = _train_and_score(
            params, model_name, X_train, y_train, X_val, y_val, X_bench,
            seq_len, n_features, horizon, max_epochs, patience, device, seed,
        )
        rows.append(result)
        return result["RMSE"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return rows


def run_afsa_search(
    n_trials: int,
    model_name: str,
    search_space: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_bench: np.ndarray,
    seq_len: int,
    n_features: int,
    horizon: int,
    max_epochs: int,
    patience: int,
    seed: int,
    init_params: dict | None = None,
) -> list[dict]:
    """简化 AFSA：随机邻域 + 向当前最优参数靠拢。"""
    rng = random.Random(seed)
    device = get_device()
    rows: list[dict] = []

    start = init_params if init_params else _sample_from_space(rng, search_space)
    best = _train_and_score(
        start, model_name, X_train, y_train, X_val, y_val, X_bench,
        seq_len, n_features, horizon, max_epochs, patience, device, seed,
    )
    rows.append(best)

    for _ in range(max(0, n_trials - 1)):
        cur = _sample_from_space(rng, search_space)
        if rng.random() < 0.5:
            merged = deepcopy(cur)
            for key in search_space:
                if rng.random() < 0.5:
                    merged[key] = best["train_params"].get(key, cur[key])
            cur = merged
        result = _train_and_score(
            cur, model_name, X_train, y_train, X_val, y_val, X_bench,
            seq_len, n_features, horizon, max_epochs, patience, device, seed,
        )
        rows.append(result)
        if result["RMSE"] < best["RMSE"]:
            best = result
    return rows


def run_hybrid_ablation(
    strategy: str,
    model_name: str,
    search_space: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_bench: np.ndarray,
    seq_len: int,
    n_features: int,
    horizon: int,
    hybrid_cfg: dict,
    n_epochs: int,
    patience: int,
    seed: int,
) -> dict[str, Any]:
    """
    S2 Optuna | S3 AFSA | S4 Optuna→AFSA | S5 AFSA→Optuna | S6 Optuna+AFSA
    S1 Random 已移除；S6 不再包含 Random 阶段。
    """
    if strategy == "S1":
        raise ValueError("S1 Random Search 已从 P04 移除，请使用 S2-S6")

    n_trials = hybrid_cfg["n_trials"]
    weights = hybrid_cfg["score_weights"]
    common = dict(
        model_name=model_name,
        search_space=search_space,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_bench=X_bench,
        seq_len=seq_len,
        n_features=n_features,
        horizon=horizon,
        max_epochs=n_epochs,
        patience=patience,
        seed=seed,
    )

    if strategy == "S2":
        rows = run_optuna_search(n_trials, **common)
    elif strategy == "S3":
        rows = run_afsa_search(n_trials, **common)
    elif strategy == "S4":
        optuna_rows = run_optuna_search(max(3, n_trials // 2), **common)
        init = min(optuna_rows, key=lambda r: r["RMSE"])["train_params"] if optuna_rows else None
        afsa_rows = run_afsa_search(n_trials, init_params=init, **common)
        rows = optuna_rows + afsa_rows
    elif strategy == "S5":
        afsa_rows = run_afsa_search(max(3, n_trials // 2), **common)
        optuna_rows = run_optuna_search(n_trials, **common)
        rows = afsa_rows + optuna_rows
    elif strategy == "S6":
        optuna_rows = run_optuna_search(max(3, n_trials // 2), **common)
        init = min(optuna_rows, key=lambda r: r["RMSE"])["train_params"] if optuna_rows else None
        afsa_rows = run_afsa_search(max(3, n_trials - len(optuna_rows)), init_params=init, **common)
        rows = optuna_rows + afsa_rows
    else:
        raise ValueError(f"未知策略: {strategy}，支持 {STRATEGIES}")

    ranked = normalize_scores(rows, weights)
    return {"strategy": strategy, "trials": len(rows), "best": ranked[0], "all_ranked": ranked}


def run_all_strategies(
    model_name: str,
    search_space: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_bench: np.ndarray,
    seq_len: int,
    n_features: int,
    horizon: int,
    hybrid_cfg: dict,
    n_epochs: int,
    patience: int,
    seed: int,
    logger,
) -> tuple[dict[str, dict], dict]:
    """运行 S2-S6 全部策略，返回 (ablation_results, global_best)。"""
    strategies = hybrid_cfg.get("strategies", STRATEGIES)
    ablation: dict[str, dict] = {}
    all_bests: list[dict] = []

    for strategy in strategies:
        if strategy == "S1":
            logger.warning("跳过已移除策略 S1")
            continue
        logger.info("混合搜索策略 %s ...", strategy)
        result = run_hybrid_ablation(
            strategy, model_name, search_space,
            X_train, y_train, X_val, y_val, X_bench,
            seq_len, n_features, horizon, hybrid_cfg, n_epochs, patience, seed,
        )
        best = {k: v for k, v in result["best"].items() if k != "model_state"}
        ablation[strategy] = {"strategy": strategy, "trials": result["trials"], "best": best}
        tagged = dict(best)
        tagged["strategy"] = strategy
        all_bests.append(tagged)
        logger.info("  %s  best_RMSE=%.4f  composite=%.4f  params=%s",
                    strategy, best["RMSE"], best["composite_score"], best["train_params"])

    global_best = min(all_bests, key=lambda x: x["composite_score"])
    return ablation, global_best


def train_params_to_best_params(train_params: dict) -> dict:
    """转换为 train_final 可读取的 best_params 格式。"""
    return {k: str(v) if not isinstance(v, str) else v for k, v in train_params.items()}
