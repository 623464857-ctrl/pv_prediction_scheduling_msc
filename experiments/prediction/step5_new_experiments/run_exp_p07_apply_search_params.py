"""
EXP-P07: H16 cnn_lstm 快速搜索 + 残差预测训练（S1-S6）。
python experiments/prediction/step5_new_experiments/run_exp_p07_apply_search_params.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.model_selection import KFold

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.prediction.step4_optuna_hybrid.exp_p04_models import build_model
from experiments.prediction.step4_optuna_hybrid.exp_p04_torch_utils import (
    eval_loss,
    get_device,
    make_loader,
    predict,
    train_with_early_stop,
)
from experiments.prediction.step5_new_experiments.exp_p05_common import (
    METRICS_DIR,
    MODELS_DIR,
    PRED_DIR,
    compute_all_metrics,
    ensure_dirs,
    load_config,
    load_meta,
    load_samples,
    save_predictions,
    set_seed,
    setup_logger,
)
from experiments.prediction.step5_new_experiments.exp_p05_residual import (
    compute_residual_targets,
    fit_residual_scaler,
    inverse_transform_residual,
    reconstruct_from_residual,
    save_residual_scaler,
    transform_residual,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

STRATEGIES = ["S1", "S2", "S3", "S4", "S5", "S6"]
N_FOLDS = 3
MAX_EPOCHS_SEARCH = 15
MAX_EPOCHS_FINAL = 50
PATIENCE = 10
SEED = 42


SEARCH_SPACE = {
    "hidden": [32, 64, 128],
    "layers": [1, 2],
    "dropout": [0.1, 0.2, 0.3],
    "lr": [0.0005, 0.001, 0.002],
    "batch_size": [128, 256],
}

SCORE_WEIGHTS = {"rmse": 0.5, "mae": 0.25, "latency": 0.15, "params": 0.10}


def _normalize(params: dict) -> dict:
    """通用搜索空间 → cnn_lstm 构造参数，同时剔除 lr/batch_size。"""
    out = dict(params)
    hidden = out.pop("hidden", None)
    layers = out.pop("layers", None)
    out.pop("lr", None)
    out.pop("batch_size", None)
    if hidden is not None:
        out["lstm_hidden"] = hidden
    if layers is not None:
        out["lstm_layers"] = layers
    return out


def _train_residual(
    params: dict,
    model_name: str,
    X_tr: np.ndarray,
    y_res_tr: np.ndarray,
    X_va: np.ndarray,
    y_res_va: np.ndarray,
    X_te: np.ndarray,
    res_scaler,
    y_last_te: np.ndarray,
    y_te_raw: np.ndarray,
    horizon: int,
    max_epochs: int,
    device: torch.device,
) -> dict:
    """训练一个残差模型，返回 val RMSE + test 指标。"""
    batch_size = int(params.pop("batch_size"))
    lr = float(params.pop("lr"))
    model_kwargs = _normalize(params)
    n_features = X_tr.shape[2]
    seq_len = X_tr.shape[1]

    model = build_model(
        model_name,
        n_features=n_features,
        seq_len=seq_len,
        horizon=horizon,
        **model_kwargs,
    ).to(device)

    train_loader = make_loader(X_tr, y_res_tr, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(X_va, y_res_va, batch_size=batch_size, shuffle=False)

    model, _ = train_with_early_stop(
        model, train_loader, val_loader, lr=lr, max_epochs=max_epochs, patience=PATIENCE, device=device,
    )

    val_rmse = float(np.sqrt(eval_loss(model, val_loader, torch.nn.MSELoss(), device)))

    delta_scaled = predict(model, X_te, device, batch_size=batch_size)
    delta_pred = inverse_transform_residual(res_scaler, delta_scaled)
    y_pred = reconstruct_from_residual(y_last_te, delta_pred)

    if horizon == 1:
        y_true_eval = y_te_raw[:, 0]
        y_pred_eval = y_pred[:, 0]
    else:
        y_true_eval = y_te_raw.ravel()
        y_pred_eval = y_pred.ravel()

    m = compute_all_metrics(y_true_eval, y_pred_eval)
    m["val_rmse"] = val_rmse
    return {"val_rmse": val_rmse, "test_metrics": m, "model_state": deepcopy(model.state_dict())}


def _normalize_scores(rows: list[dict]) -> list[dict]:
    keys = ["val_rmse", "val_rmse", "latency_ms", "params"]
    mins = {k: min(r[k] for r in rows) for k in keys}
    maxs = {k: max(r[k] for r in rows) for k in keys}

    def norm(k: str, v: float) -> float:
        if maxs[k] == mins[k]:
            return 0.0
        return (v - mins[k]) / (maxs[k] - mins[k])

    out = []
    for r in rows:
        score = (
            SCORE_WEIGHTS["rmse"] * norm("val_rmse", r["val_rmse"])
            + SCORE_WEIGHTS["mae"] * norm("val_rmse", r["val_rmse"])
            + SCORE_WEIGHTS["latency"] * norm("latency_ms", r["latency_ms"])
            + SCORE_WEIGHTS["params"] * norm("params", float(r["params"]))
        )
        item = dict(r)
        item["composite_score"] = float(score)
        out.append(item)
    return sorted(out, key=lambda x: x["composite_score"])


def run_random_search(n_trials: int, data: dict, meta: dict, horizon: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    device = get_device()
    rows = []
    for _ in range(n_trials):
        params = {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}
        result = _search_trial(deepcopy(params), data, meta, horizon, MAX_EPOCHS_SEARCH, device)
        rows.append(result)
    return rows


def run_optuna_search(n_trials: int, data: dict, meta: dict, horizon: int, seed: int) -> list[dict]:
    device = get_device()
    rows: list[dict] = []

    def objective(trial: optuna.Trial) -> float:
        params = {k: trial.suggest_categorical(k, v) for k, v in SEARCH_SPACE.items()}
        result = _search_trial(deepcopy(params), data, meta, horizon, MAX_EPOCHS_SEARCH, device)
        rows.append(result)
        return result["val_rmse"]

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return rows


def run_afsa_search(
    n_trials: int, data: dict, meta: dict, horizon: int, seed: int, init_params: dict | None = None,
) -> list[dict]:
    rng = random.Random(seed)
    device = get_device()
    rows = []
    best = None

    def sample():
        return {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}

    p0 = deepcopy(init_params) if init_params else sample()
    best = _search_trial(p0, data, meta, horizon, MAX_EPOCHS_SEARCH, device)
    rows.append(best)

    for _ in range(n_trials - 1):
        cur = sample()
        if best and rng.random() < 0.5:
            merged = deepcopy(cur)
            for k in SEARCH_SPACE:
                if rng.random() < 0.5:
                    merged[k] = best["train_params"].get(k, cur[k])
            cur = merged
        result = _search_trial(cur, data, meta, horizon, MAX_EPOCHS_SEARCH, device)
        rows.append(result)
        if result["val_rmse"] < best["val_rmse"]:
            best = result
    return rows


def _search_trial(
    params: dict,
    data: dict,
    meta: dict,
    horizon: int,
    max_epochs: int,
    device: torch.device,
) -> dict:
    X_tr = data["X_tr_res"]
    y_tr = data["y_tr_res"]
    X_va = data["X_va_res"]
    y_va = data["y_va_res"]
    X_te = data["X_te_seq"]
    res_scaler = data["res_scaler"]
    y_last_te = data["y_last_test"]
    y_te_raw = data["y_te_raw"]

    from experiments.prediction.step5_new_experiments.exp_p05_benchmark import benchmark_forward, count_parameters

    batch_size = int(params["batch_size"])
    lr = float(params["lr"])
    model_kwargs = _normalize(params)
    n_features = meta["n_features"]
    seq_len = meta["lookback"]

    model = build_model(
        "cnn_lstm",
        n_features=n_features,
        seq_len=seq_len,
        horizon=horizon,
        **model_kwargs,
    ).to(device)

    train_loader = make_loader(X_tr, y_tr, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(X_va, y_va, batch_size=batch_size, shuffle=False)

    model, _ = train_with_early_stop(
        model, train_loader, val_loader, lr=lr, max_epochs=max_epochs, patience=PATIENCE, device=device,
    )

    val_rmse = float(np.sqrt(eval_loss(model, val_loader, torch.nn.MSELoss(), device)))

    delta_scaled = predict(model, X_te, device, batch_size=batch_size)
    delta_pred = inverse_transform_residual(res_scaler, delta_scaled)
    y_pred = reconstruct_from_residual(y_last_te, delta_pred)

    if horizon == 1:
        y_true_eval = y_te_raw[:, 0]
        y_pred_eval = y_pred[:, 0]
    else:
        y_true_eval = y_te_raw.ravel()
        y_pred_eval = y_pred.ravel()

    m = compute_all_metrics(y_true_eval, y_pred_eval)

    n_params = count_parameters(model)
    sample = torch.from_numpy(X_te[:512].astype(np.float32)).to(device)
    import time
    times = []
    with torch.no_grad():
        for _ in range(5):
            _ = model(sample[:32])
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        for _ in range(10):
            _ = model(sample[:32])
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t1 = time.perf_counter()
    ms_per_sample = (t1 - t0) / (10 * 32) * 1000

    return {
        "val_rmse": val_rmse,
        "test_rmse": m["RMSE"],
        "test_mae": m["MAE"],
        "latency_ms": ms_per_sample,
        "params": float(n_params),
        "train_params": dict(params),
    }


def run_hybrid_residual_search(strategy: str, data: dict, meta: dict, horizon: int, n_trials: int, seed: int) -> dict:
    if strategy == "S1":
        rows = run_random_search(n_trials, data, meta, horizon, seed)
    elif strategy == "S2":
        rows = run_optuna_search(n_trials, data, meta, horizon, seed)
    elif strategy == "S3":
        rows = run_afsa_search(n_trials, data, meta, horizon, seed)
    elif strategy == "S4":
        optuna_rows = run_optuna_search(max(3, n_trials // 2), data, meta, horizon, seed)
        init = optuna_rows[0]["train_params"] if optuna_rows else None
        afsa_rows = run_afsa_search(n_trials, data, meta, horizon, seed, init_params=init)
        rows = optuna_rows + afsa_rows
    elif strategy == "S5":
        afsa_rows = run_afsa_search(max(3, n_trials // 2), data, meta, horizon, seed)
        optuna_rows = run_optuna_search(n_trials, data, meta, horizon, seed)
        rows = afsa_rows + optuna_rows
    elif strategy == "S6":
        rand_rows = run_random_search(max(2, n_trials // 3), data, meta, horizon, seed)
        optuna_rows = run_optuna_search(max(3, n_trials // 3), data, meta, horizon, seed)
        init = optuna_rows[0]["train_params"] if optuna_rows else None
        afsa_rows = run_afsa_search(max(3, n_trials // 3), data, meta, horizon, seed, init_params=init)
        rows = rand_rows + optuna_rows + afsa_rows
    else:
        raise ValueError(f"未知策略: {strategy}")

    ranked = _normalize_scores(rows)
    best = ranked[0]
    return {"strategy": strategy, "trials": len(rows), "best": best, "all_ranked": ranked}


def prepare_residual_data(horizon: int) -> tuple[dict, dict]:
    """准备残差模式的数据字典（train+val 合并为搜索数据，test 为评估数据）。"""
    samples = load_samples(horizon, use_step5=True)
    meta = load_meta(horizon, use_step5=True)

    X_all = np.concatenate([samples["X_train_seq"], samples["X_val_seq"]], axis=0)
    y_last_all = np.concatenate([samples["y_last_train"], samples["y_last_val"]], axis=0)
    y_raw_all = np.concatenate([samples["y_train_raw"], samples["y_val_raw"]], axis=0)

    y_res_all = compute_residual_targets(y_raw_all, y_last_all)
    res_scaler = fit_residual_scaler(y_res_all)
    y_res_scaled = transform_residual(res_scaler, y_res_all)

    n_train = len(samples["X_train_seq"])
    X_tr = X_all[:n_train]
    X_va = X_all[n_train:]
    y_tr = y_res_scaled[:n_train]
    y_va = y_res_scaled[n_train:]

    data = {
        "X_tr_res": X_tr,
        "X_va_res": X_va,
        "y_tr_res": y_tr,
        "y_va_res": y_va,
        "X_te_seq": samples["X_test_seq"],
        "res_scaler": res_scaler,
        "y_last_test": samples["y_last_test"],
        "y_te_raw": samples["y_test_raw"],
    }
    return data, meta


def kfold_final_train(
    params: dict,
    data: dict,
    meta: dict,
    horizon: int,
    strategy: str,
    logger,
) -> dict:
    """用搜索得到的最优参数进行 3-fold CV 残差训练。"""
    device = get_device()
    X_all = np.concatenate([data["X_tr_res"], data["X_va_res"]], axis=0)
    y_all = np.concatenate([data["y_tr_res"], data["y_va_res"]], axis=0)

    batch_size = int(params["batch_size"])
    lr = float(params["lr"])
    hidden = params.get("hidden", 64)
    layers = params.get("layers", 2)
    dropout = params.get("dropout", 0.2)

    fold_metrics = []
    fold_states = []

    logger.info("[%s] 3-fold CV 最终训练  batch=%d  lr=%.4f  hidden=%d  layers=%d  dropout=%.1f",
                strategy, batch_size, lr, hidden, layers, dropout)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(X_all)):
        set_seed(SEED + fold_idx)
        X_tr, X_va = X_all[tr_idx], X_all[va_idx]
        y_tr, y_va = y_all[tr_idx], y_all[va_idx]

        train_loader = make_loader(X_tr, y_tr, batch_size=batch_size, shuffle=True)
        val_loader = make_loader(X_va, y_va, batch_size=batch_size, shuffle=False)

        model = build_model(
            "cnn_lstm",
            n_features=meta["n_features"],
            seq_len=meta["lookback"],
            horizon=horizon,
            conv_channels=hidden,
            lstm_hidden=hidden,
            lstm_layers=layers,
            dropout=dropout,
        ).to(device)

        model, history = train_with_early_stop(
            model, train_loader, val_loader,
            lr=lr, max_epochs=MAX_EPOCHS_FINAL, patience=PATIENCE, device=device,
        )
        fold_states.append({k: v.cpu().numpy() for k, v in model.state_dict().items()})

        delta_scaled = predict(model, data["X_te_seq"], device, batch_size=batch_size)
        delta_pred = inverse_transform_residual(data["res_scaler"], delta_scaled)
        y_pred = reconstruct_from_residual(data["y_last_test"], delta_pred)

        if horizon == 1:
            y_true_eval = data["y_te_raw"][:, 0]
            y_pred_eval = y_pred[:, 0]
        else:
            y_true_eval = data["y_te_raw"].ravel()
            y_pred_eval = y_pred.ravel()

        m = compute_all_metrics(y_true_eval, y_pred_eval)
        m["fold"] = fold_idx
        m["best_epoch"] = len(history) - 1
        fold_metrics.append(m)
        logger.info("  Fold %d: RMSE=%.4f  MAE=%.4f  R2=%.4f",
                    fold_idx, m["RMSE"], m["MAE"], m["R2"])

    fold_df = pd.DataFrame(fold_metrics)
    mean_metrics = fold_df.drop(columns=["fold", "best_epoch"]).mean().to_dict()
    std_metrics = fold_df.drop(columns=["fold", "best_epoch"]).std().to_dict()
    logger.info("[%s] CV均值: RMSE=%.4f ± %.4f  MAE=%.4f ± %.4f",
                strategy, mean_metrics["RMSE"], std_metrics["RMSE"],
                mean_metrics["MAE"], std_metrics["MAE"])

    best_fold_idx = int(fold_df.loc[fold_df["RMSE"].idxmin(), "fold"])
    best_state = fold_states[best_fold_idx]

    model_path = MODELS_DIR / "h16" / f"cnn_lstm_residual_{strategy}.pt"
    torch.save(best_state, model_path)
    scaler_path = MODELS_DIR.parent / "samples" / "h16" / f"residual_scaler_params_{strategy}.json"
    save_residual_scaler(data["res_scaler"], scaler_path)

    logger.info("[%s] 最优模型已保存 (fold %d): %s", strategy, best_fold_idx, model_path.relative_to(PROJECT_ROOT))
    return {"cv_mean": mean_metrics, "cv_std": std_metrics, "best_fold": best_fold_idx, "fold_metrics": fold_metrics}


def main():
    parser = argparse.ArgumentParser(description="EXP-P07 快速搜索 + 残差预测")
    parser.add_argument("--strategy", type=str, choices=STRATEGIES, help="单策略，默认全部 S1-S6")
    parser.add_argument("--all", action="store_true", help="运行 S1-S6 全部")
    parser.add_argument("--n-trials", type=int, default=20, help="每个策略的搜索次数（默认20）")
    args = parser.parse_args()

    cfg = load_config()
    ensure_dirs(MODELS_DIR / "h16", PRED_DIR / "h16", METRICS_DIR / "h16")
    logger = setup_logger("exp_p07", "EXP-P07_search_residual.log")
    logger.info("=== EXP-P07: H16 cnn_lstm 快速搜索 + 残差预测 (S1-S6) ===")

    data, meta = prepare_residual_data(horizon=16)
    search_seed = cfg["residual_train"]["seed"]

    strategies = [args.strategy] if args.strategy else STRATEGIES
    all_results = {}

    for s in strategies:
        logger.info("========== [%s] 搜索阶段 ==========", s)
        search_result = run_hybrid_residual_search(s, data, meta, horizon=16, n_trials=args.n_trials, seed=search_seed)
        best_params = search_result["best"]["train_params"]

        search_summary = {
            "strategy": s,
            "n_trials": search_result["trials"],
            "search_val_rmse": search_result["best"]["val_rmse"],
            "search_test_rmse": search_result["best"]["test_rmse"],
            "composite_score": search_result["best"]["composite_score"],
            "best_params": best_params,
        }
        logger.info("[%s] 搜索最优: val_rmse=%.4f  test_rmse=%.4f  composite=%.4f",
                     s, search_summary["search_val_rmse"],
                     search_summary["search_test_rmse"],
                     search_summary["composite_score"])

        logger.info("========== [%s] 3-fold CV 最终训练 ==========", s)
        cv_result = kfold_final_train(best_params, data, meta, horizon=16, strategy=s, logger=logger)

        all_results[s] = {**search_summary, "cv_mean": cv_result["cv_mean"], "cv_std": cv_result["cv_std"]}

    out_path = METRICS_DIR / "h16" / "search_residual_cv_metrics.json"
    out_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("结果已保存: %s", out_path.relative_to(PROJECT_ROOT))

    sorted_results = sorted(all_results.items(), key=lambda kv: kv[1]["cv_mean"]["RMSE"])
    logger.info("========== CV RMSE 最终排序 ==========")
    for rank, (s, r) in enumerate(sorted_results, 1):
        m = r["cv_mean"]
        std = r["cv_std"]
        logger.info("  %d. %s  CV_RMSE=%.4f ± %.4f  search_RMSE=%.4f",
                    rank, s, m["RMSE"], std["RMSE"], r["search_test_rmse"])


if __name__ == "__main__":
    main()
