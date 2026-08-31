"""
python -m experiments.prediction.step4_optuna_hybrid.run_exp_p04_optuna --horizon 1
python -m experiments.prediction.step4_optuna_hybrid.run_exp_p04_optuna --horizon 4
python -m experiments.prediction.step4_optuna_hybrid.run_exp_p04_optuna --horizon 16
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.prediction.step4_optuna_hybrid.exp_p04_common import (
    METRICS_DIR,
    MODELS_DIR,
    PROJECT_ROOT,
    compute_all_metrics,
    load_config,
    load_sample_dir,
    load_y_scaler_from_json,
    setup_logger,
)
from experiments.prediction.step4_optuna_hybrid.exp_p04_step_audit import (
    record_step_failure,
    record_step_result,
)
from experiments.prediction.step4_optuna_hybrid.exp_p04_cv import create_rolling_folds
from experiments.prediction.step4_optuna_hybrid.exp_p04_hybrid_search import (
    run_all_strategies,
    train_params_to_best_params,
)
from experiments.prediction.step4_optuna_hybrid.exp_p04_models import build_model
from experiments.prediction.step4_optuna_hybrid.exp_p04_torch_utils import (
    eval_loss,
    get_device,
    make_loader,
    predict,
    train_with_early_stop,
)


def _convert_params(params: dict, search_space: dict) -> dict:
    out = {}
    for name, space in search_space.items():
        if name in ("lr", "batch_size"):
            continue
        vals_str = [str(v) for v in space]
        chosen = str(params.get(name, params[name]))
        if chosen in vals_str:
            try:
                out[name] = int(chosen)
            except ValueError:
                try:
                    out[name] = float(chosen)
                except ValueError:
                    out[name] = chosen
        else:
            out[name] = params[name]
    return out


def run_hybrid_search_for_model(model_name, horizon_cfg, base_cfg, logger):
    """对单个模型执行 Optuna-AFSA 混合消融搜索（S2-S6），最优参数做 3-fold 评估。"""
    horizon = horizon_cfg["horizon"]
    hdir = load_sample_dir(horizon)
    metrics_h = METRICS_DIR / f"h{horizon}"
    models_h = MODELS_DIR / f"h{horizon}"
    for d in (metrics_h, models_h):
        d.mkdir(parents=True, exist_ok=True)

    logger.info("-" * 50)
    logger.info("开始 Optuna-AFSA 混合搜索: model=%s  horizon=%s", model_name, horizon)

    X_train = np.load(hdir / "X_train_seq.npy")
    y_residual_train = np.load(hdir / "y_train.npy")
    X_val = np.load(hdir / "X_val_seq.npy")
    y_residual_val = np.load(hdir / "y_val.npy")
    y_anchor_val = np.load(hdir / "y_anchor_val.npy")
    _, _, n_features = X_train.shape
    meta = json.loads((hdir / "meta.json").read_text(encoding="utf-8"))
    seq_len = meta["lookback"]

    device = get_device()
    logger.info("设备: %s  |  train=%d  val=%d  n_features=%d  seq_len=%d  horizon=%d",
                device, len(X_train), len(X_val), n_features, seq_len, horizon)

    search_space = horizon_cfg["model_search_space"][model_name]
    hybrid_cfg = base_cfg["hybrid_search"]
    n_epochs = horizon_cfg["n_epochs_trial"]
    patience = horizon_cfg["patience_trial"]
    seed = base_cfg["reproduce_seeds"][0]

    n_total = len(X_train)
    tr_end = int(n_total * 2 / 3)
    X_quick = X_train[tr_end:]
    y_quick_residual = y_residual_train[tr_end:]
    logger.info("快速搜索: fold split at %d  quick_train=%d  strategies=%s",
                tr_end, len(X_quick), hybrid_cfg.get("strategies", ["S2", "S3", "S4", "S5", "S6"]))

    ablation, global_best = run_all_strategies(
        model_name=model_name,
        search_space=search_space,
        X_train=X_quick,
        y_train=y_quick_residual,
        X_val=X_val,
        y_val=y_residual_val,
        X_bench=X_val,
        seq_len=seq_len,
        n_features=n_features,
        horizon=horizon,
        hybrid_cfg=hybrid_cfg,
        n_epochs=n_epochs,
        patience=patience,
        seed=seed,
        logger=logger,
    )

    ablation_path = metrics_h / "hybrid_search_ablation.json"
    with open(ablation_path, "w", encoding="utf-8") as f:
        json.dump(ablation, f, indent=2, ensure_ascii=False)
    logger.info("混合搜索消融结果已保存: %s", ablation_path.name)
    logger.info("全局最优策略=%s  RMSE=%.4f  composite=%.4f",
                global_best["strategy"], global_best["RMSE"], global_best["composite_score"])

    best_params = train_params_to_best_params(global_best["train_params"])
    batch_size = int(best_params["batch_size"])
    lr = float(best_params["lr"])
    best_params_clean = _convert_params(best_params, search_space)
    y_scaler = load_y_scaler_from_json(f"h{horizon}")

    n_folds = base_cfg["n_rolling_folds"]
    train_frac = base_cfg["rolling_train_frac"]
    folds = create_rolling_folds(n_total, n_folds=n_folds, train_frac=train_frac)
    logger.info("完整 3-fold 评估: train_frac=%.3f", train_frac)

    fold_losses = []
    for fold_idx, (tr_idx, va_idx) in enumerate(folds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = build_model(
            model_name, n_features=n_features, seq_len=seq_len,
            horizon=horizon, **best_params_clean,
        ).to(device)
        train_loader = make_loader(X_train[tr_idx], y_residual_train[tr_idx],
                                   batch_size=batch_size, shuffle=True)
        val_loader = make_loader(X_train[va_idx], y_residual_train[va_idx],
                                 batch_size=batch_size, shuffle=False)
        _, history = train_with_early_stop(
            model, train_loader, val_loader,
            lr=lr, max_epochs=n_epochs, patience=patience, device=device,
        )
        vloss = min(h["val_loss"] for h in history)
        fold_losses.append(vloss)
        logger.info("  fold %d  val_loss=%.6f", fold_idx, vloss)

    avg_val_loss = float(np.mean(fold_losses))
    logger.info("3-fold 平均 val_loss=%.6f", avg_val_loss)

    best_fold_idx = int(np.argmin(fold_losses))
    _, best_va_idx = folds[best_fold_idx]
    torch.manual_seed(seed)
    np.random.seed(seed)
    model_final = build_model(
        model_name, n_features=n_features, seq_len=seq_len,
        horizon=horizon, **best_params_clean,
    ).to(device)
    train_loader = make_loader(X_train[best_va_idx[0]:], y_residual_train[best_va_idx[0]:],
                               batch_size=batch_size, shuffle=True)
    val_loader = make_loader(X_train[best_va_idx], y_residual_train[best_va_idx],
                             batch_size=batch_size, shuffle=False)
    _, _ = train_with_early_stop(
        model_final, train_loader, val_loader,
        lr=lr, max_epochs=n_epochs, patience=patience, device=device,
    )
    y_pred_residual_scaled = predict(model_final, X_val, device)
    y_pred_residual = y_scaler.inverse_transform(y_pred_residual_scaled)
    y_true_power = y_anchor_val + y_residual_val
    y_pred_power = y_anchor_val + y_pred_residual
    avg_power_metrics = compute_all_metrics(y_true_power.ravel(), y_pred_power.ravel())
    logger.info("完整验证集功率指标: RMSE=%.4f  MAE=%.4f  R2=%.4f",
                avg_power_metrics["RMSE"], avg_power_metrics["MAE"], avg_power_metrics["R2"])

    total_trials = sum(v["trials"] for v in ablation.values())
    optuna_path = metrics_h / f"{model_name}_optuna.json"
    result = {
        "model": model_name,
        "horizon": horizon,
        "search_method": "optuna_afsa_hybrid",
        "best_strategy": global_best["strategy"],
        "best_params": best_params,
        "best_value": avg_val_loss,
        "quick_best_value": global_best["RMSE"],
        "quick_composite_score": global_best["composite_score"],
        "fold_losses": fold_losses,
        "best_fold_idx": best_fold_idx,
        "avg_power_metrics": avg_power_metrics,
        "n_trials": total_trials,
        "prediction_mode": "residual",
        "hybrid_ablation_path": str(ablation_path.name),
    }
    with open(optuna_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("最优参数结果已保存: %s", optuna_path.relative_to(Path.cwd()))

    return result


def run_baseline(model_name, horizon_cfg, base_cfg, logger):
    """Baseline 固定参数在 CV 上评估（残差预测模式）。"""
    horizon = horizon_cfg["horizon"]
    hdir = load_sample_dir(horizon)
    metrics_h = METRICS_DIR / f"h{horizon}"

    X_train = np.load(hdir / "X_train_seq.npy")
    y_residual_train = np.load(hdir / "y_train.npy")
    y_anchor_train = np.load(hdir / "y_anchor_train.npy")

    meta = json.loads((hdir / "meta.json").read_text(encoding="utf-8"))
    seq_len, n_features = meta["lookback"], X_train.shape[2]
    y_scaler = load_y_scaler_from_json(f"h{horizon}")

    params = horizon_cfg["baseline_params"][model_name]
    device = get_device()
    seed = base_cfg["reproduce_seeds"][0]

    n_total = len(X_train)
    folds = create_rolling_folds(n_total, n_folds=3, train_frac=base_cfg["rolling_train_frac"])

    val_losses = []
    fold_power_metrics = []
    for fold_idx, (tr_idx, va_idx) in enumerate(folds):
        torch.manual_seed(seed)
        np.random.seed(seed)

        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr, y_va = y_residual_train[tr_idx], y_residual_train[va_idx]
        y_anchor_va = y_anchor_train[va_idx]

        model_params = {k: v for k, v in params.items() if k not in ("lr", "batch_size")}
        model = build_model(
            model_name, n_features=n_features, seq_len=seq_len,
            horizon=horizon, **model_params,
        ).to(device)

        train_loader = make_loader(X_tr, y_tr, batch_size=params["batch_size"], shuffle=True)
        val_loader = make_loader(X_va, y_va, batch_size=params["batch_size"], shuffle=False)

        _, _ = train_with_early_stop(
            model, train_loader, val_loader,
            lr=params["lr"],
            max_epochs=horizon_cfg["n_epochs_trial"],
            patience=horizon_cfg["patience_trial"],
            device=device,
        )
        vloss = eval_loss(model, val_loader, nn.MSELoss(), device)
        val_losses.append(vloss)

        y_pred_residual_scaled = predict(model, X_va, device)
        y_pred_residual = y_scaler.inverse_transform(y_pred_residual_scaled)
        y_true_power = y_anchor_va + y_va
        y_pred_power = y_anchor_va + y_pred_residual
        m = compute_all_metrics(y_true_power.ravel(), y_pred_power.ravel())
        fold_power_metrics.append(m)

        logger.info("  fold %d  val_loss=%.6f  power_RMSE=%.4f  power_MAE=%.4f",
                    fold_idx, vloss, m["RMSE"], m["MAE"])

    avg_vloss = float(np.mean(val_losses))
    avg_power_metrics = {
        "RMSE": float(np.mean([m["RMSE"] for m in fold_power_metrics])),
        "MAE": float(np.mean([m["MAE"] for m in fold_power_metrics])),
        "MAPE": float(np.mean([m["MAPE"] for m in fold_power_metrics if not np.isnan(m["MAPE"])])),
        "R2": float(np.mean([m["R2"] for m in fold_power_metrics])),
    }
    logger.info("Baseline 3-fold 平均: val_loss=%.6f  power_RMSE=%.4f  power_MAE=%.4f",
                avg_vloss, avg_power_metrics["RMSE"], avg_power_metrics["MAE"])

    result = {
        "model": model_name, "horizon": horizon,
        "best_params": params, "best_value": avg_vloss,
        "avg_power_metrics": avg_power_metrics,
        "fold_power_metrics": fold_power_metrics,
        "note": "baseline_fixed_params",
        "prediction_mode": "residual",
    }
    path = metrics_h / f"{model_name}_optuna.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("Baseline 结果已保存: %s", path.name)
    return result


def main():
    parser = argparse.ArgumentParser(description="EXP-P04 Optuna-AFSA 混合超参搜索")
    parser.add_argument("--horizon", type=int, choices=[1, 4, 16], required=True)
    parser.add_argument("--model", type=str, default=None,
                        choices=["cnn_bilstm", "all"],
                        help="指定模型，默认为 all（全部跑一遍）")
    args = parser.parse_args()

    t0 = time.time()
    horizon = args.horizon
    horizon_cfg = load_config(f"exp_p04_h{horizon}.json")
    base_cfg = load_config("exp_p04_base.json")

    log_file = horizon_cfg["log_file"].replace(".log", "_optuna.log")
    logger = setup_logger("optuna", log_file)
    logger.info("=" * 60)
    logger.info("EXP-P04 Optuna-AFSA 混合搜索  horizon=%d", horizon)

    if args.model and args.model != "all":
        models_to_run = [args.model]
    else:
        models_to_run = horizon_cfg["baseline_models"] + list(horizon_cfg["model_search_space"].keys())

    all_results = []
    for mname in models_to_run:
        try:
            if mname in horizon_cfg["baseline_models"]:
                r = run_baseline(mname, horizon_cfg, base_cfg, logger)
            else:
                r = run_hybrid_search_for_model(mname, horizon_cfg, base_cfg, logger)
            all_results.append(r)
        except Exception as e:
            logger.error("模型 %s 搜索失败: %s", mname, e, exc_info=True)

    logger.info("=" * 60)
    logger.info("所有模型混合搜索完成！")
    for r in all_results:
        logger.info("  %-15s  best_val_loss=%.6f", r["model"], r["best_value"])

    elapsed = time.time() - t0
    hs = f"h{horizon}"
    metrics_h = METRICS_DIR / hs
    primary = next((r for r in all_results if r["model"] == "cnn_bilstm"), None)
    if primary is None:
        raise RuntimeError("cnn_bilstm 混合搜索未成功（无结果或运行失败）")

    pm = primary.get("avg_power_metrics", {})
    summary = {
        "models_completed": [r["model"] for r in all_results],
        "best_strategy": primary.get("best_strategy"),
        "best_val_loss": round(primary["best_value"], 6),
        "val_RMSE": round(pm.get("RMSE", float("nan")), 4),
        "val_MAE": round(pm.get("MAE", float("nan")), 4),
        "val_R2": round(pm.get("R2", float("nan")), 4),
        "n_trials": primary.get("n_trials"),
        "elapsed_sec": round(elapsed, 1),
    }
    artifacts = [
        str((metrics_h / "hybrid_search_ablation.json").relative_to(PROJECT_ROOT)),
        str((metrics_h / "cnn_bilstm_optuna.json").relative_to(PROJECT_ROOT)),
    ]
    record_step_result(
        horizon, "optuna", "success", log_file,
        summary=summary, duration_sec=elapsed, artifacts=artifacts,
    )
    return horizon, log_file


if __name__ == "__main__":
    t0 = time.time()
    try:
        main()
    except Exception as e:
        record_step_failure("optuna", t0, e)
        raise
