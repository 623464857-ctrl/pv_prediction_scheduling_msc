"""
明月湖数据集 Optuna-AFSA 混合超参搜索
python -m experiments.prediction.step2_hyperparameter_search.run_exp_p04_optuna_mingyuehu --horizon 1
python -m experiments.prediction.step2_hyperparameter_search.run_exp_p04_optuna_mingyuehu --horizon 4
python -m experiments.prediction.step2_hyperparameter_search.run_exp_p04_optuna_mingyuehu --horizon 16
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.prediction.step2_hyperparameter_search.exp_p04_common import (
    METRICS_DIR,
    MODELS_DIR,
    compute_all_metrics,
    setup_logger,
)
from experiments.prediction.step5_reporting.exp_p04_step_audit import (
    record_step_failure,
    record_step_result,
)
from experiments.prediction.step2_hyperparameter_search.exp_p04_cv import create_rolling_folds
from experiments.prediction.step2_hyperparameter_search.exp_p04_hybrid_search import (
    run_all_strategies,
    train_params_to_best_params,
)
from experiments.prediction.step3_deep_learning.exp_p04_models import build_model
from experiments.prediction.step3_deep_learning.exp_p04_torch_utils import (
    eval_loss,
    get_device,
    make_loader,
    predict,
    train_with_early_stop,
)

# =============================================================================
# 明月湖配置
# =============================================================================
MINGYUEHU_CFG = {
    "lookback": 16,
    "n_rolling_folds": 3,
    "rolling_train_frac": 0.667,
    "reproduce_seeds": [42, 43, 44, 45, 46],
    "hybrid_search": {
        "strategies": ["S2", "S3", "S4", "S5", "S6"],
        "n_trials": 20,
        "score_weights": {
            "rmse": 0.5,
            "mae": 0.25,
            "latency": 0.15,
            "params": 0.10
        }
    },
}

# 样本目录加载器（明月湖专用）
def load_mingyuehu_sample_dir(horizon: int, lookback: int = None) -> Path:
    """返回明月湖样本目录"""
    if lookback is None:
        lookback = MINGYUEHU_CFG["lookback"]
    from experiments.prediction.step2_hyperparameter_search.exp_p04_common import SAMPLES_DIR
    return SAMPLES_DIR / f"mingyuehu_h{horizon}_lb{lookback}"


def load_mingyuehu_y_scaler(horizon: int, lookback: int = None):
    """从 JSON 参数文件重建 y 的 StandardScaler (明月湖专用)"""
    from sklearn.preprocessing import StandardScaler
    
    hdir = load_mingyuehu_sample_dir(horizon, lookback)
    params = json.loads((hdir / "scaler_params.json").read_text(encoding="utf-8"))
    scaler = StandardScaler()
    scaler.mean_ = np.array(params["y_mean"])
    scaler.scale_ = np.array(params["y_scale"])
    scaler.n_features_in_ = len(params["y_mean"])
    scaler.n_samples_seen_ = None
    return scaler


# =============================================================================
# Horizon 配置
# =============================================================================
HORIZON_CONFIGS = {
    1: {
        "horizon": 1,
        "lookback": 16,
        "n_epochs_trial": 30,
        "patience_trial": 5,
        "baseline_params": {
            "cnn_bilstm": {
                "lr": 0.001, "batch_size": 64,
                "hidden_size": 64, "num_layers": 2, "dropout": 0.2,
                "cnn_channels": [32, 64], "kernel_size": 3,
            }
        },
        "model_search_space": {
            "cnn_bilstm": {
                "lr": [0.0005, 0.001, 0.002],
                "batch_size": [32, 64, 128],
                "hidden_size": [32, 64, 128],
                "num_layers": [1, 2, 3],
                "dropout": [0.1, 0.2, 0.3],
                "cnn_channels": ["32,64", "64,128"],
                "kernel_size": [3, 5],
            }
        },
    },
    4: {
        "horizon": 4,
        "lookback": 48,
        "n_epochs_trial": 30,
        "patience_trial": 5,
        "baseline_params": {
            "cnn_bilstm": {
                "lr": 0.001, "batch_size": 64,
                "hidden_size": 64, "num_layers": 2, "dropout": 0.2,
                "cnn_channels": [32, 64], "kernel_size": 3,
            }
        },
        "model_search_space": {
            "cnn_bilstm": {
                "lr": [0.0005, 0.001, 0.002],
                "batch_size": [32, 64, 128],
                "hidden_size": [32, 64, 128],
                "num_layers": [1, 2, 3],
                "dropout": [0.1, 0.2, 0.3],
                "cnn_channels": ["32,64", "64,128"],
                "kernel_size": [3, 5],
            }
        },
    },
    16: {
        "horizon": 16,
        "lookback": 96,
        "n_epochs_trial": 30,
        "patience_trial": 5,
        "baseline_params": {
            "cnn_bilstm": {
                "lr": 0.001, "batch_size": 64,
                "hidden_size": 64, "num_layers": 2, "dropout": 0.2,
                "cnn_channels": [32, 64], "kernel_size": 3,
            }
        },
        "model_search_space": {
            "cnn_bilstm": {
                "lr": [0.0005, 0.001, 0.002],
                "batch_size": [32, 64, 128],
                "hidden_size": [32, 64, 128],
                "num_layers": [1, 2, 3],
                "dropout": [0.1, 0.2, 0.3],
                "cnn_channels": ["32,64", "64,128"],
                "kernel_size": [3, 5],
            }
        },
    },
}


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


def run_hybrid_search_for_model(model_name, horizon_cfg, logger):
    """对单个模型执行 Optuna-AFSA 混合消融搜索"""
    horizon = horizon_cfg["horizon"]
    lookback = horizon_cfg["lookback"]
    hdir = load_mingyuehu_sample_dir(horizon, lookback)
    
    metrics_h = METRICS_DIR / f"mingyuehu_h{horizon}"
    models_h = MODELS_DIR / f"mingyuehu_h{horizon}"
    for d in (metrics_h, models_h):
        d.mkdir(parents=True, exist_ok=True)

    logger.info("-" * 50)
    logger.info("明月湖 Optuna-AFSA 混合搜索: model=%s horizon=%d", model_name, horizon)

    X_train = np.load(hdir / "X_train_seq.npy")
    y_residual_train = np.load(hdir / "y_train.npy")
    X_val = np.load(hdir / "X_val_seq.npy")
    y_residual_val = np.load(hdir / "y_val.npy")
    y_anchor_val = np.load(hdir / "y_anchor_val.npy")
    _, _, n_features = X_train.shape
    meta = json.loads((hdir / "meta.json").read_text(encoding="utf-8"))
    seq_len = meta["lookback"]

    device = get_device()
    logger.info("设备: %s | train=%d val=%d n_features=%d seq_len=%d horizon=%d",
                device, len(X_train), len(X_val), n_features, seq_len, horizon)

    search_space = horizon_cfg["model_search_space"][model_name]
    hybrid_cfg = MINGYUEHU_CFG["hybrid_search"]
    n_epochs = horizon_cfg["n_epochs_trial"]
    patience = horizon_cfg["patience_trial"]
    seed = MINGYUEHU_CFG["reproduce_seeds"][0]

    n_total = len(X_train)
    tr_end = int(n_total * 2 / 3)
    X_quick = X_train[tr_end:]
    y_quick_residual = y_residual_train[tr_end:]
    logger.info("快速搜索: fold split at %d quick_train=%d", tr_end, len(X_quick))

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

    ablation_path = metrics_h / "mingyuehu_hybrid_search_ablation.json"
    with open(ablation_path, "w", encoding="utf-8") as f:
        json.dump(ablation, f, indent=2, ensure_ascii=False)
    logger.info("混合搜索消融结果已保存: %s", ablation_path.name)

    best_params = train_params_to_best_params(global_best["train_params"])
    batch_size = int(best_params["batch_size"])
    lr = float(best_params["lr"])
    best_params_clean = _convert_params(best_params, search_space)
    y_scaler = load_mingyuehu_y_scaler(horizon, lookback)

    n_folds = MINGYUEHU_CFG["n_rolling_folds"]
    train_frac = MINGYUEHU_CFG["rolling_train_frac"]
    folds = create_rolling_folds(n_total, n_folds=n_folds, train_frac=train_frac)
    logger.info("完整 3-fold 评估")

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
        logger.info("  fold %d val_loss=%.6f", fold_idx, vloss)

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
    train_with_early_stop(
        model_final, train_loader, val_loader,
        lr=lr, max_epochs=n_epochs, patience=patience, device=device,
    )
    y_pred_residual_scaled = predict(model_final, X_val, device)
    y_pred_residual = y_scaler.inverse_transform(y_pred_residual_scaled)
    y_true_power = y_anchor_val + y_residual_val
    y_pred_power = y_anchor_val + y_pred_residual
    avg_power_metrics = compute_all_metrics(y_true_power.ravel(), y_pred_power.ravel())
    logger.info("完整验证集功率指标: RMSE=%.4f MAE=%.4f R2=%.4f",
                avg_power_metrics["RMSE"], avg_power_metrics["MAE"], avg_power_metrics["R2"])

    total_trials = sum(v["trials"] for v in ablation.values())
    optuna_path = metrics_h / f"mingyuehu_{model_name}_optuna.json"
    result = {
        "dataset": "mingyuehu",
        "model": model_name,
        "horizon": horizon,
        "lookback": lookback,
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
    }
    with open(optuna_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("最优参数结果已保存: %s", optuna_path.name)

    return result


def main():
    parser = argparse.ArgumentParser(description="明月湖 Optuna-AFSA 混合超参搜索")
    parser.add_argument("--horizon", type=int, choices=[1, 4, 16], required=True)
    parser.add_argument("--model", type=str, default="cnn_bilstm",
                        choices=["cnn_bilstm", "all"])
    args = parser.parse_args()

    t0 = time.time()
    horizon = args.horizon
    horizon_cfg = HORIZON_CONFIGS[horizon]

    log_file = f"EXP-P04_mingyuehu_h{horizon}_optuna.log"
    logger = setup_logger("optuna_mingyuehu", log_file)
    logger.info("=" * 60)
    logger.info("明月湖数据集 Optuna-AFSA 混合搜索 horizon=%d", horizon)

    result = run_hybrid_search_for_model(args.model, horizon_cfg, logger)

    logger.info("=" * 60)
    elapsed = time.time() - t0
    pm = result.get("avg_power_metrics", {})
    summary = {
        "dataset": "mingyuehu",
        "model": result["model"],
        "best_strategy": result.get("best_strategy"),
        "best_val_loss": round(result["best_value"], 6),
        "val_RMSE": round(pm.get("RMSE", float("nan")), 4),
        "val_MAE": round(pm.get("MAE", float("nan")), 4),
        "val_R2": round(pm.get("R2", float("nan")), 4),
        "elapsed_sec": round(elapsed, 1),
    }
    logger.info("结果: %s", summary)
    
    metrics_h = METRICS_DIR / f"mingyuehu_h{horizon}"
    artifacts = [
        str((metrics_h / "mingyuehu_hybrid_search_ablation.json").relative_to(PROJECT_ROOT)),
        str((metrics_h / f"mingyuehu_{args.model}_optuna.json").relative_to(PROJECT_ROOT)),
    ]
    record_step_result(
        horizon, "optuna_mingyuehu", "success", log_file,
        summary=summary, duration_sec=elapsed, artifacts=artifacts,
    )
    return horizon, log_file


if __name__ == "__main__":
    t0 = time.time()
    try:
        main()
    except Exception as e:
        record_step_failure("optuna_mingyuehu", t0, e)
        raise
