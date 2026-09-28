"""
亳州数据集 Optuna-AFSA 混合超参搜索 (EXP-P05)
python -m experiments.prediction.step2_hyperparameter_search.exp_02_hybrid_search_bozhou --horizon 1
python -m experiments.prediction.step2_hyperparameter_search.exp_02_hybrid_search_bozhou --horizon 4
python -m experiments.prediction.step2_hyperparameter_search.exp_02_hybrid_search_bozhou --horizon 16
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.prediction.step2_hyperparameter_search.exp_02_common import (
    METRICS_DIR,
    MODELS_DIR,
    compute_all_metrics,
    setup_logger,
    STEP4_ROOT,
)
from experiments.prediction.step5_reporting.exp_05_step_audit import (
    record_step_failure,
    record_step_result,
)
from experiments.prediction.step2_hyperparameter_search.exp_02_cv_split import create_rolling_folds
from experiments.prediction.step2_hyperparameter_search.exp_02_hybrid_optimizer import (
    run_all_strategies,
    train_params_to_best_params,
)
from experiments.prediction.step3_deep_learning.exp_03_models import build_model
from experiments.prediction.step3_deep_learning.exp_03_torch_utils import (
    get_device,
    make_loader,
    predict,
    train_with_early_stop,
)

# =============================================================================
# 亳州配置
# =============================================================================
BOZHOU_CFG = {
    "lookback": 16,
    "n_rolling_folds": 3,
    "rolling_train_frac": 0.667,
    "reproduce_seeds": [42, 43, 44, 45, 46],
    "hybrid_search": {
        "strategies": ["S1"],
        "n_trials": 5,
        "score_weights": {
            "rmse": 0.5,
            "mae": 0.25,
            "latency": 0.15,
            "params": 0.10
        }
    },
}


def load_bozhou_sample_dir(horizon: int, lookback: int = None) -> Path:
    """返回亳州样本目录（适配 samples 目录结构）"""
    if lookback is None:
        lookback = BOZHOU_CFG["lookback"]
    lb_suffix = f"_lb{lookback}"
    if horizon == 1:
        return STEP4_ROOT / "samples" / f"bozhou_h1{lb_suffix}"
    elif horizon == 4:
        return STEP4_ROOT / "samples" / f"bozhou_h4{lb_suffix}"
    elif horizon == 16:
        return STEP4_ROOT / "samples" / f"bozhou_h16{lb_suffix}"
    else:
        raise ValueError(f"Unsupported horizon: {horizon}")


def load_bozhou_y_scaler(horizon: int, lookback: int = None):
    """从 JSON 参数文件重建 y 的 StandardScaler"""
    from sklearn.preprocessing import StandardScaler
    
    hdir = load_bozhou_sample_dir(horizon, lookback)
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
        "n_epochs_trial": 10,
        "patience_trial": 3,
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
                "cnn_channels_0": [32, 64],
                "cnn_channels_1": [64, 128],
                "kernel_size": [3, 5],
            }
        },
    },
    4: {
        "horizon": 4,
        "lookback": 48,
        "n_epochs_trial": 10,
        "patience_trial": 3,
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
                "cnn_channels_0": [32, 64],
                "cnn_channels_1": [64, 128],
                "kernel_size": [3, 5],
            }
        },
    },
    16: {
        "horizon": 16,
        "lookback": 96,
        "n_epochs_trial": 10,
        "patience_trial": 3,
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
                "cnn_channels_0": [32, 64],
                "cnn_channels_1": [64, 128],
                "kernel_size": [3, 5],
            }
        },
    },
}


def _convert_params(params: dict, search_space: dict) -> dict:
    """将 Optuna 参数转换为模型参数格式"""
    out = {}
    for name, space in search_space.items():
        if name in ("lr", "batch_size"):
            continue
        if name.startswith("cnn_channels_"):
            continue
        # 从 params 中获取值，如果不存在则使用默认值
        vals_str = [str(v) for v in space]
        chosen = str(params.get(name, space[0]))
        if chosen in vals_str:
            try:
                out[name] = int(chosen)
            except ValueError:
                try:
                    out[name] = float(chosen)
                except ValueError:
                    out[name] = chosen
        else:
            val = params.get(name)
            if val is not None:
                out[name] = val
    
    # 组合 cnn_channels
    cnn_0 = params.get("cnn_channels_0", 32)
    cnn_1 = params.get("cnn_channels_1", 64)
    if isinstance(cnn_0, str):
        cnn_0 = int(cnn_0)
    if isinstance(cnn_1, str):
        cnn_1 = int(cnn_1)
    out["cnn_channels"] = [cnn_0, cnn_1]
    
    return out


def run_hybrid_search_for_model(model_name: str, horizon_cfg: dict, logger) -> dict:
    """对单个模型执行 Optuna-AFSA 混合消融搜索"""
    horizon = horizon_cfg["horizon"]
    lookback = horizon_cfg["lookback"]
    hdir = load_bozhou_sample_dir(horizon, lookback)
    
    metrics_h = METRICS_DIR / f"bozhou_h{horizon}"
    models_h = MODELS_DIR / f"bozhou_h{horizon}"
    for d in (metrics_h, models_h):
        d.mkdir(parents=True, exist_ok=True)

    logger.info("-" * 50)
    logger.info("亳州 Optuna-AFSA 混合搜索: model=%s horizon=%d", model_name, horizon)

    # 加载样本数据
    X_train = np.load(hdir / "X_train_seq.npy")
    y_train = np.load(hdir / "y_train.npy")
    X_val = np.load(hdir / "X_val_seq.npy")
    y_val = np.load(hdir / "y_val.npy")
    _, _, n_features = X_train.shape
    meta = json.loads((hdir / "meta.json").read_text(encoding="utf-8"))
    seq_len = meta["lookback"]

    device = get_device()
    logger.info("设备: %s | train=%d val=%d n_features=%d seq_len=%d horizon=%d",
                device, len(X_train), len(X_val), n_features, seq_len, horizon)

    search_space = horizon_cfg["model_search_space"][model_name]
    hybrid_cfg = BOZHOU_CFG["hybrid_search"]
    n_epochs = horizon_cfg["n_epochs_trial"]
    patience = horizon_cfg["patience_trial"]
    seed = BOZHOU_CFG["reproduce_seeds"][0]

    # 快速搜索：使用训练集后 1/3
    n_total = len(X_train)
    tr_end = int(n_total * 2 / 3)
    X_quick = X_train[tr_end:]
    y_quick = y_train[tr_end:]
    logger.info("快速搜索: fold split at %d quick_train=%d", tr_end, len(X_quick))

    # 执行混合搜索策略
    ablation, global_best = run_all_strategies(
        model_name=model_name,
        search_space=search_space,
        X_train=X_quick,
        y_train=y_quick,
        X_val=X_val,
        y_val=y_val,
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

    # 保存消融结果
    ablation_path = metrics_h / "bozhou_hybrid_search_ablation.json"
    with open(ablation_path, "w", encoding="utf-8") as f:
        json.dump(ablation, f, indent=2, ensure_ascii=False)
    logger.info("混合搜索消融结果已保存: %s", ablation_path.name)

    # 准备最佳参数
    best_params = train_params_to_best_params(global_best["train_params"])
    batch_size = int(best_params["batch_size"])
    lr = float(best_params["lr"])
    best_params_clean = _convert_params(best_params, search_space)
    y_scaler = load_bozhou_y_scaler(horizon, lookback)

    # 完整 3-fold 评估
    n_folds = BOZHOU_CFG["n_rolling_folds"]
    train_frac = BOZHOU_CFG["rolling_train_frac"]
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
        train_loader = make_loader(X_train[tr_idx], y_train[tr_idx],
                                   batch_size=batch_size, shuffle=True)
        val_loader = make_loader(X_train[va_idx], y_train[va_idx],
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

    # 在最佳 fold 上训练最终模型
    best_fold_idx = int(np.argmin(fold_losses))
    _, best_va_idx = folds[best_fold_idx]
    torch.manual_seed(seed)
    np.random.seed(seed)
    model_final = build_model(
        model_name, n_features=n_features, seq_len=seq_len,
        horizon=horizon, **best_params_clean,
    ).to(device)
    train_loader = make_loader(X_train[best_va_idx[0]:], y_train[best_va_idx[0]:],
                               batch_size=batch_size, shuffle=True)
    val_loader = make_loader(X_train[best_va_idx], y_train[best_va_idx],
                             batch_size=batch_size, shuffle=False)
    train_with_early_stop(
        model_final, train_loader, val_loader,
        lr=lr, max_epochs=n_epochs, patience=patience, device=device,
    )
    
    # 计算验证集指标
    y_pred_scaled = predict(model_final, X_val, device)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_true = y_scaler.inverse_transform(y_val)
    avg_power_metrics = compute_all_metrics(y_true.ravel(), y_pred.ravel())
    logger.info("完整验证集功率指标: RMSE=%.4f MAE=%.4f R2=%.4f",
                avg_power_metrics["RMSE"], avg_power_metrics["MAE"], avg_power_metrics["R2"])

    # 保存最终结果
    total_trials = sum(v["trials"] for v in ablation.values())
    optuna_path = metrics_h / f"bozhou_{model_name}_optuna.json"
    result = {
        "dataset": "bozhou",
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
    parser = argparse.ArgumentParser(description="亳州数据集 Optuna-AFSA 混合超参搜索 (EXP-P05)")
    parser.add_argument("--horizon", type=int, choices=[1, 4, 16], required=True,
                        help="预测步长: 1, 4, 或 16")
    parser.add_argument("--model", type=str, default="cnn_bilstm",
                        choices=["cnn_bilstm", "all"],
                        help="模型类型 (默认: cnn_bilstm)")
    args = parser.parse_args()

    t0 = time.time()
    horizon = args.horizon
    horizon_cfg = HORIZON_CONFIGS[horizon]

    log_file = f"EXP-P05_bozhou_h{horizon}_hybrid_search.log"
    logger = setup_logger("hybrid_bozhou", log_file)
    logger.info("=" * 60)
    logger.info("亳州数据集 Optuna-AFSA 混合搜索 horizon=%d", horizon)

    result = run_hybrid_search_for_model(args.model, horizon_cfg, logger)

    logger.info("=" * 60)
    elapsed = time.time() - t0
    pm = result.get("avg_power_metrics", {})
    summary = {
        "dataset": "bozhou",
        "model": result["model"],
        "best_strategy": result.get("best_strategy"),
        "best_val_loss": round(result["best_value"], 6),
        "val_RMSE": round(pm.get("RMSE", float("nan")), 4),
        "val_MAE": round(pm.get("MAE", float("nan")), 4),
        "val_R2": round(pm.get("R2", float("nan")), 4),
        "elapsed_sec": round(elapsed, 1),
    }
    logger.info("结果: %s", summary)
    
    metrics_h = METRICS_DIR / f"bozhou_h{horizon}"
    artifacts = [
        str((metrics_h / "bozhou_hybrid_search_ablation.json").relative_to(PROJECT_ROOT)),
        str((metrics_h / f"bozhou_{args.model}_optuna.json").relative_to(PROJECT_ROOT)),
    ]
    record_step_result(
        horizon, "hybrid_search", "success", log_file,
        summary=summary, duration_sec=elapsed, artifacts=artifacts,
    )
    return horizon, log_file


if __name__ == "__main__":
    t0 = time.time()
    try:
        main()
    except Exception as e:
        record_step_failure("hybrid_search", t0, e)
        raise
