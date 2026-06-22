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

import joblib
import numpy as np
import optuna
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.prediction.step4_optuna_hybrid.exp_p04_common import (
    METRICS_DIR,
    MODELS_DIR,
    SAMPLES_DIR,
    compute_all_metrics,
    load_config,
    setup_logger,
)
from experiments.prediction.step4_optuna_hybrid.exp_p04_cv import create_rolling_folds
from experiments.prediction.step4_optuna_hybrid.exp_p04_models import build_model
from experiments.prediction.step4_optuna_hybrid.exp_p04_torch_utils import (
    eval_loss,
    get_device,
    make_loader,
    run_epoch,
    train_with_early_stop,
)


def _objective(model_name, search_space, trial: optuna.Trial,
               X_train, y_train, X_val, y_val,
               seq_len, n_features, horizon, n_epochs, patience, device):
    """单 trial 目标函数（固定随机种子）。"""
    torch.manual_seed(42)
    np.random.seed(42)

    params = {}
    for name, space in search_space.items():
        if isinstance(space, list):
            if all(isinstance(v, int) for v in space):
                params[name] = trial.suggest_categorical(name, [str(v) for v in space])
                params[name] = int(params[name])
            elif all(isinstance(v, float) for v in space):
                params[name] = trial.suggest_float(name, min(space), max(space), log=True)
            else:
                params[name] = trial.suggest_categorical(name, space)
        else:
            params[name] = space

    batch_size = int(params.pop("batch_size"))
    lr = float(params.pop("lr"))

    model = build_model(
        model_name,
        n_features=n_features,
        seq_len=seq_len,
        horizon=horizon,
        **params,
    )
    model = model.to(device)

    train_loader = make_loader(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, batch_size=batch_size, shuffle=False)

    _, history = train_with_early_stop(
        model, train_loader, val_loader,
        lr=lr, max_epochs=n_epochs, patience=patience, device=device,
    )
    best_val = min(h["val_loss"] for h in history)
    return best_val


def _suggest_with_fallback(search_space, trial, prefix=""):
    out = {}
    for name, space in search_space.items():
        key = f"{prefix}{name}" if prefix else name
        if isinstance(space, list):
            vals_str = [str(v) for v in space]
            chosen = trial.suggest_categorical(key, vals_str)
            try:
                out[name] = int(chosen)
            except ValueError:
                try:
                    out[name] = float(chosen)
                except ValueError:
                    out[name] = chosen
        else:
            out[name] = space
    return out


def objective_with_fixed_fold(model_name, search_space, trial,
                               X_train, y_train, X_val, y_val,
                               seq_len, n_features, horizon, n_epochs, patience, device):
    torch.manual_seed(42)
    np.random.seed(42)

    params = _suggest_with_fallback(search_space, trial)
    batch_size = int(params.pop("batch_size"))
    lr = float(params.pop("lr"))

    model = build_model(
        model_name, n_features=n_features, seq_len=seq_len, horizon=horizon, **params
    ).to(device)

    train_loader = make_loader(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, batch_size=batch_size, shuffle=False)

    _, history = train_with_early_stop(
        model, train_loader, val_loader,
        lr=lr, max_epochs=n_epochs, patience=patience, device=device,
    )
    return min(h["val_loss"] for h in history)


def run_optuna_for_model(model_name, horizon_cfg, base_cfg, logger):
    """对单个模型执行 Optuna 调参（trial 用单 fold），最优参数做 3-fold 评估，结果存 metrics/{h}/{model}_optuna.json。"""
    horizon = horizon_cfg["horizon"]
    hdir = SAMPLES_DIR / f"h{horizon}"
    metrics_h = METRICS_DIR / f"h{horizon}"
    models_h = MODELS_DIR / f"h{horizon}"
    for d in (metrics_h, models_h):
        d.mkdir(parents=True, exist_ok=True)

    logger.info("-" * 50)
    logger.info("开始 Optuna 调参: model=%s  horizon=%s", model_name, horizon)

    X_train = np.load(hdir / "X_train_seq.npy")
    y_train = np.load(hdir / "y_train.npy")
    X_val = np.load(hdir / "X_val_seq.npy")
    y_val = np.load(hdir / "y_val.npy")
    _, _, n_features = X_train.shape
    meta = json.loads((hdir / "meta.json").read_text(encoding="utf-8"))
    seq_len = meta["lookback"]

    device = get_device()
    logger.info("设备: %s  |  train=%d  val=%d  n_features=%d  seq_len=%d  horizon=%d",
                device, len(X_train), len(X_val), n_features, seq_len, horizon)

    search_space = horizon_cfg["model_search_space"][model_name]
    n_trials = base_cfg["optuna_n_trials_per_model"]
    n_epochs = horizon_cfg["n_epochs_trial"]
    patience = horizon_cfg["patience_trial"]

    # Step 1: 用单 fold（后 1/3 的训练数据）快速筛选参数
    n_total = len(X_train)
    tr_end = int(n_total * 2 / 3)
    X_quick = X_train[tr_end:]
    y_quick = y_train[tr_end:]
    logger.info("Trial 快速搜索: fold split at %d  quick_train=%d", tr_end, len(X_quick))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(
        lambda trial: objective_with_fixed_fold(
            model_name, search_space, trial,
            X_quick, y_quick, X_val, y_val,
            seq_len, n_features, horizon, n_epochs, patience, device,
        ),
        n_trials=n_trials,
        n_jobs=1,
        show_progress_bar=False,
    )
    logger.info("快速搜索完成: best_quick_val_loss=%.6f  trials=%d", study.best_value, len(study.trials))

    # Step 2: 用最优参数在完整 3-fold 上评估
    # Optuna best_params 返回字符串（categorical），需转回正确类型
    def _convert_params(params: dict, search_space: dict) -> dict:
        out = {}
        for name, space in search_space.items():
            vals_str = [str(v) for v in space]
            chosen = str(params[name])
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

    best_params = study.best_params
    batch_size = int(best_params["batch_size"])
    lr = float(best_params["lr"])
    best_params_clean = _convert_params(
        {k: v for k, v in best_params.items() if k not in ("batch_size", "lr")},
        {k: v for k, v in search_space.items() if k not in ("lr", "batch_size")}
    )

    n_folds = base_cfg["n_rolling_folds"]
    train_frac = base_cfg["rolling_train_frac"]
    folds = create_rolling_folds(n_total, n_folds=n_folds, train_frac=train_frac)
    logger.info("完整 3-fold 评估: train_frac=%.3f", train_frac)

    fold_losses = []
    for fold_idx, (tr_idx, va_idx) in enumerate(folds):
        torch.manual_seed(42)
        np.random.seed(42)
        model = build_model(
            model_name, n_features=n_features, seq_len=seq_len,
            horizon=horizon, **best_params_clean
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
        logger.info("  fold %d  val_loss=%.6f", fold_idx, vloss)

    avg_val_loss = float(np.mean(fold_losses))
    logger.info("3-fold 平均 val_loss=%.6f", avg_val_loss)

    # 保存结果
    optuna_path = metrics_h / f"{model_name}_optuna.json"
    result = {
        "model": model_name,
        "horizon": horizon,
        "best_params": best_params,
        "best_value": avg_val_loss,      # 3-fold 平均值
        "quick_best_value": study.best_value,  # 单 fold 搜索最优值
        "fold_losses": fold_losses,
        "n_trials": len(study.trials),
    }
    with open(optuna_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("Optuna 结果已保存: %s", optuna_path.relative_to(Path.cwd()))

    return result


def run_baseline(model_name, horizon_cfg, base_cfg, logger):
    """Baseline LSTM/BiLSTM，直接用固定参数在 CV 上评估。"""
    horizon = horizon_cfg["horizon"]
    hdir = SAMPLES_DIR / f"h{horizon}"
    metrics_h = METRICS_DIR / f"h{horizon}"

    X_train = np.load(hdir / "X_train_seq.npy")
    y_train = np.load(hdir / "y_train.npy")
    X_val = np.load(hdir / "X_val_seq.npy")
    y_val = np.load(hdir / "y_val.npy")

    meta = json.loads((hdir / "meta.json").read_text(encoding="utf-8"))
    seq_len, n_features = meta["lookback"], X_train.shape[2]

    params = horizon_cfg["baseline_params"][model_name]
    device = get_device()

    n_total = len(X_train)
    folds = create_rolling_folds(n_total, n_folds=3, train_frac=base_cfg["rolling_train_frac"])

    val_losses = []
    for fold_idx, (tr_idx, va_idx) in enumerate(folds):
        torch.manual_seed(42)
        np.random.seed(42)

        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]

        model_params = {k: v for k, v in params.items() if k not in ("lr", "batch_size")}
        model = build_model(
            model_name, n_features=n_features, seq_len=seq_len,
            horizon=horizon, **model_params
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
        logger.info("  fold %d  val_loss=%.6f", fold_idx, vloss)

    avg_vloss = float(np.mean(val_losses))
    result = {
        "model": model_name, "horizon": horizon,
        "best_params": params, "best_value": avg_vloss,
        "note": "baseline_fixed_params",
    }
    path = metrics_h / f"{model_name}_optuna.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("Baseline 结果已保存: %s  avg_val_loss=%.6f", path.name, avg_vloss)
    return result


def main():
    parser = argparse.ArgumentParser(description="EXP-P04 Optuna 调参")
    parser.add_argument("--horizon", type=int, choices=[1, 4, 16], required=True)
    parser.add_argument("--model", type=str, default=None,
                        choices=["lstm", "bilstm", "cnn_lstm", "cnn_bilstm", "minipatchtst", "all"],
                        help="指定模型，默认为 all（全部跑一遍）")
    args = parser.parse_args()

    horizon = args.horizon
    horizon_cfg = load_config(f"exp_p04_h{horizon}.json")
    base_cfg = load_config("exp_p04_base.json")

    log_file = horizon_cfg["log_file"].replace(".log", "_optuna.log")
    logger = setup_logger("optuna", log_file)
    logger.info("=" * 60)
    logger.info("EXP-P04 Optuna 调参  horizon=%d", horizon)

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
                r = run_optuna_for_model(mname, horizon_cfg, base_cfg, logger)
            all_results.append(r)
        except Exception as e:
            logger.error("模型 %s 调参失败: %s", mname, e, exc_info=True)

    logger.info("=" * 60)
    logger.info("所有模型调参完成！")
    for r in all_results:
        logger.info("  %-15s  best_val_loss=%.6f", r["model"], r["best_value"])


if __name__ == "__main__":
    main()
