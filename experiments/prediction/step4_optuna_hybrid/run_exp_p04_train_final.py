"""
python -m experiments.prediction.step4_optuna_hybrid.run_exp_p04_train_final --horizon 1
"""

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.prediction.step4_optuna_hybrid.exp_p04_common import (
    METRICS_DIR,
    MODELS_DIR,
    PRED_DIR,
    SAMPLES_DIR,
    compute_all_metrics,
    compute_metrics_multi_step,
    load_config,
    load_y_scaler_from_json,
    save_predictions,
    save_train_history,
    setup_logger,
)
from experiments.prediction.step4_optuna_hybrid.exp_p04_models import build_model
from experiments.prediction.step4_optuna_hybrid.exp_p04_torch_utils import (
    eval_loss,
    get_device,
    make_loader,
    predict,
    train_with_early_stop,
)


def _best_params(params: dict, model_name: str) -> dict:
    """兼容从 optuna JSON 读取的参数（包含 batch_size/lr）。"""
    out = {}
    skip = {"batch_size", "lr"}
    for k, v in params.items():
        if k in skip:
            continue
        if isinstance(v, str):
            try:
                out[k] = int(v)
            except ValueError:
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
        else:
            out[k] = v
    return out


def train_and_evaluate(model_name: str, params: dict, horizon: int, logger, device,
                       X_train, y_train, X_val, y_val, X_test, y_test,
                       y_scaler, seq_len, n_features, max_epochs, patience, lr, batch_size):
    """用最优参数训练并在测试集上评估。"""
    torch.manual_seed(42)
    np.random.seed(42)

    model = build_model(
        model_name,
        n_features=n_features,
        seq_len=seq_len,
        horizon=horizon,
        **_best_params(params, model_name),
    ).to(device)

    train_loader = make_loader(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, batch_size=batch_size, shuffle=False)

    t0 = time.time()
    model, history = train_with_early_stop(
        model, train_loader, val_loader,
        lr=lr, max_epochs=max_epochs, patience=patience, device=device,
    )
    train_time = time.time() - t0

    # 预测
    y_pred_scaled = predict(model, X_test, device)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)

    # 反标准化真实值
    y_test_raw = y_scaler.inverse_transform(y_test)

    metrics = compute_all_metrics(y_test_raw.ravel(), y_pred.ravel())
    metrics["training_time_sec"] = round(train_time, 2)

    # 多步指标
    if horizon > 1:
        step_metrics = compute_metrics_multi_step(y_test_raw, y_pred)
        metrics["step_metrics"] = step_metrics
        metrics["avg_MAE"] = float(np.mean([s["MAE"] for s in step_metrics]))
        metrics["avg_RMSE"] = float(np.mean([s["RMSE"] for s in step_metrics]))
        metrics["avg_MAPE"] = float(np.mean([s["MAPE"] for s in step_metrics if not np.isnan(s["MAPE"])]))
        metrics["avg_R2"] = float(np.mean([s["R2"] for s in step_metrics]))

    return model, history, metrics


def run_all_models(horizon: int, horizon_cfg: dict, base_cfg: dict, logger):
    """加载样本 + 加载最优参数 + 训练 + 保存模型和指标。"""
    hdir = SAMPLES_DIR / f"h{horizon}"
    metrics_h = METRICS_DIR / f"h{horizon}"
    models_h = MODELS_DIR / f"h{horizon}"
    pred_h = PRED_DIR / f"h{horizon}"
    for d in (metrics_h, models_h, pred_h):
        d.mkdir(parents=True, exist_ok=True)

    # 加载样本
    X_train = np.load(hdir / "X_train_seq.npy")
    y_train = np.load(hdir / "y_train.npy")
    X_val = np.load(hdir / "X_val_seq.npy")
    y_val = np.load(hdir / "y_val.npy")
    X_test = np.load(hdir / "X_test_seq.npy")
    y_test = np.load(hdir / "y_test.npy")
    y_scaler = load_y_scaler_from_json(f"h{horizon}")

    meta = json.loads((hdir / "meta.json").read_text(encoding="utf-8"))
    seq_len, n_features = meta["lookback"], X_train.shape[2]

    device = get_device()
    logger.info("=" * 60)
    logger.info("最终训练  horizon=%d  train=%d  val=%d  test=%d",
                horizon, len(X_train), len(X_val), len(X_test))

    max_epochs = base_cfg["final_max_epochs"]
    patience = base_cfg["final_patience"]
    lr = base_cfg["final_lr"]

    all_models = horizon_cfg["baseline_models"] + list(horizon_cfg["model_search_space"].keys())

    results = {}
    for mname in all_models:
        optuna_path = metrics_h / f"{mname}_optuna.json"
        if not optuna_path.exists():
            logger.warning("跳过 %s（无 Optuna 结果）", mname)
            continue

        params_json = json.loads(optuna_path.read_text(encoding="utf-8"))
        params = params_json["best_params"]
        batch_size = int(params.pop("batch_size", 64))
        lr_use = float(params.pop("lr", lr))

        logger.info("-" * 40)
        logger.info("训练模型: %s  horizon=%d", mname, horizon)
        logger.info("  参数: %s", {**params, "batch_size": batch_size, "lr": lr_use})

        try:
            model, history, metrics = train_and_evaluate(
                mname, params, horizon, logger, device,
                X_train, y_train, X_val, y_val, X_test, y_test,
                y_scaler, seq_len, n_features,
                max_epochs, patience, lr_use, batch_size,
            )

            # 保存模型
            model_path = models_h / f"{mname}_final.pt"
            torch.save(model.state_dict(), model_path)
            logger.info("  模型已保存: %s", model_path.name)

            # 保存训练历史
            hist_path = save_train_history(f"h{horizon}", f"{mname}_final", history)
            logger.info("  历史已保存: %s", hist_path.name)

            # 保存预测
            y_pred_scaled = predict(model, X_test, device)
            y_pred = y_scaler.inverse_transform(y_pred_scaled)
            y_test_raw = y_scaler.inverse_transform(y_test)
            pred_path = save_predictions(f"h{horizon}", mname, y_test_raw.ravel(), y_pred.ravel())
            logger.info("  预测已保存: %s", pred_path.name)

            # 保存指标
            metrics_path = metrics_h / f"{mname}_test_metrics.json"
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            logger.info("  指标: MAE=%.4f  RMSE=%.4f  MAPE=%.2f%%  R2=%.4f",
                        metrics["MAE"], metrics["RMSE"], metrics["MAPE"], metrics["R2"])

            results[mname] = metrics

        except Exception as e:
            logger.error("模型 %s 训练失败: %s", mname, e, exc_info=True)

    logger.info("=" * 60)
    logger.info("所有模型最终训练完成！")
    return results


def main():
    parser = argparse.ArgumentParser(description="EXP-P04 最终训练")
    parser.add_argument("--horizon", type=int, choices=[1, 4, 16], required=True)
    args = parser.parse_args()

    horizon = args.horizon
    horizon_cfg = load_config(f"exp_p04_h{horizon}.json")
    base_cfg = load_config("exp_p04_base.json")

    log_file = horizon_cfg["log_file"].replace(".log", "_final_train.log")
    logger = setup_logger("final_train", log_file)
    logger.info("=" * 60)
    logger.info("EXP-P04 最终训练  horizon=%d", horizon)

    run_all_models(horizon, horizon_cfg, base_cfg, logger)


if __name__ == "__main__":
    main()
