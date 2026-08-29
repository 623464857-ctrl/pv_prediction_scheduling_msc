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
    PROJECT_ROOT,
    SAMPLES_DIR,
    compute_all_metrics,
    compute_metrics_multi_step,
    load_config,
    load_sample_dir,
    load_y_scaler_from_json,
    save_predictions,
    save_train_history,
    setup_logger,
)
from experiments.prediction.step4_optuna_hybrid.exp_p04_step_audit import (
    record_step_failure,
    record_step_result,
)
from experiments.prediction.step4_optuna_hybrid.exp_p04_features import (
    load_sample_arrays_with_feature_selection,
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


def train_and_evaluate_residual(model_name: str, params: dict, horizon: int, logger, device,
                                 X_train, y_residual_train, X_val, y_residual_val,
                                 X_test, y_residual_test, y_anchor_test,
                                 y_scaler, seq_len, n_features, max_epochs, patience, lr, batch_size):
    """用最优参数训练残差模型并在测试集上评估（重构后的真实功率）。"""
    torch.manual_seed(42)
    np.random.seed(42)

    model = build_model(
        model_name,
        n_features=n_features,
        seq_len=seq_len,
        horizon=horizon,
        **_best_params(params, model_name),
    ).to(device)

    train_loader = make_loader(X_train, y_residual_train, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_residual_val, batch_size=batch_size, shuffle=False)

    t0 = time.time()
    model, history = train_with_early_stop(
        model, train_loader, val_loader,
        lr=lr, max_epochs=max_epochs, patience=patience, device=device,
    )
    train_time = time.time() - t0

    # 预测残差 (标准化后)
    y_pred_residual_scaled = predict(model, X_test, device)
    # 反标准化残差预测
    y_pred_residual = y_scaler.inverse_transform(y_pred_residual_scaled)
    # 重构功率: y_hat = y_anchor + y_residual_pred
    y_pred_power = (y_anchor_test + y_pred_residual).astype(np.float32)

    # 反标准化真实残差，计算真实功率
    y_test_residual_raw = y_scaler.inverse_transform(y_residual_test)
    y_true_power = (y_anchor_test + y_test_residual_raw).astype(np.float32)

    metrics = compute_all_metrics(y_true_power.ravel(), y_pred_power.ravel())
    metrics["training_time_sec"] = round(train_time, 2)
    metrics["prediction_mode"] = "residual"

    # 多步指标
    if horizon > 1:
        step_metrics = compute_metrics_multi_step(y_true_power, y_pred_power)
        metrics["step_metrics"] = step_metrics
        metrics["avg_MAE"] = float(np.mean([s["MAE"] for s in step_metrics]))
        metrics["avg_RMSE"] = float(np.mean([s["RMSE"] for s in step_metrics]))
        metrics["avg_MAPE"] = float(np.mean([s["MAPE"] for s in step_metrics if not np.isnan(s["MAPE"])]))
        metrics["avg_R2"] = float(np.mean([s["R2"] for s in step_metrics]))

    return model, history, metrics, y_pred_power


def train_and_evaluate(model_name: str, params: dict, horizon: int, logger, device,
                       X_train, y_train, X_val, y_val, X_test, y_test,
                       y_scaler, seq_len, n_features, max_epochs, patience, lr, batch_size):
    """用最优参数训练并在测试集上评估（直接预测模式）。"""
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

    return model, history, metrics, y_pred


def run_all_models(horizon: int, horizon_cfg: dict, base_cfg: dict, logger):
    """加载样本 + 加载最优参数 + 训练 + 保存模型和指标。"""
    hdir = load_sample_dir(horizon)
    metrics_h = METRICS_DIR / f"h{horizon}"
    models_h = MODELS_DIR / f"h{horizon}"
    pred_h = PRED_DIR / f"h{horizon}"
    for d in (metrics_h, models_h, pred_h):
        d.mkdir(parents=True, exist_ok=True)

    # 加载样本（含可选特征筛选）
    samples = load_sample_arrays_with_feature_selection(
        hdir, base_cfg, logger, horizon, target_mode="residual",
    )
    X_train = samples["X_train"]
    y_residual_train = samples["y_train"]
    X_val = samples["X_val"]
    y_residual_val = samples["y_val"]
    X_test = samples["X_test"]
    y_residual_test = samples["y_test"]
    y_anchor_test = samples["y_anchor_test"]
    y_scaler = load_y_scaler_from_json(f"h{horizon}")

    meta = samples["meta"]
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
            model, history, metrics, y_pred_power = train_and_evaluate_residual(
                mname, params, horizon, logger, device,
                X_train, y_residual_train, X_val, y_residual_val,
                X_test, y_residual_test, y_anchor_test,
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

            # 计算真实功率用于保存预测
            y_test_residual_raw = y_scaler.inverse_transform(y_residual_test)
            y_true_power = (y_anchor_test + y_test_residual_raw).astype(np.float32)
            # 保存预测 (真实功率 vs 预测功率)
            pred_path = save_predictions(f"h{horizon}", mname, y_true_power.ravel(), y_pred_power.ravel())
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

    t0 = time.time()
    horizon = args.horizon
    horizon_cfg = load_config(f"exp_p04_h{horizon}.json")
    base_cfg = load_config("exp_p04_base.json")

    log_file = horizon_cfg["log_file"].replace(".log", "_final_train.log")
    logger = setup_logger("final_train", log_file)
    logger.info("=" * 60)
    logger.info("EXP-P04 最终训练  horizon=%d", horizon)

    results = run_all_models(horizon, horizon_cfg, base_cfg, logger)
    elapsed = time.time() - t0
    hs = f"h{horizon}"

    if "cnn_bilstm" not in results:
        raise RuntimeError("cnn_bilstm 最终训练未成功")

    m = results["cnn_bilstm"]
    summary = {
        "models_trained": list(results.keys()),
        "test_RMSE": round(m["RMSE"], 4),
        "test_MAE": round(m["MAE"], 4),
        "test_R2": round(m["R2"], 4),
        "training_time_sec": m.get("training_time_sec"),
        "elapsed_sec": round(elapsed, 1),
    }
    artifacts = [
        f"data/prediction/step4_optuna_hybrid/models/{hs}/cnn_bilstm_final.pt",
        f"data/prediction/step4_optuna_hybrid/metrics/{hs}/cnn_bilstm_test_metrics.json",
        f"data/prediction/step4_optuna_hybrid/predictions/{hs}/cnn_bilstm_test.csv",
    ]
    record_step_result(
        horizon, "train_final", "success", log_file,
        summary=summary, duration_sec=elapsed, artifacts=artifacts,
    )
    return horizon, log_file


if __name__ == "__main__":
    t0 = time.time()
    try:
        main()
    except Exception as e:
        record_step_failure("train_final", t0, e)
        raise
