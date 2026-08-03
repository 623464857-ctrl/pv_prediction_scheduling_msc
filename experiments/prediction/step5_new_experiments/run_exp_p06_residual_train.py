"""
Step 6: 残差预测训练（读取 Optuna-AFSA 混合搜索最优参数）

在 Step6 搜索完成后，用最优超参数进行完整残差预测训练与评估。

执行方式：
    # 用最优策略参数训练（自动选择每个模型的最优策略）
    python experiments/prediction/step5_new_experiments/run_exp_p06_residual_train.py --horizon 1
    python experiments/prediction/step5_new_experiments/run_exp_p06_residual_train.py --horizon 4
    python experiments/prediction/step5_new_experiments/run_exp_p06_residual_train.py --horizon 16

    # 指定某个模型用特定策略参数训练
    python experiments/prediction/step5_new_experiments/run_exp_p06_residual_train.py --horizon 1 --model lstm --strategy S3

输出：
    data/prediction/step5_new_experiments/metrics/h{1,4,16}/residual_optuna_metrics.json
    data/prediction/step5_new_experiments/predictions/h{1,4,16}/{model}_residual_optuna_test.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import r2_score

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
from experiments.prediction.step5_new_experiments.exp_p05_benchmark import benchmark_forward
from experiments.prediction.step5_new_experiments.exp_p05_common import (
    METRICS_DIR,
    MODELS_DIR,
    PRED_DIR,
    RESIDUAL_MODELS,
    compute_all_metrics,
    ensure_dirs,
    load_config,
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


def compute_metrics_with_y_scale(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scale: float,
) -> dict:
    """
    同时返回原始尺度指标 + scaled space 指标。
    y_scale: scaler 的 y_scale (std of y before scaling)
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # 原始尺度
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mask = np.abs(y_true) > 0.01
    if mask.any():
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    else:
        mape = float("nan")
    r2 = float(r2_score(y_true, y_pred))

    # scaled space（除以 y_scale 后等价于在 scaled data 上评估）
    mae_s = mae / y_scale
    rmse_s = rmse / y_scale
    nrmse = rmse_s  # nRMSE = RMSE / σ = RMSE / y_scale
    r2_s = float(r2_score(y_true / y_scale, y_pred / y_scale))

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2,
        "nRMSE": nrmse,
        "y_scale": y_scale,
        "MAE_scaled": mae_s,
        "RMSE_scaled": rmse_s,
        "R2_scaled": r2_s,
    }


def load_config() -> dict:
    path = PROJECT_ROOT / "data" / "prediction" / "step5_new_experiments" / "config" / "exp_p06_config.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_search_results(horizon: int) -> dict:
    """读取 Step6 混合搜索结果。"""
    path = METRICS_DIR / f"h{horizon}" / "hybrid_search_full.json"
    if not path.exists():
        raise FileNotFoundError(f"搜索结果不存在，请先运行 Step6 混合搜索：{path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_best_params_for_model(search_results: dict, model_name: str) -> dict:
    """
    从搜索结果中选出最优策略（3-fold 评估 val_RMSE 最低）的参数。
    对齐 Step4 策略：以 3-fold 完整评估结果为最终判断依据。
    """
    candidates = []
    for key, entry in search_results.items():
        if entry.get("model") != model_name:
            continue
        if "error" in entry:
            continue
        full_eval = entry.get("full_eval", {})
        quick_best = entry.get("quick_best", {})
        candidates.append({
            "strategy": entry.get("strategy"),
            "model": model_name,
            "params": quick_best.get("train_params", {}),
            # 3-fold 评估结果（用于最终选优，对齐 Step4）
            "fold_val_rmses": full_eval.get("fold_val_rmses", []),
            "avg_val_RMSE": full_eval.get("avg_val_RMSE", float("inf")),
            "avg_test_RMSE": full_eval.get("avg_test_RMSE", float("inf")),
            # quick subset 结果（辅助参考）
            "quick_RMSE": quick_best.get("RMSE", float("inf")),
            "composite_score": quick_best.get("composite_score", float("inf")),
        })

    if not candidates:
        raise ValueError(f"模型 {model_name} 在搜索结果中未找到有效记录")

    # 按 3-fold 评估 val_RMSE 排序（primary，对齐 Step4）+ composite_score 细排
    candidates.sort(key=lambda x: (x["avg_val_RMSE"], x["composite_score"]))
    chosen = candidates[0]
    return {
        "strategy": chosen["strategy"],
        "params": chosen["params"],
        "avg_val_RMSE": chosen["avg_val_RMSE"],
        "avg_test_RMSE": chosen["avg_test_RMSE"],
        "fold_val_rmses": chosen["fold_val_rmses"],
        "quick_RMSE": chosen["quick_RMSE"],
        "composite_score": chosen["composite_score"],
    }


def train_one_model_residual(
    model_name: str,
    horizon: int,
    cfg: dict,
    best_params: dict,
    logger,
) -> dict:
    """
    用搜索得到的最优参数训练单个残差预测模型。
    best_params 格式: {"strategy": "S3", "params": {hidden, layers, dropout, lr, batch_size, ...}, ...}
    """
    from experiments.prediction.step5_new_experiments.exp_p05_common import SAMPLES_DIR

    hdir = SAMPLES_DIR / f"h{horizon}"

    # 加载样本
    X_train = np.load(hdir / "X_train_seq.npy")
    X_val = np.load(hdir / "X_val_seq.npy")
    X_test = np.load(hdir / "X_test_seq.npy")
    y_train_raw = np.load(hdir / "y_train_raw.npy")
    y_val_raw = np.load(hdir / "y_val_raw.npy")
    y_test_raw = np.load(hdir / "y_test_raw.npy")
    y_last_train = np.load(hdir / "y_last_train.npy")
    y_last_val = np.load(hdir / "y_last_val.npy")
    y_last_test = np.load(hdir / "y_last_test.npy")

    with open(hdir / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)

    set_seed(cfg["residual_train"]["seed"])
    device = get_device()

    # 残差目标变换
    y_res_train = compute_residual_targets(y_train_raw, y_last_train)
    y_res_val = compute_residual_targets(y_val_raw, y_last_val)
    res_scaler = fit_residual_scaler(y_res_train)

    scaler_path = hdir / "residual_scaler_optuna_params.json"
    save_residual_scaler(res_scaler, scaler_path)

    y_train = transform_residual(res_scaler, y_res_train)
    y_val = transform_residual(res_scaler, y_res_val)

    # 构建模型：用搜索参数
    train_params = best_params["params"]
    batch_size = int(train_params.pop("batch_size"))
    lr = float(train_params.pop("lr"))

    model_kwargs = _normalize_params(train_params, model_name)
    model = build_model(
        model_name,
        n_features=meta["n_features"],
        seq_len=meta["lookback"],
        horizon=horizon,
        **model_kwargs,
    ).to(device)

    train_loader = make_loader(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, batch_size=batch_size, shuffle=False)

    t0 = time.time()
    model, history = train_with_early_stop(
        model,
        train_loader,
        val_loader,
        lr=lr,
        max_epochs=cfg["residual_train"]["max_epochs"],
        patience=cfg["residual_train"]["patience"],
        device=device,
    )
    train_time = time.time() - t0

    # 保存模型
    model_path = MODELS_DIR / f"h{horizon}" / f"{model_name}_residual_optuna.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)

    # 测试集评估
    delta_scaled = predict(model, X_test, device, batch_size=batch_size)
    delta_pred = inverse_transform_residual(res_scaler, delta_scaled)
    y_pred = reconstruct_from_residual(y_last_test, delta_pred)

    if horizon == 1:
        y_true_eval = y_test_raw[:, 0]
        y_pred_eval = y_pred[:, 0]
    else:
        y_true_eval = y_test_raw.ravel()
        y_pred_eval = y_pred.ravel()

    metrics = compute_all_metrics(y_true_eval, y_pred_eval)

    # 使用 scaler 的 y_scale（σ）计算一致的 nRMSE 和 scaled-space R²
    y_scale = res_scaler.scale_[0] if res_scaler.scale_.ndim == 1 else res_scaler.scale_[0, 0]
    metrics["y_scale"] = float(y_scale)
    metrics["nRMSE"] = float(metrics["RMSE"] / y_scale)  # 一致的 nRMSE = RMSE / σ
    y_true_scaled = (y_true_eval - res_scaler.mean_[0]) / y_scale
    y_pred_scaled = (y_pred_eval - res_scaler.mean_[0]) / y_scale
    metrics["R2_scaled"] = float(r2_score(y_true_scaled, y_pred_scaled))  # scaled-space R²

    metrics["training_time_sec"] = train_time
    metrics["search_strategy"] = best_params["strategy"]
    metrics["search_quick_val_rmse"] = best_params.get("quick_RMSE", -1)
    metrics["search_full_val_rmse"] = best_params.get("avg_val_RMSE", -1)
    metrics["search_full_test_rmse"] = best_params.get("avg_test_RMSE", -1)

    # 推理效率
    sample = torch.from_numpy(X_test[:512].astype(np.float32))
    bench = benchmark_forward(model, sample, device=device, warmup_iters=10, repeat_iters=100)
    metrics["inference_ms_per_sample"] = bench["ms_per_sample"]
    metrics["params"] = bench["params"]

    save_predictions(horizon, f"{model_name}_residual_optuna", y_true_eval, y_pred_eval)

    logger.info(
        "%s (optuna) RMSE=%.4f nRMSE=%.4f R2=%.4f R2_scaled=%.4f | 搜索策略=%s  3fold_Val=%.4f  3fold_Test=%.4f",
        model_name, metrics["RMSE"], metrics["nRMSE"], metrics["R2"], metrics["R2_scaled"],
        best_params["strategy"], best_params.get("avg_val_RMSE", -1), best_params.get("avg_test_RMSE", -1),
    )
    return metrics


def _normalize_params(params: dict, model_name: str) -> dict:
    """将搜索参数的通用命名映射到各模型构造器实际参数名。"""
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


def main():
    parser = argparse.ArgumentParser(description="EXP-P06 残差预测训练（读取搜索最优参数）")
    parser.add_argument("--horizon", type=int, choices=[1, 4, 16], required=True)
    parser.add_argument("--model", type=str, choices=RESIDUAL_MODELS, default=None,
                        help="指定模型（默认全部 5 个）")
    parser.add_argument("--strategy", type=str, choices=["S2", "S3", "S4", "S5", "S6"], default=None,
                        help="强制使用特定策略的参数（默认自动选择最优）")
    args = parser.parse_args()

    cfg = load_config()
    ensure_dirs(MODELS_DIR / f"h{args.horizon}", METRICS_DIR / f"h{args.horizon}", PRED_DIR / f"h{args.horizon}")
    logger = setup_logger("residual_optuna_p06", f"EXP-P06_h{args.horizon}_residual_train.log")

    search_results = load_search_results(args.horizon)
    models = [args.model] if args.model else RESIDUAL_MODELS

    all_metrics = {}
    for m in models:
        if m not in RESIDUAL_MODELS:
            logger.warning("跳过未知模型: %s", m)
            continue

        try:
            if args.strategy:
                # 指定策略：直接取该策略的参数
                key = f"{m}_{args.strategy}"
                entry = search_results.get(key, {})
                if "error" in entry or "quick_best" not in entry:
                    logger.error("策略 %s 在搜索结果中未找到模型 %s", args.strategy, m)
                    continue
                quick_best = entry.get("quick_best", {})
                full_eval = entry.get("full_eval", {})
                best_params = {
                    "strategy": args.strategy,
                    "params": quick_best.get("train_params", {}),
                    "quick_RMSE": quick_best.get("RMSE", float("inf")),
                    "avg_val_RMSE": full_eval.get("avg_val_RMSE", float("inf")),
                    "avg_test_RMSE": full_eval.get("avg_test_RMSE", float("inf")),
                    "composite_score": quick_best.get("composite_score", float("inf")),
                }
            else:
                # 自动选择最优策略
                best_params = get_best_params_for_model(search_results, m)

            all_metrics[f"{m}_residual_optuna"] = train_one_model_residual(
                m, args.horizon, cfg, best_params, logger
            )
        except Exception as e:
            logger.error("模型 %s 训练失败: %s", m, e, exc_info=True)
            all_metrics[m] = {"error": str(e)}

    out = METRICS_DIR / f"h{args.horizon}" / "residual_optuna_metrics.json"
    out.write_text(json.dumps(all_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("残差训练完成，结果已保存: %s", out.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
