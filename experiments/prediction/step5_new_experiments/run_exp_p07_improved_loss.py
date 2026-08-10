"""
Step 7: 改进损失函数训练（Phase 1 & Phase 2）

Phase 1 改进：
1. 非对称MSE损失 - 鼓励预测更高峰值
2. 物理约束 - 夜间归零、辐照度上限
3. 分段加权损失 - 峰值和夜间区域权重增加

Phase 2 改进（针对峰值过冲和日落下降问题）：
1. 移除 peak_weight - 避免峰值区域过冲
2. Huber损失替代asymmetric_mse - 对异常值更鲁棒
3. 日落单调性约束 - 改善日落下降段预测
4. 物理先验单调性正则化

执行方式：
    # Phase 1: 非对称MSE
    python experiments/prediction/step5_new_experiments/run_exp_p07_improved_loss.py --horizon 1 --loss asymmetric_mse

    # Phase 1: 组合损失
    python experiments/prediction/step5_new_experiments/run_exp_p07_improved_loss.py --horizon 1 --loss combined

    # Phase 2: Huber损失
    python experiments/prediction/step5_new_experiments/run_exp_p07_improved_loss.py --horizon 1 --loss huber --delta 1.0

    # Phase 2: 组合V2（无peak_weight，含日落约束）
    python experiments/prediction/step5_new_experiments/run_exp_p07_improved_loss.py --horizon 1 --loss combined_v2

输出：
    data/prediction/step5_new_experiments/metrics/h{1,4,16}/improved_loss_{loss_name}_metrics.json
    data/prediction/step5_new_experiments/predictions/h{1,4,16}/{model}_improved_{loss_name}_test.csv
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
    get_device,
    make_loader,
    predict,
)
from experiments.prediction.step5_new_experiments.exp_p05_benchmark import benchmark_forward
from experiments.prediction.step5_new_experiments.exp_p05_common import (
    METRICS_DIR,
    MODELS_DIR,
    PRED_DIR,
    RESIDUAL_MODELS,
    SAMPLES_DIR,
    compute_all_metrics,
    ensure_dirs,
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
from experiments.prediction.step5_new_experiments.exp_p06_losses import (
    CombinedPeakLoss,
    CombinedV2Loss,
    AsymmetricMSELoss,
    AsymmetricHuberLoss,
    HuberLoss,
    QuantileWeightedLoss,
    nighttime_zero_constraint,
    irradiance_upper_bound,
    train_with_early_stop_constrained,
    get_device as get_device_from_losses,
)


# ============================================================================
# 配置加载
# ============================================================================

def load_config() -> dict:
    path = PROJECT_ROOT / "data" / "prediction" / "step5_new_experiments" / "config" / "exp_p06_config.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_search_results(horizon: int) -> dict:
    """读取 Step6 混合搜索结果获取最优参数。"""
    path = METRICS_DIR / f"h{horizon}" / "hybrid_search_full.json"
    if not path.exists():
        raise FileNotFoundError(f"搜索结果不存在：{path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_best_params_for_model(search_results: dict, model_name: str) -> dict:
    """从搜索结果中选出最优策略参数。如果找不到，使用默认参数。"""
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
            "params": quick_best.get("train_params", {}),
            "avg_val_RMSE": full_eval.get("avg_val_RMSE", float("inf")),
            "avg_test_RMSE": full_eval.get("avg_test_RMSE", float("inf")),
        })

    if not candidates:
        # 如果搜索结果中没有，使用默认参数
        return {
            "strategy": "default",
            "params": {
                "hidden": 64,
                "layers": 2,
                "dropout": 0.2,
                "batch_size": 256,
                "lr": 0.001,
            },
            "avg_val_RMSE": float("inf"),
            "avg_test_RMSE": float("inf"),
        }

    candidates.sort(key=lambda x: x["avg_val_RMSE"])
    return candidates[0]


def _normalize_params(params: dict, model_name: str) -> dict:
    """将搜索参数的通用命名映射到各模型构造器实际参数名。"""
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


# ============================================================================
# 指标计算
# ============================================================================

def compute_metrics_with_y_scale(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scale: float,
) -> dict:
    """计算原始尺度和scaled space指标。"""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mask = np.abs(y_true) > 0.01
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else float("nan")
    r2 = float(r2_score(y_true, y_pred))

    mae_s = mae / y_scale
    rmse_s = rmse / y_scale
    r2_s = float(r2_score(y_true / y_scale, y_pred / y_scale))

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2,
        "nRMSE": rmse_s,
        "y_scale": y_scale,
        "MAE_scaled": mae_s,
        "RMSE_scaled": rmse_s,
        "R2_scaled": r2_s,
    }


def compute_segmented_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    capacity: float = 1.0,
) -> dict:
    """计算分段指标：全天、白天、高峰值、低功率。"""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # 全天
    all_metrics = compute_all_metrics(y_true, y_pred, capacity=capacity)

    # 高峰值 (top 20%)
    peak_threshold = np.percentile(y_true, 80)
    peak_mask = y_true >= peak_threshold
    peak_metrics = compute_all_metrics(y_true[peak_mask], y_pred[peak_mask], capacity=capacity) if peak_mask.any() else {}

    # 低功率/夜间 (bottom 5%)
    low_threshold = np.percentile(y_true, 5)
    low_mask = y_true <= low_threshold
    low_metrics = compute_all_metrics(y_true[low_mask], y_pred[low_mask], capacity=capacity) if low_mask.any() else {}

    # 中间区域
    mid_mask = ~peak_mask & ~low_mask
    mid_metrics = compute_all_metrics(y_true[mid_mask], y_pred[mid_mask], capacity=capacity) if mid_mask.any() else {}

    return {
        "all": all_metrics,
        "peak": peak_metrics,
        "low_power": low_metrics,
        "mid": mid_metrics,
    }


# ============================================================================
# 主训练函数
# ============================================================================

def train_with_improved_loss(
    model_name: str,
    horizon: int,
    cfg: dict,
    loss_type: str = "asymmetric_mse",
    loss_params: dict | None = None,
    apply_physics: bool = True,
    logger=None,
) -> dict:
    """
    使用改进损失函数训练单个残差预测模型。

    Args:
        model_name: 模型名称
        horizon: 预测步长
        cfg: 配置字典
        loss_type: 损失函数类型
        loss_params: 损失函数参数
        apply_physics: 是否应用物理约束
        logger: 日志记录器
    """
    loss_params = loss_params or {}
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

    # 加载辐照度特征用于物理约束 (daylight_flag 在索引6)
    daylight_flag_train = X_train[:, -1, 6]  # 最后一个时间步的daylight_flag
    daylight_flag_val = X_val[:, -1, 6]
    daylight_flag_test = X_test[:, -1, 6]

    with open(hdir / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)

    set_seed(cfg["residual_train"]["seed"])
    device = get_device()

    # 残差目标变换
    y_res_train = compute_residual_targets(y_train_raw, y_last_train)
    y_res_val = compute_residual_targets(y_val_raw, y_last_val)
    res_scaler = fit_residual_scaler(y_res_train)

    scaler_path = hdir / f"residual_scaler_{loss_type}_params.json"
    save_residual_scaler(res_scaler, scaler_path)

    y_train = transform_residual(res_scaler, y_res_train)
    y_val = transform_residual(res_scaler, y_res_val)

    # 加载搜索参数
    search_results = load_search_results(horizon)
    best_params = get_best_params_for_model(search_results, model_name)
    train_params = best_params["params"]
    batch_size = int(train_params.pop("batch_size"))
    lr = float(train_params.pop("lr"))

    # 构建模型
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

    # 创建损失函数
    if loss_type == "asymmetric_mse":
        alpha = loss_params.get("alpha", 0.5)
        criterion = AsymmetricMSELoss(alpha=alpha)
        logger.info(f"使用非对称MSE损失，alpha={alpha}")
    elif loss_type == "asymmetric_huber":
        alpha = loss_params.get("alpha", 0.5)
        delta = loss_params.get("delta", 1.0)
        criterion = AsymmetricHuberLoss(alpha=alpha, delta=delta)
        logger.info(f"使用非对称Huber损失，alpha={alpha}, delta={delta}")
    elif loss_type == "quantile_weighted":
        criterion = QuantileWeightedLoss(
            peak_weight=loss_params.get("peak_weight", 2.0),
            night_weight=loss_params.get("night_weight", 3.0),
        )
        logger.info("使用分段加权损失")
    elif loss_type == "combined":
        criterion = CombinedPeakLoss(
            alpha=loss_params.get("alpha", 0.5),
            peak_weight=loss_params.get("peak_weight", 2.0),
            night_weight=loss_params.get("night_weight", 3.0),
            smoothness_weight=loss_params.get("smoothness_weight", 0.01),
        )
        logger.info("使用组合损失")
    elif loss_type == "huber":
        # Phase 2: 标准Huber损失，对异常值鲁棒，避免峰值过冲
        delta = loss_params.get("huber_delta", 0.1)
        criterion = HuberLoss(delta=delta)
        logger.info(f"使用Huber损失，delta={delta}")
    elif loss_type == "combined_v2":
        # Phase 2: 无peak_weight，含日落单调性约束
        criterion = CombinedV2Loss(
            huber_delta=loss_params.get("huber_delta", 0.1),
            smoothness_weight=loss_params.get("smoothness_weight", 0.05),
            sunset_weight=loss_params.get("sunset_weight", 0.1),
            night_weight=loss_params.get("night_weight", 0.0),
        )
        logger.info("使用Phase 2组合损失（无peak_weight，含日落约束）")
    else:
        criterion = torch.nn.MSELoss()
        logger.warning(f"未知损失函数类型 {loss_type}，使用标准MSE")

    # 训练
    t0 = time.time()
    model, history = train_with_early_stop_constrained(
        model,
        train_loader,
        val_loader,
        criterion=criterion,
        lr=lr,
        max_epochs=cfg["residual_train"]["max_epochs"],
        patience=cfg["residual_train"]["patience"],
        device=device,
        apply_physics=apply_physics,
        capacity=cfg.get("capacity_pu", 1.0),
    )
    train_time = time.time() - t0

    # 保存模型
    model_path = MODELS_DIR / f"h{horizon}" / f"{model_name}_{loss_type}_improved.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)

    # 推理预测
    delta_scaled = predict(model, X_test, device, batch_size=batch_size)
    delta_pred = inverse_transform_residual(res_scaler, delta_scaled)
    y_pred = reconstruct_from_residual(y_last_test, delta_pred)

    # 从配置获取容量
    capacity = cfg.get("capacity_pu", 1.0)

    # 应用后处理约束
    if apply_physics:
        # 夜间归零（使用 daylight_flag）
        y_pred = nighttime_zero_constraint(
            y_pred,
            daylight_flag=daylight_flag_test,
        )
        # 辐照度上限（使用归一化辐照度）
        irradiance_test = X_test[:, -1, 0]  # total_irradiance (归一化)
        y_pred = irradiance_upper_bound(
            y_pred,
            irradiance=irradiance_test,
            capacity=capacity,
        )

    if horizon == 1:
        y_true_eval = y_test_raw[:, 0]
        y_pred_eval = y_pred[:, 0]
    else:
        y_true_eval = y_test_raw.ravel()
        y_pred_eval = y_pred.ravel()

    # 计算指标
    y_scale = res_scaler.scale_[0] if res_scaler.scale_.ndim == 1 else res_scaler.scale_[0, 0]
    metrics = compute_metrics_with_y_scale(y_true_eval, y_pred_eval, y_scale)

    # 分段指标
    segmented = compute_segmented_metrics(y_true_eval, y_pred_eval, capacity=capacity)
    metrics["segmented"] = segmented

    metrics["training_time_sec"] = train_time
    metrics["loss_type"] = loss_type
    metrics["loss_params"] = loss_params
    metrics["apply_physics"] = apply_physics
    metrics["search_strategy"] = best_params["strategy"]

    # 推理效率
    sample = torch.from_numpy(X_test[:512].astype(np.float32))
    bench = benchmark_forward(model, sample, device=device, warmup_iters=10, repeat_iters=100)
    metrics["inference_ms_per_sample"] = bench["ms_per_sample"]
    metrics["params"] = bench["params"]

    # 保存预测
    save_predictions(horizon, f"{model_name}_{loss_type}_improved", y_true_eval, y_pred_eval)

    logger.info(
        "%s (%s) RMSE=%.4f nRMSE=%.4f | Peak RMSE=%.4f | Low RMSE=%.4f | 训练时间=%.1fs",
        model_name, loss_type,
        metrics["RMSE"], metrics["nRMSE"],
        segmented.get("peak", {}).get("RMSE", float("nan")),
        segmented.get("low_power", {}).get("RMSE", float("nan")),
        train_time,
    )

    return metrics


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="EXP-P07 改进损失函数训练 (Phase 1 & Phase 2)"
    )
    parser.add_argument("--horizon", type=int, choices=[1, 4, 16], required=True,
                        help="预测步长")
    parser.add_argument("--model", type=str, choices=RESIDUAL_MODELS, default=None,
                        help="指定模型（默认全部 5 个）")
    parser.add_argument("--loss", type=str,
                        choices=["asymmetric_mse", "asymmetric_huber", "quantile_weighted",
                                 "combined", "mse", "huber", "combined_v2"],
                        default="asymmetric_mse",
                        help="损失函数类型")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="非对称系数（低估惩罚倍数），Phase 1 专属")
    parser.add_argument("--delta", type=float, default=1.0,
                        help="Huber损失的delta参数，控制鲁棒区域")
    parser.add_argument("--peak-weight", type=float, default=2.0,
                        help="峰值区域权重倍数（Phase 1，Phase 2 已移除）")
    parser.add_argument("--night-weight", type=float, default=0.0,
                        help="夜间区域加权倍数（建议0.0，在原始功率空间应用）")
    parser.add_argument("--huber-delta", type=float, default=0.1,
                        help="CombinedV2和huber中Huber损失的delta参数（建议0.05~0.3）")
    parser.add_argument("--smoothness-weight", type=float, default=0.05,
                        help="平滑正则化权重（CombinedV2）")
    parser.add_argument("--sunset-weight", type=float, default=0.1,
                        help="日落单调性约束权重（CombinedV2）")
    parser.add_argument("--no-physics", action="store_true",
                        help="禁用物理约束")
    args = parser.parse_args()

    cfg = load_config()
    ensure_dirs(MODELS_DIR / f"h{args.horizon}", METRICS_DIR / f"h{args.horizon}", PRED_DIR / f"h{args.horizon}")

    loss_suffix = args.loss
    logger_name = f"improved_loss_{loss_suffix}"
    logger = setup_logger(logger_name, f"EXP-P07_h{args.horizon}_{loss_suffix}.log")

    models = [args.model] if args.model else RESIDUAL_MODELS
    loss_params = {
        "alpha": args.alpha,
        "delta": args.delta,
        "peak_weight": args.peak_weight,
        "night_weight": args.night_weight,
        "huber_delta": args.huber_delta,
        "smoothness_weight": args.smoothness_weight,
        "sunset_weight": args.sunset_weight,
    }

    logger.info("=" * 60)
    logger.info("Phase 1 & Phase 2 改进损失函数训练")
    logger.info("损失函数: %s", args.loss)
    if args.loss in ("asymmetric_mse", "asymmetric_huber"):
        logger.info("非对称系数 alpha: %.2f, delta: %.2f", args.alpha, args.delta)
    elif args.loss == "huber":
        logger.info("Huber delta: %.2f", args.huber_delta)
    elif args.loss == "combined_v2":
        logger.info("Huber delta: %.2f, 平滑权重: %.3f, 日落权重: %.3f, 夜间权重: %.1f",
                    args.huber_delta, args.smoothness_weight, args.sunset_weight, args.night_weight)
    elif args.loss == "quantile_weighted":
        logger.info("峰值权重: %.1f, 夜间权重: %.1f", args.peak_weight, args.night_weight)
    logger.info("物理约束: %s", "启用" if not args.no_physics else "禁用")
    logger.info("=" * 60)

    all_metrics = {}
    for m in models:
        if m not in RESIDUAL_MODELS:
            logger.warning("跳过未知模型: %s", m)
            continue

        try:
            metrics = train_with_improved_loss(
                model_name=m,
                horizon=args.horizon,
                cfg=cfg,
                loss_type=args.loss,
                loss_params=loss_params,
                apply_physics=not args.no_physics,
                logger=logger,
            )
            all_metrics[f"{m}_{args.loss}_improved"] = metrics
        except Exception as e:
            logger.error("模型 %s 训练失败: %s", m, e, exc_info=True)
            all_metrics[m] = {"error": str(e)}

    # 保存结果
    out = METRICS_DIR / f"h{args.horizon}" / f"improved_loss_{loss_suffix}_metrics.json"
    out.write_text(json.dumps(all_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("训练完成，结果已保存: %s", out.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
