"""
简化版参数搜索 - 仅使用随机搜索，避免 Optuna 问题
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.prediction.step5_new_experiments.exp_p05_common import (
    MODELS_DIR,
    METRICS_DIR,
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
from experiments.prediction.step4_optuna_hybrid.exp_p04_models import build_model
from experiments.prediction.step4_optuna_hybrid.exp_p04_torch_utils import (
    eval_loss,
    get_device,
    make_loader,
    train_with_early_stop,
)
from experiments.prediction.step5_new_experiments.exp_p05_benchmark import benchmark_forward


# ========== 残差预测辅助函数 ==========

def compute_residual_targets(y_raw: np.ndarray, y_last: np.ndarray) -> np.ndarray:
    """计算残差目标: residual = y_raw - y_last"""
    if y_last.ndim == 1:
        return y_raw - y_last[:, np.newaxis]
    return y_raw - y_last


def fit_residual_scaler(y_res: np.ndarray) -> dict:
    """拟合残差的标准化参数"""
    mean = float(np.mean(y_res))
    std = float(np.std(y_res))
    if std < 1e-6:
        std = 1.0
    return {"mean": mean, "std": std}


def transform_residual(scaler: dict, y_res: np.ndarray) -> np.ndarray:
    """标准化残差"""
    return (y_res - scaler["mean"]) / scaler["std"]


def inverse_transform_residual(scaler: dict, y_res_scaled: np.ndarray) -> np.ndarray:
    """反标准化残差"""
    return y_res_scaled * scaler["std"] + scaler["mean"]


def reconstruct_from_residual(y_last: np.ndarray, delta_pred: np.ndarray) -> np.ndarray:
    """从残差重构预测: y_pred = y_last + delta_pred"""
    if delta_pred.ndim == 2:
        return y_last[:, np.newaxis] + delta_pred
    return y_last + delta_pred


# ========== 搜索空间 ==========

SEARCH_SPACE = {
    "hidden": [32, 64, 128],
    "layers": [1, 2],
    "dropout": [0.1, 0.2, 0.3],
    "lr": [0.0005, 0.001, 0.002],
    "batch_size": [128, 256],
}


def normalize_model_params(params: dict, model_name: str) -> dict:
    """把通用搜索空间中的 hidden/layers 映射到各模型构造器的实际参数名"""
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


def train_and_score(
    params: dict,
    model_name: str,
    data: dict,
    meta: dict,
    horizon: int,
    max_epochs: int,
    patience: int,
    device: torch.device,
) -> dict:
    """训练并评估单个参数组合"""
    batch_size = int(params.pop("batch_size"))
    lr = float(params.pop("lr"))
    model_kwargs = normalize_model_params(params, model_name)
    
    model = build_model(
        model_name,
        n_features=meta["n_features"],
        seq_len=meta["lookback"],
        horizon=horizon,
        **model_kwargs,
    ).to(device)

    # 残差预测处理
    y_last_train = data["y_last_train"]
    y_last_val = data["y_last_val"]
    
    y_res_train = compute_residual_targets(data["y_train_raw"], y_last_train)
    y_res_val = compute_residual_targets(data["y_val_raw"], y_last_val)
    
    res_scaler = fit_residual_scaler(y_res_train)
    y_train_scaled = transform_residual(res_scaler, y_res_train)
    y_val_scaled = transform_residual(res_scaler, y_res_val)
    
    train_loader = make_loader(data["X_train_seq"], y_train_scaled, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(data["X_val_seq"], y_val_scaled, batch_size=batch_size, shuffle=False)

    # 训练
    model, _ = train_with_early_stop(
        model, train_loader, val_loader, lr=lr, max_epochs=max_epochs, patience=patience, device=device
    )
    
    # 验证集评估
    val_rmse = float(np.sqrt(eval_loss(model, val_loader, torch.nn.MSELoss(), device)))
    
    # 推理计时
    sample = torch.from_numpy(data["X_test_seq"][:512].astype(np.float32))
    bench = benchmark_forward(model, sample, device=device, warmup_iters=3, repeat_iters=10)
    
    return {
        "RMSE": val_rmse,
        "MAE": val_rmse,
        "latency_ms": bench["ms_per_sample"],
        "params": float(bench["params"]),
        "model_state": deepcopy(model.state_dict()),
        "train_params": {"batch_size": batch_size, "lr": lr, **params},
        "_res_scaler": res_scaler,
        "_y_last_val": y_last_val,
    }


def random_search(
    n_trials: int,
    model_name: str,
    data: dict,
    meta: dict,
    horizon: int,
    max_epochs: int,
    patience: int,
    seed: int,
) -> list[dict]:
    """随机搜索"""
    rng = random.Random(seed)
    device = get_device()
    rows = []
    
    for i in range(n_trials):
        params = {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}
        print(f"  Trial {i+1}/{n_trials}: {params}")
        try:
            result = train_and_score(deepcopy(params), model_name, data, meta, horizon, max_epochs, patience, device)
            rows.append(result)
            print(f"    RMSE={result['RMSE']:.4f}")
        except Exception as e:
            print(f"    Failed: {e}")
    
    return rows


def main():
    parser = argparse.ArgumentParser(description="简化版参数搜索")
    parser.add_argument("--horizon", type=int, choices=[1, 4, 16], required=True)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()
    
    cfg = load_config()
    ensure_dirs(MODELS_DIR / f"h{args.horizon}", PRED_DIR / f"h{args.horizon}")
    logger = setup_logger("random_search", f"EXP-P05_h{args.horizon}_random_search.log")
    
    samples = load_samples(args.horizon, use_step5=True)
    meta = load_meta(args.horizon, use_step5=True)
    model_name = cfg["hybrid_search"]["target_model"]
    
    logger.info("开始随机搜索: horizon=%d, trials=%d", args.horizon, args.trials)
    t0 = time.time()
    rows = random_search(
        args.trials, model_name, samples, meta, args.horizon,
        args.max_epochs, args.patience, seed=42
    )
    
    # 按 RMSE 排序
    rows.sort(key=lambda x: x["RMSE"])
    best = rows[0]
    
    logger.info("搜索完成，耗时 %.1fs", time.time() - t0)
    logger.info("最优参数: %s", best["train_params"])
    logger.info("最优 RMSE: %.4f", best["RMSE"])
    
    # 使用最优参数重新训练并保存
    logger.info("使用最优参数重新训练...")
    set_seed(42)
    
    # 准备残差数据
    y_last_train = samples["y_last_train"]
    y_last_val = samples["y_last_val"]
    y_last_test = samples["y_last_test"]
    
    y_res_train = compute_residual_targets(samples["y_train_raw"], y_last_train)
    y_res_val = compute_residual_targets(samples["y_val_raw"], y_last_val)
    res_scaler = fit_residual_scaler(y_res_train)
    
    y_train = transform_residual(res_scaler, y_res_train)
    y_val = transform_residual(res_scaler, y_res_val)
    
    # 构建模型
    train_params = best["train_params"]
    batch_size = int(train_params.get("batch_size", 128))
    lr = float(train_params.get("lr", 0.001))
    normalized = normalize_model_params(dict(train_params), model_name)
    
    model = build_model(
        model_name,
        n_features=meta["n_features"],
        seq_len=meta["lookback"],
        horizon=args.horizon,
        **normalized,
    ).to(get_device())
    
    train_loader = make_loader(samples["X_train_seq"], y_train, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(samples["X_val_seq"], y_val, batch_size=batch_size, shuffle=False)
    
    # 训练
    t0 = time.time()
    model, _ = train_with_early_stop(
        model, train_loader, val_loader,
        lr=lr, max_epochs=args.max_epochs, patience=args.patience, device=get_device()
    )
    train_time = time.time() - t0
    
    # 保存模型
    model_path = MODELS_DIR / f"h{args.horizon}" / f"{model_name}_best.pt"
    torch.save(model.state_dict(), model_path)
    
    # 预测
    from experiments.prediction.step4_optuna_hybrid.exp_p04_torch_utils import predict
    
    delta_scaled = predict(model, samples["X_test_seq"], get_device(), batch_size=batch_size)
    delta_pred = inverse_transform_residual(res_scaler, delta_scaled)
    y_pred = reconstruct_from_residual(y_last_test, delta_pred)
    
    # 计算指标
    y_true = samples["y_test_raw"]
    if args.horizon == 1:
        y_true_eval = y_true[:, 0]
        y_pred_eval = y_pred[:, 0]
    else:
        y_true_eval = y_true.ravel()
        y_pred_eval = y_pred.ravel()
    
    metrics = compute_all_metrics(y_true_eval, y_pred_eval)
    metrics["training_time_sec"] = train_time
    
    logger.info("最终指标: RMSE=%.4f, MAE=%.4f", metrics["RMSE"], metrics["MAE"])
    
    # 保存预测
    save_predictions(args.horizon, f"{model_name}_best", y_true_eval, y_pred_eval)
    
    # 保存结果
    result = {
        "horizon": args.horizon,
        "trials": len(rows),
        "best_params": best["train_params"],
        "search_rmse": best["RMSE"],
        "final_metrics": metrics,
        "model_path": str(model_path),
    }
    
    out = MODELS_DIR / f"h{args.horizon}" / "random_search_result.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    logger.info("结果已保存: %s", out)
    
    return result


if __name__ == "__main__":
    main()
