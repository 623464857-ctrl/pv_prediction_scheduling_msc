"""
明月湖数据集多 Seed 复现脚本
python -m experiments.prediction.step3_deep_learning.run_exp_p04_reproduce_mingyuehu --horizon 1
用多个 seed 重复最终训练，统计均值和标准差。
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.prediction.step2_hyperparameter_search.exp_p04_common import (
    METRICS_DIR,
    MODELS_DIR,
    PRED_DIR,
    PROJECT_ROOT,
    compute_all_metrics,
    save_predictions,
    save_train_history,
    setup_logger,
)
from experiments.prediction.step5_reporting.exp_p04_step_audit import (
    record_step_failure,
    record_step_result,
)
from experiments.prediction.step3_deep_learning.exp_p04_models import build_model
from experiments.prediction.step3_deep_learning.exp_p04_torch_utils import (
    get_device,
    make_loader,
    predict,
    train_with_early_stop,
)

# =============================================================================
# 明月湖配置
# =============================================================================
MINGYUEHU_CFG = {
    "final_max_epochs": 50,
    "final_patience": 8,
    "final_lr": 0.001,
    "reproduce_seeds": [42, 43, 44, 45, 46],
}

HORIZON_CONFIGS = {
    1: {"horizon": 1, "lookback": 16},
    4: {"horizon": 4, "lookback": 48},
    16: {"horizon": 16, "lookback": 96},
}


def load_mingyuehu_sample_dir(horizon: int, lookback: int = None) -> Path:
    """返回明月湖样本目录"""
    if lookback is None:
        lookback = HORIZON_CONFIGS[horizon]["lookback"]
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


def load_mingyuehu_test_timestamps(horizon: int, lookback: int = None) -> pd.Series:
    """加载明月湖测试集时间戳"""
    hdir = load_mingyuehu_sample_dir(horizon, lookback)
    return pd.read_csv(hdir / "test_timestamps.csv", parse_dates=["timestamp"])["timestamp"]


def save_mingyuehu_predictions(horizon: int, model_name: str, y_true: np.ndarray,
                                y_pred: np.ndarray, lookback: int = None) -> Path:
    """保存明月湖预测结果到指定目录"""
    ts = load_mingyuehu_test_timestamps(horizon, lookback)
    y_true_flat = np.asarray(y_true).ravel()
    y_pred_flat = np.asarray(y_pred).ravel()
    n_flat = len(y_true_flat)
    n_ts = len(ts)

    if n_flat != n_ts:
        # horizon > 1: repeat each timestamp for each horizon step
        ts_vals = np.repeat(ts.values, horizon)
        ts_vals = ts_vals[:n_flat]
    else:
        ts_vals = ts.values

    out = pd.DataFrame({
        "timestamp": ts_vals,
        "y_true": y_true_flat,
        "y_pred": y_pred_flat,
        "model_name": model_name,
    })
    path = PRED_DIR / f"mingyuehu_h{horizon}" / f"{model_name}_test.csv"
    out.to_csv(path, index=False)
    return path


def _best_params(params: dict, model_name: str) -> dict:
    """解析最优参数，兼容从 optuna JSON 读取的参数"""
    out = {}
    for k, v in params.items():
        if k in {"batch_size", "lr"}:
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


def run_reproduce(horizon: int, logger) -> dict:
    """多 seed 复现"""
    lookback = HORIZON_CONFIGS[horizon]["lookback"]
    hdir = load_mingyuehu_sample_dir(horizon, lookback)

    metrics_h = METRICS_DIR / f"mingyuehu_h{horizon}"
    models_h = MODELS_DIR / f"mingyuehu_h{horizon}"
    pred_h = PRED_DIR / f"mingyuehu_h{horizon}"
    for d in (metrics_h, models_h, pred_h):
        d.mkdir(parents=True, exist_ok=True)

    # 加载样本
    X_train = np.load(hdir / "X_train_seq.npy")
    y_residual_train = np.load(hdir / "y_train.npy")
    X_val = np.load(hdir / "X_val_seq.npy")
    y_residual_val = np.load(hdir / "y_val.npy")
    X_test = np.load(hdir / "X_test_seq.npy")
    y_residual_test = np.load(hdir / "y_test.npy")
    y_anchor_test = np.load(hdir / "y_anchor_test.npy")
    y_scaler = load_mingyuehu_y_scaler(horizon, lookback)

    meta = json.loads((hdir / "meta.json").read_text(encoding="utf-8"))
    seq_len, n_features = meta["lookback"], X_train.shape[2]
    device = get_device()

    seeds = MINGYUEHU_CFG["reproduce_seeds"]
    max_epochs = MINGYUEHU_CFG["final_max_epochs"]
    patience = MINGYUEHU_CFG["final_patience"]
    default_lr = MINGYUEHU_CFG["final_lr"]

    logger.info("=" * 60)
    logger.info("明月湖多 Seed 复现  horizon=%d  seeds=%s", horizon, seeds)

    # 加载 Optuna 结果
    optuna_path = metrics_h / "mingyuehu_cnn_bilstm_optuna.json"
    if not optuna_path.exists():
        raise FileNotFoundError(f"缺少 Optuna 结果: {optuna_path}")

    params_json = json.loads(optuna_path.read_text(encoding="utf-8"))
    params = params_json["best_params"]
    batch_size = int(params.pop("batch_size", 64))
    lr_use = float(params.pop("lr", default_lr))

    mname = "cnn_bilstm"
    logger.info("-" * 40)
    logger.info("模型: %s  参数: %s", mname, {**params, "batch_size": batch_size, "lr": lr_use})

    all_metrics = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = build_model(
            mname,
            n_features=n_features,
            seq_len=seq_len,
            horizon=horizon,
            **_best_params(params, mname),
        ).to(device)

        train_loader = make_loader(X_train, y_residual_train, batch_size=batch_size, shuffle=True)
        val_loader = make_loader(X_val, y_residual_val, batch_size=batch_size, shuffle=False)

        t0 = time.time()
        model, _ = train_with_early_stop(
            model, train_loader, val_loader,
            lr=lr_use, max_epochs=max_epochs, patience=patience, device=device,
        )
        elapsed = time.time() - t0

        # 残差预测 → 重构功率
        y_pred_residual_scaled = predict(model, X_test, device)
        y_pred_residual = y_scaler.inverse_transform(y_pred_residual_scaled)
        y_pred_power = (y_anchor_test + y_pred_residual).astype(np.float32)
        # 真实功率
        y_test_residual_raw = y_scaler.inverse_transform(y_residual_test)
        y_true_power = (y_anchor_test + y_test_residual_raw).astype(np.float32)
        # 计算指标
        metrics = compute_all_metrics(y_true_power.ravel(), y_pred_power.ravel())
        metrics["seed"] = seed
        metrics["training_time_sec"] = round(elapsed, 2)
        all_metrics.append(metrics)

        logger.info("  seed=%d  MAE=%.4f  RMSE=%.4f  R2=%.4f  time=%.1fs",
                    seed, metrics["MAE"], metrics["RMSE"], metrics["R2"], elapsed)

    # 汇总统计
    rows_df = pd.DataFrame(all_metrics)
    mean_row = rows_df[["MAE", "RMSE", "MAPE", "R2", "training_time_sec"]].mean().to_dict()
    std_row = rows_df[["MAE", "RMSE", "MAPE", "R2"]].std().to_dict()

    summary = {
        "model": mname,
        "dataset": "mingyuehu",
        "horizon": horizon,
        "lookback": lookback,
        "seeds": seeds,
        "mean": {k: round(v, 6) for k, v in mean_row.items()},
        "std": {k: round(v, 6) for k, v in std_row.items()},
        "per_seed": all_metrics,
        "prediction_mode": "residual",
    }

    out_path = metrics_h / f"mingyuehu_cnn_bilstm_reproduce.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("  汇总  MAE=%.4f±%.4f  RMSE=%.4f±%.4f  R2=%.4f±%.4f",
                mean_row["MAE"], std_row["MAE"],
                mean_row["RMSE"], std_row["RMSE"],
                mean_row["R2"], std_row["R2"])

    # 保存 seed=42 的预测（用于绘图）
    torch.manual_seed(42)
    np.random.seed(42)
    model_seed = build_model(
        mname, n_features=n_features, seq_len=seq_len,
        horizon=horizon, **_best_params(params, mname),
    ).to(device)
    train_loader = make_loader(X_train, y_residual_train, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_residual_val, batch_size=batch_size, shuffle=False)
    model_seed, _ = train_with_early_stop(
        model_seed, train_loader, val_loader,
        lr=lr_use, max_epochs=max_epochs, patience=patience, device=device,
    )
    # 残差预测 → 重构功率
    y_pred_residual_scaled = predict(model_seed, X_test, device)
    y_pred_residual = y_scaler.inverse_transform(y_pred_residual_scaled)
    y_pred_power_seed = (y_anchor_test + y_pred_residual).astype(np.float32)
    # 真实功率
    y_test_residual_raw = y_scaler.inverse_transform(y_residual_test)
    y_true_power_seed = (y_anchor_test + y_test_residual_raw).astype(np.float32)
    pred_path = save_mingyuehu_predictions(horizon, f"{mname}_seed42",
                                          y_true_power_seed.ravel(), y_pred_power_seed.ravel(),
                                          lookback)
    logger.info("  seed=42 预测已保存: %s", pred_path.name)

    model_path = models_h / f"mingyuehu_cnn_bilstm_seed42.pt"
    torch.save(model_seed.state_dict(), model_path)
    logger.info("  seed=42 模型已保存: %s", model_path.name)

    # 保存训练历史
    hist_path = save_train_history(f"mingyuehu_h{horizon}", f"mingyuehu_{mname}_seed42", [])
    logger.info("  训练历史路径: %s", hist_path.name)

    logger.info("=" * 60)
    logger.info("明月湖多 Seed 复现完成！")
    return summary


def main():
    parser = argparse.ArgumentParser(description="明月湖多 Seed 复现")
    parser.add_argument("--horizon", type=int, choices=[1, 4, 16], required=True)
    args = parser.parse_args()

    t0 = time.time()
    horizon = args.horizon

    log_file = f"EXP-P04_mingyuehu_h{horizon}_reproduce.log"
    logger = setup_logger("reproduce_mingyuehu", log_file)
    logger.info("=" * 60)
    logger.info("明月湖多 Seed 复现  horizon=%d", horizon)

    summary = run_reproduce(horizon, logger)
    elapsed = time.time() - t0

    mean, std = summary["mean"], summary["std"]
    result_summary = {
        "dataset": "mingyuehu",
        "model": "cnn_bilstm",
        "horizon": horizon,
        "seeds": summary["seeds"],
        "RMSE_mean": round(mean["RMSE"], 4),
        "RMSE_std": round(std["RMSE"], 4),
        "MAE_mean": round(mean["MAE"], 4),
        "MAE_std": round(std["MAE"], 4),
        "R2_mean": round(mean["R2"], 4),
        "R2_std": round(std["R2"], 4),
        "elapsed_sec": round(elapsed, 1),
    }
    logger.info("结果: %s", result_summary)

    metrics_h = METRICS_DIR / f"mingyuehu_h{horizon}"
    artifacts = [
        str((metrics_h / "mingyuehu_cnn_bilstm_reproduce.json").relative_to(PROJECT_ROOT)),
        str((MODELS_DIR / f"mingyuehu_h{horizon}" / "mingyuehu_cnn_bilstm_seed42.pt").relative_to(PROJECT_ROOT)),
        str((PRED_DIR / f"mingyuehu_h{horizon}" / "cnn_bilstm_seed42_test.csv").relative_to(PROJECT_ROOT)),
    ]
    record_step_result(
        horizon, "reproduce_mingyuehu", "success", log_file,
        summary=result_summary, duration_sec=elapsed, artifacts=artifacts,
    )
    return horizon, log_file


if __name__ == "__main__":
    t0 = time.time()
    try:
        main()
    except Exception as e:
        record_step_failure("reproduce_mingyuehu", t0, e)
        raise
