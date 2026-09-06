"""
明月湖数据集最终训练
python -m experiments.prediction.step3_deep_learning.run_exp_p04_train_final_mingyuehu --horizon 1
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.prediction.step2_hyperparameter_search.exp_p04_common import (
    METRICS_DIR,
    MODELS_DIR,
    PRED_DIR,
    PROJECT_ROOT,
    compute_all_metrics,
    compute_metrics_multi_step,
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
    "lookback": 16,
    "final_max_epochs": 50,
    "final_patience": 8,
    "final_lr": 0.001,
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


def _best_params(params: dict, model_name: str) -> dict:
    """兼容从 optuna JSON 读取的参数"""
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


def train_and_evaluate_residual(model_name: str, params: dict, horizon: int, lookback: int,
                                logger, device, X_train, y_residual_train, X_val, y_residual_val,
                                X_test, y_residual_test, y_anchor_test,
                                y_scaler, seq_len, n_features, max_epochs, patience, lr, batch_size):
    """用最优参数训练残差模型并在测试集上评估"""
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


def run_all_models(horizon: int, logger):
    """加载样本 + 加载最优参数 + 训练 + 保存模型和指标"""
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
    logger.info("=" * 60)
    logger.info("明月湖最终训练 horizon=%d | train=%d val=%d test=%d",
                horizon, len(X_train), len(X_val), len(X_test))

    max_epochs = MINGYUEHU_CFG["final_max_epochs"]
    patience = MINGYUEHU_CFG["final_patience"]
    lr = MINGYUEHU_CFG["final_lr"]

    # 加载 Optuna 结果
    optuna_path = metrics_h / f"mingyuehu_cnn_bilstm_optuna.json"
    if not optuna_path.exists():
        logger.error("缺少 Optuna 结果: %s", optuna_path)
        raise FileNotFoundError(f"缺少 {optuna_path}")

    params_json = json.loads(optuna_path.read_text(encoding="utf-8"))
    params = params_json["best_params"]
    batch_size = int(params.pop("batch_size", 64))
    lr_use = float(params.pop("lr", lr))

    mname = "cnn_bilstm"
    logger.info("-" * 40)
    logger.info("训练模型: %s horizon=%d", mname, horizon)
    logger.info("  参数: %s", {**params, "batch_size": batch_size, "lr": lr_use})

    model, history, metrics, y_pred_power = train_and_evaluate_residual(
        mname, params, horizon, lookback, logger, device,
        X_train, y_residual_train, X_val, y_residual_val,
        X_test, y_residual_test, y_anchor_test,
        y_scaler, seq_len, n_features,
        max_epochs, patience, lr_use, batch_size,
    )

    # 保存模型
    model_path = models_h / f"mingyuehu_{mname}_final.pt"
    torch.save(model.state_dict(), model_path)
    logger.info("  模型已保存: %s", model_path.name)

    # 保存训练历史
    hist_path = save_train_history(f"mingyuehu_h{horizon}", f"mingyuehu_{mname}_final", history)
    logger.info("  历史已保存: %s", hist_path.name)

    # 保存预测 (真实功率 vs 预测功率)
    y_test_residual_raw = y_scaler.inverse_transform(y_residual_test)
    y_true_power = (y_anchor_test + y_test_residual_raw).astype(np.float32)
    pred_path = save_predictions(f"mingyuehu_h{horizon}", mname, y_true_power.ravel(), y_pred_power.ravel())
    logger.info("  预测已保存: %s", pred_path.name)

    # 保存指标
    metrics_path = metrics_h / f"mingyuehu_{mname}_test_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info("  指标: MAE=%.4f RMSE=%.4f MAPE=%.2f%% R2=%.4f",
                metrics["MAE"], metrics["RMSE"], metrics["MAPE"], metrics["R2"])

    return {mname: metrics}


def main():
    parser = argparse.ArgumentParser(description="明月湖最终训练")
    parser.add_argument("--horizon", type=int, choices=[1, 4, 16], required=True)
    args = parser.parse_args()

    t0 = time.time()
    horizon = args.horizon

    log_file = f"EXP-P04_mingyuehu_h{horizon}_final_train.log"
    logger = setup_logger("final_train_mingyuehu", log_file)
    logger.info("=" * 60)
    logger.info("明月湖最终训练 horizon=%d", horizon)

    results = run_all_models(horizon, logger)
    elapsed = time.time() - t0

    m = results["cnn_bilstm"]
    summary = {
        "dataset": "mingyuehu",
        "model": "cnn_bilstm",
        "horizon": horizon,
        "test_RMSE": round(m["RMSE"], 4),
        "test_MAE": round(m["MAE"], 4),
        "test_R2": round(m["R2"], 4),
        "test_MAPE": round(m["MAPE"], 2) if not np.isnan(m["MAPE"]) else None,
        "training_time_sec": m.get("training_time_sec"),
        "elapsed_sec": round(elapsed, 1),
    }
    artifacts = [
        f"data/prediction/step3_deep_learning/models/mingyuehu_h{horizon}/mingyuehu_cnn_bilstm_final.pt",
        f"data/prediction/step4_evaluation/metrics/mingyuehu_h{horizon}/mingyuehu_cnn_bilstm_test_metrics.json",
        f"data/prediction/step3_deep_learning/predictions/mingyuehu_h{horizon}/cnn_bilstm_test.csv",
    ]
    record_step_result(
        horizon, "train_final_mingyuehu", "success", log_file,
        summary=summary, duration_sec=elapsed, artifacts=artifacts,
    )
    return horizon, log_file


if __name__ == "__main__":
    t0 = time.time()
    try:
        main()
    except Exception as e:
        record_step_failure("train_final_mingyuehu", t0, e)
        raise
