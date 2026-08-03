"""
EXP-P08: 物理约束特征实验

添加物理约束特征来改善峰值预测：
1. 理论最大功率 (基于辐照度)
2. 温度校正因子
3. 晴空指数
4. 剩余功率空间

执行方式：
    python experiments/prediction/step5_new_experiments/run_exp_p08_physics.py --horizon 1
    python experiments/prediction/step5_new_experiments/run_exp_p08_physics.py --horizon 1 --model lstm

输出：
    data/prediction/step5_new_experiments/metrics/h{1,4,16}/physics_features_metrics.json
    data/prediction/step5_new_experiments/predictions/h{1,4,16}/{model}_physics_test.csv
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

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
from experiments.prediction.step5_new_experiments.exp_p05_common import (
    METRICS_DIR,
    MODELS_DIR,
    PRED_DIR,
    RESIDUAL_MODELS,
    compute_all_metrics,
    compute_segmented_metrics,
    ensure_dirs,
    load_config,
    save_predictions,
    set_seed,
    setup_logger,
)
from experiments.prediction.step5_new_experiments.exp_p05_features import (
    build_p05_features,
    build_windows_from_df,
    get_p05_feature_columns,
)
from experiments.prediction.step5_new_experiments.exp_p08_physics_features import (
    build_physics_features,
    get_physics_feature_columns,
    get_all_feature_columns,
)


def compute_peak_low_metrics(y_true, y_pred, daylight_flag=None):
    """
    计算峰值、低谷分段指标。与 run_exp_p07_improved_loss.py 保持一致。

    峰值: top 20% (>=80th percentile)
    低谷: bottom 5% (<=5th percentile)
    中间: 其余 75%
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # 与 exp_p07 保持一致的阈值
    peak_threshold = np.percentile(y_true, 80)
    low_threshold = np.percentile(y_true, 5)

    results = {}

    # 峰值区间 (top 20%)
    peak_mask = y_true >= peak_threshold
    if peak_mask.any():
        results["peak"] = compute_all_metrics(y_true[peak_mask], y_pred[peak_mask])

    # 低谷区间 (bottom 5%)
    low_mask = y_true <= low_threshold
    if low_mask.any():
        results["low_power"] = compute_all_metrics(y_true[low_mask], y_pred[low_mask])

    # 中间区间
    mid_mask = ~peak_mask & ~low_mask
    if mid_mask.any():
        results["mid"] = compute_all_metrics(y_true[mid_mask], y_pred[mid_mask])

    # 全区间
    results["all"] = compute_all_metrics(y_true, y_pred)

    return results


def prepare_physics_samples(df, horizon, lookback, feature_cols):
    """
    准备带物理特征的样本。
    """
    import pandas as pd
    import numpy as np

    # 构建物理特征
    df_feat = build_physics_features(df)

    # 获取完整特征列
    all_cols = get_all_feature_columns()

    # 检查特征列
    missing = [c for c in all_cols if c not in df_feat.columns]
    if missing:
        raise ValueError(f"缺少特征列: {missing}")

    df_feat = df_feat.dropna(subset=all_cols + ["power_pu"]).reset_index()

    # 构建时序窗口
    feat = df_feat[all_cols].to_numpy(dtype=np.float64)
    target = df_feat["power_pu"].to_numpy(dtype=np.float64)
    daylight = df_feat["daylight_flag"].to_numpy(dtype=np.float64) if "daylight_flag" in df_feat.columns else np.zeros(len(df_feat))
    timestamps = pd.to_datetime(df_feat["timestamp"]).to_numpy() if "timestamp" in df_feat.columns else np.arange(len(df_feat))

    n = len(target)
    n_samples = n - lookback - horizon + 1
    if n_samples <= 0:
        raise ValueError("时序长度不足，无法构造窗口样本")

    X_seq = np.zeros((n_samples, lookback, len(all_cols)), dtype=np.float32)
    y = np.zeros((n_samples, horizon), dtype=np.float32)
    y_last = np.zeros((n_samples, 1), dtype=np.float32)
    day_flag = np.zeros((n_samples, 1), dtype=np.float32)
    ts_out = np.empty(n_samples, dtype="datetime64[ns]")
    valid = np.ones(n_samples, dtype=bool)

    for i in range(n_samples):
        x_win = feat[i : i + lookback]
        y_win = target[i + lookback : i + lookback + horizon]
        last_idx = i + lookback - 1
        if np.isnan(x_win).any() or np.isnan(y_win).any():
            valid[i] = False
        X_seq[i] = x_win
        y[i] = y_win
        y_last[i, 0] = target[last_idx]
        day_flag[i, 0] = daylight[i + lookback]
        ts_out[i] = timestamps[i + lookback]

    mask = valid
    return X_seq[mask], y[mask], y_last[mask], day_flag[mask], ts_out[mask]


def train_physics_model(
    model_name: str,
    horizon: int,
    cfg: dict,
    logger,
) -> dict:
    """训练带物理特征的模型。"""
    import pandas as pd

    # 加载原始数据
    data_path = PROJECT_ROOT / cfg["data_raw_path"]
    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.set_index("timestamp")

    # 基础特征
    df_feat = build_p05_features(df)
    # 物理特征
    df_feat = build_physics_features(df_feat)

    lookback = cfg["lookback"]
    all_feature_cols = get_all_feature_columns()

    # 准备样本
    X_all, y_all, y_last_all, day_all, ts_all = prepare_physics_samples(
        df_feat, horizon, lookback, all_feature_cols
    )

    logger.info("总样本: %d  X=%s  y=%s  n_features=%d",
                len(X_all), X_all.shape, y_all.shape, len(all_feature_cols))

    # 划分数据集
    n = len(X_all)
    n_train_val = int(n * (cfg["train_frac"] + cfg["val_frac"]))
    n_train = int(n_train_val * cfg["train_frac"] / (cfg["train_frac"] + cfg["val_frac"]))
    n_val = n_train_val - n_train

    splits = {
        "train": slice(0, n_train),
        "val": slice(n_train, n_train_val),
        "test": slice(n_train_val, n),
    }

    X_train, X_val, X_test = X_all[splits["train"]], X_all[splits["val"]], X_all[splits["test"]]
    y_train, y_val, y_test = y_all[splits["train"]], y_all[splits["val"]], y_all[splits["test"]]
    y_last_test = y_last_all[splits["test"]]
    day_test = day_all[splits["test"]]

    # 标准化
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, len(all_feature_cols)))

    def transform_x(x):
        s = x.shape
        return scaler.transform(x.reshape(-1, s[-1])).reshape(s).astype(np.float32)

    X_train = transform_x(X_train)
    X_val = transform_x(X_val)
    X_test = transform_x(X_test)

    # 标签标准化 (用于训练)
    y_scaler = StandardScaler()
    y_scaler.fit(y_train)
    y_train_scaled = y_scaler.transform(y_train).astype(np.float32)
    y_val_scaled = y_scaler.transform(y_val).astype(np.float32)

    # 训练配置
    train_cfg = cfg.get("physics_train", {})
    batch_size = train_cfg.get("batch_size", 64)
    lr = train_cfg.get("lr", 0.001)
    max_epochs = train_cfg.get("max_epochs", 100)
    patience = train_cfg.get("patience", 15)
    seed = train_cfg.get("seed", 42)

    set_seed(seed)
    device = get_device()

    # 构建模型
    model = build_model(
        model_name,
        n_features=len(all_feature_cols),
        seq_len=lookback,
        horizon=horizon,
    ).to(device)

    # 统计参数
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    train_loader = make_loader(X_train, y_train_scaled, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val_scaled, batch_size=batch_size, shuffle=False)

    t0 = time.time()
    model, history = train_with_early_stop(
        model,
        train_loader,
        val_loader,
        lr=lr,
        max_epochs=max_epochs,
        patience=patience,
        device=device,
    )
    train_time = time.time() - t0

    # 保存模型
    model_path = MODELS_DIR / f"h{horizon}" / f"{model_name}_physics.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)

    # 测试评估
    y_pred_scaled = predict(model, X_test, device, batch_size=batch_size)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)

    if horizon == 1:
        y_true_eval = y_test[:, 0]
        y_pred_eval = y_pred[:, 0]
    else:
        y_true_eval = y_test.ravel()
        y_pred_eval = y_pred.ravel()

    # 基本指标
    metrics = compute_all_metrics(y_true_eval, y_pred_eval)

    # 分段指标
    metrics["segmented"] = compute_peak_low_metrics(
        y_true_eval, y_pred_eval,
        daylight_flag=day_test.ravel() if day_test is not None else None
    )

    # 使用 y_scaler 的 scale_ 计算一致的 nRMSE
    y_scale = y_scaler.scale_[0] if y_scaler.scale_.ndim == 1 else y_scaler.scale_[0, 0]
    metrics["y_scale"] = float(y_scale)
    metrics["nRMSE"] = float(metrics["RMSE"] / y_scale)
    metrics["MAE_scaled"] = float(metrics["MAE"] / y_scale)
    metrics["RMSE_scaled"] = float(metrics["RMSE"] / y_scale)
    y_true_scaled = (y_true_eval - y_scaler.mean_[0]) / y_scale
    y_pred_scaled_metric = (y_pred_eval - y_scaler.mean_[0]) / y_scale
    metrics["R2_scaled"] = float(r2_score(y_true_scaled, y_pred_scaled_metric))

    metrics["training_time_sec"] = train_time
    metrics["inference_ms_per_sample"] = 0.0  # TODO: benchmark
    metrics["params"] = params

    # 保存预测
    save_predictions(horizon, f"{model_name}_physics", y_true_eval, y_pred_eval)

    # 保存 scaler
    scaler_params = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "feature_cols": all_feature_cols,
        "y_mean": y_scaler.mean_.tolist(),
        "y_scale": y_scaler.scale_.tolist(),
    }
    scaler_dir = METRICS_DIR / f"h{horizon}" / "physics_scalers"
    scaler_dir.mkdir(parents=True, exist_ok=True)
    (scaler_dir / f"{model_name}_scaler.json").write_text(
        json.dumps(scaler_params, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    # 打印结果
    logger.info(
        "%s (physics) RMSE=%.4f nRMSE=%.4f R2=%.4f | Peak_R2=%.3f Low_RMSE=%.4f | time=%.1fs",
        model_name,
        metrics["RMSE"],
        metrics["nRMSE"],
        metrics["R2"],
        metrics["segmented"].get("peak", {}).get("R2", -1),
        metrics["segmented"].get("low_power", {}).get("RMSE", -1),
        train_time,
    )

    return metrics


def main():
    parser = argparse.ArgumentParser(description="EXP-P08 物理约束特征实验")
    parser.add_argument("--horizon", type=int, choices=[1, 4, 16], required=True)
    parser.add_argument("--model", type=str, choices=RESIDUAL_MODELS, default=None,
                        help="指定模型（默认全部 5 个）")
    parser.add_argument("--config", type=str, default="exp_p05_base.json",
                        help="配置文件名")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(
        MODELS_DIR / f"h{args.horizon}",
        METRICS_DIR / f"h{args.horizon}",
        PRED_DIR / f"h{args.horizon}"
    )
    logger = setup_logger("physics_exp", f"EXP-P08_h{args.horizon}_physics.log")

    models = [args.model] if args.model else RESIDUAL_MODELS

    logger.info("=" * 60)
    logger.info("EXP-P08 物理约束特征实验 horizon=%d", args.horizon)
    logger.info("物理特征: %s", get_physics_feature_columns())

    all_metrics = {}
    for m in models:
        try:
            all_metrics[f"{m}_physics"] = train_physics_model(m, args.horizon, cfg, logger)
        except Exception as e:
            logger.error("模型 %s 训练失败: %s", m, e, exc_info=True)
            all_metrics[m] = {"error": str(e)}

    # 保存结果
    out = METRICS_DIR / f"h{args.horizon}" / "physics_features_metrics.json"
    out.write_text(json.dumps(all_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("物理特征实验完成，结果已保存: %s", out.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
