"""
python -m experiments.prediction.step4_optuna_hybrid.run_exp_p04_reproduce --horizon 1
多 seed 复现：用多个 seed 重复最终训练，统计均值和标准差（seed 列表见 exp_p04_base.json）。
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

from experiments.prediction.step4_optuna_hybrid.exp_p04_common import (
    METRICS_DIR,
    MODELS_DIR,
    PRED_DIR,
    SAMPLES_DIR,
    compute_all_metrics,
    load_config,
    load_sample_dir,
    load_y_scaler_from_json,
    save_predictions,
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
    get_device,
    make_loader,
    predict,
    train_with_early_stop,
)


def _best_params(params: dict, model_name: str) -> dict:
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


def run_reproduce(horizon: int, horizon_cfg: dict, base_cfg: dict, logger):
    """多 seed 复现。"""
    hdir = load_sample_dir(horizon)
    metrics_h = METRICS_DIR / f"h{horizon}"
    models_h = MODELS_DIR / f"h{horizon}"
    pred_h = PRED_DIR / f"h{horizon}"
    for d in (metrics_h, models_h, pred_h):
        d.mkdir(parents=True, exist_ok=True)

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

    seeds = base_cfg["reproduce_seeds"]
    all_models = horizon_cfg["baseline_models"] + list(horizon_cfg["model_search_space"].keys())
    max_epochs = base_cfg["final_max_epochs"]
    patience = base_cfg["final_patience"]
    default_lr = base_cfg["final_lr"]

    logger.info("=" * 60)
    logger.info("多 Seed 复现  horizon=%d  seeds=%s", horizon, seeds)

    summaries: dict[str, dict] = {}
    for mname in all_models:
        optuna_path = metrics_h / f"{mname}_optuna.json"
        if not optuna_path.exists():
            logger.warning("跳过 %s（无 Optuna 结果）", mname)
            continue

        params_json = json.loads(optuna_path.read_text(encoding="utf-8"))
        params = params_json["best_params"]
        batch_size = int(params.pop("batch_size", 64))
        lr_use = float(params.pop("lr", default_lr))

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
            "model": mname, "horizon": horizon, "seeds": seeds,
            "mean": {k: round(v, 6) for k, v in mean_row.items()},
            "std": {k: round(v, 6) for k, v in std_row.items()},
            "per_seed": all_metrics,
            "prediction_mode": "residual",
        }

        out_path = metrics_h / f"{mname}_reproduce.json"
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
        pred_path = save_predictions(f"h{horizon}", f"{mname}_seed42",
                                    y_true_power_seed.ravel(), y_pred_power_seed.ravel())
        logger.info("  seed=42 预测已保存: %s", pred_path.name)

        model_path = models_h / f"{mname}_seed42.pt"
        torch.save(model_seed.state_dict(), model_path)
        summaries[mname] = summary

    logger.info("=" * 60)
    logger.info("多 Seed 复现完成！")
    return summaries


def main():
    parser = argparse.ArgumentParser(description="EXP-P04 多 Seed 复现")
    parser.add_argument("--horizon", type=int, choices=[1, 4, 16], required=True)
    args = parser.parse_args()

    t0 = time.time()
    horizon = args.horizon
    horizon_cfg = load_config(f"exp_p04_h{horizon}.json")
    base_cfg = load_config("exp_p04_base.json")

    log_file = horizon_cfg["log_file"].replace(".log", "_reproduce.log")
    logger = setup_logger("reproduce", log_file)
    logger.info("=" * 60)
    logger.info("EXP-P04 多 Seed 复现  horizon=%d", horizon)

    summaries = run_reproduce(horizon, horizon_cfg, base_cfg, logger)
    elapsed = time.time() - t0
    hs = f"h{horizon}"

    if "cnn_bilstm" not in summaries:
        raise RuntimeError("cnn_bilstm 多 seed 复现未成功")

    rep = summaries["cnn_bilstm"]
    mean, std = rep["mean"], rep["std"]
    summary = {
        "seeds": rep["seeds"],
        "RMSE_mean": round(mean["RMSE"], 4),
        "RMSE_std": round(std["RMSE"], 4),
        "MAE_mean": round(mean["MAE"], 4),
        "MAE_std": round(std["MAE"], 4),
        "R2_mean": round(mean["R2"], 4),
        "R2_std": round(std["R2"], 4),
        "elapsed_sec": round(elapsed, 1),
    }
    artifacts = [
        f"data/prediction/step4_optuna_hybrid/metrics/{hs}/cnn_bilstm_reproduce.json",
        f"data/prediction/step4_optuna_hybrid/models/{hs}/cnn_bilstm_seed42.pt",
        f"data/prediction/step4_optuna_hybrid/predictions/{hs}/cnn_bilstm_seed42_test.csv",
    ]
    record_step_result(
        horizon, "reproduce", "success", log_file,
        summary=summary, duration_sec=elapsed, artifacts=artifacts,
    )
    return horizon, log_file


if __name__ == "__main__":
    t0 = time.time()
    try:
        main()
    except Exception as e:
        record_step_failure("reproduce", t0, e)
        raise
