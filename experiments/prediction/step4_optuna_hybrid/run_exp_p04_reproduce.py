"""
python -m experiments.prediction.step4_optuna_hybrid.run_exp_p04_reproduce --horizon 1
多 seed 复现：用 3 个 seed 重复最终训练，统计均值和标准差。
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
    load_y_scaler_from_json,
    save_predictions,
    setup_logger,
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
    hdir = SAMPLES_DIR / f"h{horizon}"
    metrics_h = METRICS_DIR / f"h{horizon}"
    models_h = MODELS_DIR / f"h{horizon}"
    pred_h = PRED_DIR / f"h{horizon}"
    for d in (metrics_h, models_h, pred_h):
        d.mkdir(parents=True, exist_ok=True)

    X_train = np.load(hdir / "X_train_seq.npy")
    y_train = np.load(hdir / "y_train.npy")
    X_val = np.load(hdir / "X_val_seq.npy")
    y_val = np.load(hdir / "y_val.npy")
    X_test = np.load(hdir / "X_test_seq.npy")
    y_test_raw_arr = np.load(hdir / "y_test_raw.npy")
    y_scaler = load_y_scaler_from_json(f"h{horizon}")

    meta = json.loads((hdir / "meta.json").read_text(encoding="utf-8"))
    seq_len, n_features = meta["lookback"], X_train.shape[2]
    device = get_device()

    seeds = base_cfg["reproduce_seeds"]
    all_models = horizon_cfg["baseline_models"] + list(horizon_cfg["model_search_space"].keys())
    max_epochs = base_cfg["final_max_epochs"]
    patience = base_cfg["final_patience"]
    default_lr = base_cfg["final_lr"]

    logger.info("=" * 60)
    logger.info("多 Seed 复现  horizon=%d  seeds=%s", horizon, seeds)

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

            train_loader = make_loader(X_train, y_train, batch_size=batch_size, shuffle=True)
            val_loader = make_loader(X_val, y_val, batch_size=batch_size, shuffle=False)

            t0 = time.time()
            model, _ = train_with_early_stop(
                model, train_loader, val_loader,
                lr=lr_use, max_epochs=max_epochs, patience=patience, device=device,
            )
            elapsed = time.time() - t0

            y_pred = y_scaler.inverse_transform(predict(model, X_test, device))
            metrics = compute_all_metrics(y_test_raw_arr.ravel(), y_pred.ravel())
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
        train_loader = make_loader(X_train, y_train, batch_size=batch_size, shuffle=True)
        val_loader = make_loader(X_val, y_val, batch_size=batch_size, shuffle=False)
        model_seed, _ = train_with_early_stop(
            model_seed, train_loader, val_loader,
            lr=lr_use, max_epochs=max_epochs, patience=patience, device=device,
        )
        y_pred_seed = y_scaler.inverse_transform(predict(model_seed, X_test, device))
        pred_path = save_predictions(f"h{horizon}", f"{mname}_seed42",
                                    y_test_raw_arr.ravel(), y_pred_seed.ravel())
        logger.info("  seed=42 预测已保存: %s", pred_path.name)

        model_path = models_h / f"{mname}_seed42.pt"
        torch.save(model_seed.state_dict(), model_path)

    logger.info("=" * 60)
    logger.info("多 Seed 复现完成！")


def main():
    parser = argparse.ArgumentParser(description="EXP-P04 多 Seed 复现")
    parser.add_argument("--horizon", type=int, choices=[1, 4, 16], required=True)
    args = parser.parse_args()

    horizon = args.horizon
    horizon_cfg = load_config(f"exp_p04_h{horizon}.json")
    base_cfg = load_config("exp_p04_base.json")

    log_file = horizon_cfg["log_file"].replace(".log", "_reproduce.log")
    logger = setup_logger("reproduce", log_file)
    logger.info("=" * 60)
    logger.info("EXP-P04 多 Seed 复现  horizon=%d", horizon)

    run_reproduce(horizon, horizon_cfg, base_cfg, logger)


if __name__ == "__main__":
    main()
