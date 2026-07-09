"""
Step 3: 残差预测建模（LSTM / BiLSTM / CNN-LSTM / CNN-BiLSTM / PatchTST）
python experiments/prediction/step5_new_experiments/run_exp_p05_residual_train.py --horizon 1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.prediction.step4_optuna_hybrid.exp_p04_models import build_model
from experiments.prediction.step4_optuna_hybrid.exp_p04_torch_utils import (
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
    ensure_dirs,
    load_config,
    load_meta,
    load_samples,
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


DEFAULT_MODEL_KWARGS = {
    "lstm": {"hidden": 64, "layers": 2, "dropout": 0.2},
    "bilstm": {"hidden": 64, "layers": 2, "dropout": 0.2},
    "cnn_lstm": {"conv_channels": 32, "lstm_hidden": 64, "lstm_layers": 2, "dropout": 0.2},
    "cnn_bilstm": {"conv_channels": 32, "bilstm_hidden": 64, "bilstm_layers": 2, "dropout": 0.2},
    "patchtst": {"patch_len": 4, "stride": 2, "d_model": 64, "n_heads": 4, "num_layers": 2, "dropout": 0.2},
}


def train_one_model(model_name: str, horizon: int, cfg: dict, logger) -> dict:
    samples = load_samples(horizon, use_step5=True)
    meta = load_meta(horizon, use_step5=True)
    rt = cfg["residual_train"]
    set_seed(rt["seed"])
    device = get_device()

    y_last_train = samples["y_last_train"]
    y_last_val = samples["y_last_val"]
    y_last_test = samples["y_last_test"]

    y_res_train = compute_residual_targets(samples["y_train_raw"], y_last_train)
    y_res_val = compute_residual_targets(samples["y_val_raw"], y_last_val)
    res_scaler = fit_residual_scaler(y_res_train)
    save_residual_scaler(
        res_scaler,
        MODELS_DIR.parent / "samples" / f"h{horizon}" / "residual_scaler_params.json",
    )

    y_train = transform_residual(res_scaler, y_res_train)
    y_val = transform_residual(res_scaler, y_res_val)

    model = build_model(
        model_name,
        n_features=meta["n_features"],
        seq_len=meta["lookback"],
        horizon=horizon,
        **DEFAULT_MODEL_KWARGS.get(model_name, {}),
    )
    train_loader = make_loader(samples["X_train_seq"], y_train, batch_size=rt["batch_size"], shuffle=True)
    val_loader = make_loader(samples["X_val_seq"], y_val, batch_size=rt["batch_size"], shuffle=False)

    t0 = time.time()
    model, history = train_with_early_stop(
        model,
        train_loader,
        val_loader,
        lr=rt["lr"],
        max_epochs=rt["max_epochs"],
        patience=rt["patience"],
        device=device,
    )
    train_time = time.time() - t0

    model_path = MODELS_DIR / f"h{horizon}" / f"{model_name}_residual.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)

    delta_scaled = predict(model, samples["X_test_seq"], device, batch_size=rt["batch_size"])
    delta_pred = inverse_transform_residual(res_scaler, delta_scaled)
    y_pred = reconstruct_from_residual(y_last_test, delta_pred)

    y_true = samples["y_test_raw"]
    if horizon == 1:
        y_true_eval = y_true[:, 0]
        y_pred_eval = y_pred[:, 0]
    else:
        y_true_eval = y_true.ravel()
        y_pred_eval = y_pred.ravel()

    metrics = compute_all_metrics(y_true_eval, y_pred_eval)
    metrics["training_time_sec"] = train_time
    save_predictions(horizon, f"{model_name}_residual", y_true_eval, y_pred_eval)

    hist_path = METRICS_DIR / f"h{horizon}" / f"{model_name}_residual_train_history.csv"
    import pandas as pd

    pd.DataFrame(history).to_csv(hist_path, index=False)
    logger.info("%s residual RMSE=%.4f MAE=%.4f", model_name, metrics["RMSE"], metrics["MAE"])
    return metrics


def main():
    parser = argparse.ArgumentParser(description="EXP-P05 残差预测训练")
    parser.add_argument("--horizon", type=int, choices=[1, 4, 16], required=True)
    parser.add_argument("--models", nargs="*", default=None, help="默认训练全部 5 个模型")
    args = parser.parse_args()

    cfg = load_config()
    models = args.models or cfg["residual_models"]
    ensure_dirs(MODELS_DIR / f"h{args.horizon}", METRICS_DIR / f"h{args.horizon}", PRED_DIR / f"h{args.horizon}")
    logger = setup_logger("residual_train", f"EXP-P05_h{args.horizon}_residual_train.log")

    all_metrics = {}
    for m in models:
        if m not in RESIDUAL_MODELS:
            logger.warning("跳过未知模型: %s", m)
            continue
        all_metrics[f"{m}_residual"] = train_one_model(m, args.horizon, cfg, logger)

    out = METRICS_DIR / f"h{args.horizon}" / "residual_metrics.json"
    out.write_text(json.dumps(all_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("残差训练完成，指标: %s", out)


if __name__ == "__main__":
    main()
