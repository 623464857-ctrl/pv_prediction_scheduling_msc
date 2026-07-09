"""
Step 4 + 5: 白天/夜间分段评价 + 推理时间标准化重测
python experiments/prediction/step5_new_experiments/run_exp_p05_evaluation.py --horizon 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.prediction.step4_optuna_hybrid.exp_p04_models import build_model
from experiments.prediction.step5_new_experiments.exp_p05_benchmark import benchmark_forward
from experiments.prediction.step5_new_experiments.exp_p05_common import (
    BENCHMARK_DIR,
    METRICS_DIR,
    MODEL_DISPLAY_NAMES,
    PRED_DIR,
    RESIDUAL_MODELS,
    compute_segmented_metrics,
    ensure_dirs,
    get_device,
    load_config,
    load_meta,
    load_samples,
    setup_logger,
)


DEFAULT_MODEL_KWARGS = {
    "lstm": {"hidden": 64, "layers": 2, "dropout": 0.2},
    "bilstm": {"hidden": 64, "layers": 2, "dropout": 0.2},
    "cnn_lstm": {"conv_channels": 32, "lstm_hidden": 64, "lstm_layers": 2, "dropout": 0.2},
    "cnn_bilstm": {"conv_channels": 32, "bilstm_hidden": 64, "bilstm_layers": 2, "dropout": 0.2},
    "patchtst": {"patch_len": 4, "stride": 2, "d_model": 64, "n_heads": 4, "num_layers": 2, "dropout": 0.2},
}


def load_prediction_csv(horizon: int, model_key: str) -> tuple[np.ndarray, np.ndarray] | None:
    path = PRED_DIR / f"h{horizon}" / f"{model_key}_test.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df["y_true"].values, df["y_pred"].values


def evaluate_predictions(horizon: int, cfg: dict, logger) -> dict:
    samples = load_samples(horizon, use_step5=True)
    daylight = samples.get("daylight_flag_test")
    capacity = cfg["capacity_pu"]
    threshold = cfg["daytime_threshold"]

    keys = [
        "persistence", "moving_average", "ridge", "xgboost", "lightgbm",
        "lstm_residual", "bilstm_residual", "cnn_lstm_residual", "cnn_bilstm_residual", "patchtst_residual",
    ]
    segmented = {}
    for key in keys:
        loaded = load_prediction_csv(horizon, key)
        if loaded is None:
            continue
        y_true, y_pred = loaded
        segmented[key] = compute_segmented_metrics(
            y_true, y_pred, daylight_flag=daylight, capacity=capacity, threshold_ratio=threshold
        )
        logger.info("%s daytime RMSE=%.4f", key, segmented[key]["daytime_only"]["RMSE"])
    return segmented


def benchmark_models(horizon: int, cfg: dict, logger) -> dict:
    samples = load_samples(horizon, use_step5=True)
    meta = load_meta(horizon, use_step5=True)
    device = get_device()
    bench_cfg = cfg["inference_benchmark"]
    batch_size = bench_cfg["batch_size"]
    X = samples["X_test_seq"][:batch_size]

    results = {}
    for model_name in RESIDUAL_MODELS:
        model_key = f"{model_name}_residual"
        ckpt = PROJECT_ROOT / "data/prediction/step5_new_experiments/models" / f"h{horizon}" / f"{model_name}_residual.pt"
        if not ckpt.exists():
            logger.warning("无模型权重，跳过 benchmark: %s", model_key)
            continue
        model = build_model(
            model_name,
            n_features=meta["n_features"],
            seq_len=meta["lookback"],
            horizon=horizon,
            **DEFAULT_MODEL_KWARGS.get(model_name, {}),
        )
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        sample = torch.from_numpy(X.astype(np.float32))
        bench = benchmark_forward(
            model,
            sample,
            warmup_iters=bench_cfg["warmup_iters"],
            repeat_iters=bench_cfg["repeat_iters"],
            device=device,
        )
        results[model_key] = bench
        logger.info("%s inference %.3f ms/sample", model_key, bench["ms_per_sample"])
    return results


def main():
    parser = argparse.ArgumentParser(description="EXP-P05 分段评价与推理计时")
    parser.add_argument("--horizon", type=int, choices=[1, 4, 16], required=True)
    args = parser.parse_args()

    cfg = load_config()
    ensure_dirs(METRICS_DIR / f"h{args.horizon}", BENCHMARK_DIR / f"h{args.horizon}")
    logger = setup_logger("evaluation", f"EXP-P05_h{args.horizon}_evaluation.log")

    segmented = evaluate_predictions(args.horizon, cfg, logger)
    benchmark = benchmark_models(args.horizon, cfg, logger)

    seg_path = METRICS_DIR / f"h{args.horizon}" / "segmented_metrics.json"
    seg_path.write_text(json.dumps(segmented, indent=2, ensure_ascii=False), encoding="utf-8")

    bench_path = BENCHMARK_DIR / f"h{args.horizon}" / "inference_benchmark.json"
    bench_path.write_text(json.dumps(benchmark, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("评价完成")


if __name__ == "__main__":
    main()
