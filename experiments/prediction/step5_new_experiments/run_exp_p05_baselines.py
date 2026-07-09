"""
Step 1: 强基线评估
python experiments/prediction/step5_new_experiments/run_exp_p05_baselines.py --horizon 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.prediction.step5_new_experiments.baselines import (
    LightGBMBaseline,
    MovingAverageBaseline,
    PersistenceBaseline,
    RidgeBaseline,
    XGBoostBaseline,
    evaluate_lightgbm,
    evaluate_moving_average,
    evaluate_persistence,
    evaluate_ridge,
    evaluate_xgboost,
)
from experiments.prediction.step5_new_experiments.exp_p05_common import (
    METRICS_DIR,
    PRED_DIR,
    ensure_dirs,
    load_config,
    load_samples,
    save_predictions,
    setup_logger,
)


def main():
    parser = argparse.ArgumentParser(description="EXP-P05 强基线评估")
    parser.add_argument("--horizon", type=int, choices=[1, 4, 16], required=True)
    args = parser.parse_args()

    horizon = args.horizon
    cfg = load_config()
    ensure_dirs(METRICS_DIR / f"h{horizon}", PRED_DIR / f"h{horizon}")
    logger = setup_logger("baselines", f"EXP-P05_h{horizon}_baselines.log")

    samples = load_samples(horizon, use_step5=True)
    X_train, y_train = samples["X_train_seq"], samples["y_train_raw"]
    X_test, y_test = samples["X_test_seq"], samples["y_test_raw"]
    y_last = samples["y_last_test"]
    horizon_steps = horizon

    results = {}

    # Persistence
    logger.info("评估 Persistence...")
    metrics = evaluate_persistence(y_test, y_last, horizon=horizon_steps)
    results["persistence"] = metrics
    y_pred = PersistenceBaseline().predict(None, y_last, horizon=horizon_steps)
    save_predictions(horizon, "persistence", y_test, y_pred)

    # Moving Average
    logger.info("评估 Moving Average...")
    metrics = evaluate_moving_average(
        X_test, y_test, window=cfg["moving_average_window"], horizon=horizon_steps
    )
    results["moving_average"] = metrics
    y_pred = MovingAverageBaseline(cfg["moving_average_window"]).predict(X_test, horizon=horizon_steps)
    save_predictions(horizon, "moving_average", y_test, y_pred)

    # Ridge
    logger.info("评估 Ridge...")
    metrics = evaluate_ridge(X_train, y_train, X_test, y_test, alpha=cfg["ridge_alpha"])
    results["ridge"] = metrics
    ridge = RidgeBaseline(alpha=cfg["ridge_alpha"]).fit(X_train, y_train)
    save_predictions(horizon, "ridge", y_test, ridge.predict(X_test))

    # XGBoost
    logger.info("评估 XGBoost...")
    try:
        metrics = evaluate_xgboost(X_train, y_train, X_test, y_test, params=cfg["xgb_params"])
        results["xgboost"] = metrics
        xgb = XGBoostBaseline(**cfg["xgb_params"]).fit(X_train, y_train)
        save_predictions(horizon, "xgboost", y_test, xgb.predict(X_test))
    except ImportError:
        logger.warning("未安装 xgboost，跳过")

    # LightGBM
    logger.info("评估 LightGBM...")
    try:
        metrics = evaluate_lightgbm(X_train, y_train, X_test, y_test, params=cfg["lgbm_params"])
        results["lightgbm"] = metrics
        lgb = LightGBMBaseline(**cfg["lgbm_params"]).fit(X_train, y_train)
        save_predictions(horizon, "lightgbm", y_test, lgb.predict(X_test))
    except ImportError:
        logger.warning("未安装 lightgbm，跳过")

    out = METRICS_DIR / f"h{horizon}" / "baseline_metrics.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("基线指标已保存: %s", out)


if __name__ == "__main__":
    main()
