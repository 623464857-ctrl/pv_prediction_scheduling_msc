"""
实验编号: EXP-P02-SVR
实验名称: 基础预测模型对比 — SVR
实验目的: 验证 RBF 核支持向量回归在光伏单步预测中的基线性能
运行方式: python experiments/prediction/step2_baseline_models/run_exp_p02_train_svr.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from exp_p02_common import (  # noqa: E402
    MODELS_DIR,
    append_log_summary,
    ensure_dirs,
    load_config,
    load_samples,
    save_predictions,
    set_seed,
    setup_logger,
)

LOG_NAME = "EXP-P02_SVR.log"


def main() -> None:
    logger = setup_logger("EXP-P02-SVR", LOG_NAME)
    cfg = load_config()
    set_seed(cfg["random_seed"])
    ensure_dirs()

    data = load_samples()
    X_train, X_val, X_test = data["X_train_flat"], data["X_val_flat"], data["X_test_flat"]
    y_train, y_val, y_test = data["y_train"], data["y_val"], data["y_test"]

    # RBF SVR 在全量 ~5 万样本上过慢；固定种子子采样用于拟合，验证/测试仍用全量
    max_svr_train = 10000
    rng = np.random.RandomState(cfg["random_seed"])
    if len(X_train) > max_svr_train:
        idx = rng.choice(len(X_train), size=max_svr_train, replace=False)
        X_fit, y_fit = X_train[idx], y_train[idx]
        logger.info("SVR 训练子样本: %d / %d (验证与测试全量)", max_svr_train, len(X_train))
    else:
        X_fit, y_fit = X_train, y_train

    # 全量 RBF SVR 在 ~5 万样本上极慢；按验证集轻量网格选优（6 组）
    grid = [(c, eps) for c in (1, 10, 100) for eps in (0.01, 0.05)]
    best_rmse = float("inf")
    best_params = {"C": 10, "epsilon": 0.01}
    best_model = None

    for c, eps in grid:
        logger.info("SVR 拟合中 C=%s epsilon=%s ...", c, eps)
        model = SVR(kernel="rbf", C=c, epsilon=eps, gamma="scale", cache_size=500)
        model.fit(X_fit, y_fit)
        pred_val = model.predict(X_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, pred_val)))
        logger.info("SVR C=%s epsilon=%s | val_rmse=%.6f", c, eps, rmse)
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = {"C": c, "epsilon": eps}
            best_model = model

    model_path = MODELS_DIR / "svr.joblib"
    joblib.dump(best_model, model_path)
    y_pred = best_model.predict(X_test)
    pred_path = save_predictions("svr", y_test, y_pred)

    logger.info("最优参数: C=%s, epsilon=%s | val_rmse=%.6f", best_params["C"], best_params["epsilon"], best_rmse)
    logger.info("模型: %s | 预测: %s", model_path.name, pred_path.name)

    append_log_summary(
        LOG_NAME,
        [
            "=" * 60,
            "【EXP-P02-SVR 摘要】",
            f"- 核: RBF, gamma=scale",
            f"- 训练子样本: {len(X_fit)} (验证/测试全量)",
            f"- 最优: C={best_params['C']}, epsilon={best_params['epsilon']}, val_rmse={best_rmse:.6f}",
            f"- 产出: models/svr.joblib, predictions/svr_test.csv",
            "=" * 60,
        ],
    )
    logger.info("EXP-P02-SVR 结束")


if __name__ == "__main__":
    main()
