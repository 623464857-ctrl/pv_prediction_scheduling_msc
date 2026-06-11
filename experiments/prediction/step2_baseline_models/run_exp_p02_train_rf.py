"""
实验编号: EXP-P02-RF
实验名称: 基础预测模型对比 — Random Forest
实验目的: 验证随机森林对非线性特征组合的拟合能力
运行方式: python experiments/prediction/step2_baseline_models/run_exp_p02_train_rf.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestRegressor

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

LOG_NAME = "EXP-P02_RF.log"


def main() -> None:
    logger = setup_logger("EXP-P02-RF", LOG_NAME)
    cfg = load_config()
    set_seed(cfg["random_seed"])
    ensure_dirs()

    data = load_samples()
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=18,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=cfg["random_seed"],
        n_jobs=-1,
    )
    logger.info("训练 Random Forest (n_estimators=300)...")
    model.fit(data["X_train_flat"], data["y_train"])

    model_path = MODELS_DIR / "rf.joblib"
    joblib.dump(model, model_path)
    y_pred = model.predict(data["X_test_flat"])
    pred_path = save_predictions("randomforest", data["y_test"], y_pred)

    logger.info("模型: %s | 预测: %s", model_path.name, pred_path.name)

    append_log_summary(
        LOG_NAME,
        [
            "=" * 60,
            "【EXP-P02-RF 摘要】",
            "- 参数: n_estimators=300, max_depth=18, max_features=sqrt",
            f"- 产出: models/rf.joblib, predictions/randomforest_test.csv",
            "=" * 60,
        ],
    )
    logger.info("EXP-P02-RF 结束")


if __name__ == "__main__":
    main()
