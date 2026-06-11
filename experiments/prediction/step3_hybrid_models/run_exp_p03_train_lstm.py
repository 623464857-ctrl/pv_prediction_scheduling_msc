"""
实验编号: EXP-P03-LSTM
实验名称: 混合深度学习模型对比 — LSTM 基准
实验目的: 作为纯时序基准，与 CNN-LSTM 对比验证 CNN 局部特征提取的增益
运行方式: python experiments/prediction/step3_hybrid_models/run_exp_p03_train_lstm.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from exp_p03_common import (
    MODELS_DIR,
    METRICS_DIR,
    FIGURES_DIR,
    PRED_DIR,
    append_log_summary,
    ensure_dirs,
    load_config,
    load_samples,
    load_test_timestamps,
    plot_loss_curve,
    plot_pred_curve,
    save_predictions,
    save_train_history,
    set_seed,
    setup_logger,
)
from exp_p03_models import LSTMRegressor
from exp_p03_torch_utils import get_device, make_loader, predict, train_with_early_stop

LOG_NAME = "EXP-P03_LSTM.log"


def main() -> None:
    logger = setup_logger("EXP-P03-LSTM", LOG_NAME)
    cfg = load_config()
    set_seed(cfg["random_seed"])
    ensure_dirs()

    data = load_samples()
    n_features = data["X_train_seq"].shape[2]
    device = get_device()
    logger.info("设备: %s | 输入: [batch, 16, %d]", device, n_features)

    t0 = time.time()
    train_loader = make_loader(data["X_train_seq"], data["y_train"], batch_size=256, shuffle=True)
    val_loader = make_loader(data["X_val_seq"], data["y_val"], batch_size=256, shuffle=False)

    model, history = train_with_early_stop(
        LSTMRegressor(n_features=n_features, hidden=64, layers=2, dropout=0.2),
        train_loader,
        val_loader,
        lr=1e-3,
        max_epochs=50,
        patience=8,
        device=device,
    )
    training_time = time.time() - t0

    import torch
    torch.save(model.state_dict(), MODELS_DIR / "lstm.pt")
    save_train_history("lstm", history)

    t1 = time.time()
    y_pred = predict(model, data["X_test_seq"], device)
    pred_path = save_predictions("lstm", data["y_test"], y_pred)
    predict_time = time.time() - t1

    best = min(history, key=lambda x: x["val_loss"])
    logger.info("最佳 epoch=%d | val_loss=%.6f | 训练耗时=%.1fs", best["epoch"], best["val_loss"], training_time)
    logger.info("预测: %s | 预测耗时=%.1fs", pred_path.name, predict_time)

    plot_loss_curve(METRICS_DIR / "lstm_train_history.csv", "LSTM Training Loss", FIGURES_DIR / "loss_lstm.png")
    ts = load_test_timestamps()
    df_pred = pd.DataFrame({"timestamp": ts, "y_true": data["y_test"], "y_pred": y_pred})
    plot_pred_curve(df_pred, "LSTM — test predictions", FIGURES_DIR / "pred_lstm.png", n_points=5 * 96)

    append_log_summary(
        LOG_NAME,
        [
            "=" * 60,
            "【EXP-P03-LSTM 摘要】",
            "- 结构: LSTM(64, 2层) -> Linear(64,1)",
            f"- 最佳验证损失: {best['val_loss']:.6f} (epoch {best['epoch']})",
            f"- 训练耗时: {training_time:.1f}s",
            "- 产出: models/lstm.pt, metrics/lstm_train_history.csv, predictions/lstm_test.csv",
            "- 图表: figures/loss_lstm.png, figures/pred_lstm.png",
            "=" * 60,
        ],
    )
    logger.info("EXP-P03-LSTM 结束")


if __name__ == "__main__":
    main()
