"""
实验编号: EXP-P02-BiLSTM
实验名称: 基础预测模型对比 — BiLSTM
实验目的: 验证双向时序编码是否优于单向 LSTM
运行方式: python experiments/prediction/step2_baseline_models/run_exp_p02_train_bilstm.py
"""

from __future__ import annotations

import sys
from pathlib import Path

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
    save_train_history,
    set_seed,
    setup_logger,
)
from exp_p02_torch_utils import BiLSTMRegressor, get_device, make_loader, predict, train_with_early_stop  # noqa: E402

LOG_NAME = "EXP-P02_BiLSTM.log"


def main() -> None:
    logger = setup_logger("EXP-P02-BiLSTM", LOG_NAME)
    cfg = load_config()
    set_seed(cfg["random_seed"])
    ensure_dirs()

    data = load_samples()
    n_features = data["X_train_seq"].shape[2]
    device = get_device()
    logger.info("设备: %s | 输入: [batch, 16, %d]", device, n_features)

    train_loader = make_loader(data["X_train_seq"], data["y_train"], batch_size=256, shuffle=True)
    val_loader = make_loader(data["X_val_seq"], data["y_val"], batch_size=256, shuffle=False)

    model, history = train_with_early_stop(
        BiLSTMRegressor(n_features, hidden=64, num_layers=2, dropout=0.2),
        train_loader,
        val_loader,
        lr=1e-3,
        max_epochs=50,
        patience=8,
        device=device,
    )

    import torch

    torch.save(model.state_dict(), MODELS_DIR / "bilstm.pt")
    save_train_history("bilstm", history)
    y_pred = predict(model, data["X_test_seq"], device)
    pred_path = save_predictions("bilstm", data["y_test"], y_pred)

    best = min(history, key=lambda x: x["val_loss"])
    logger.info("最佳 epoch=%d | val_loss=%.6f", best["epoch"], best["val_loss"])
    logger.info("预测: %s", pred_path.name)

    append_log_summary(
        LOG_NAME,
        [
            "=" * 60,
            "【EXP-P02-BiLSTM 摘要】",
            "- 结构: 2层 BiLSTM hidden=64×2 -> Linear(128,1), Dropout=0.2",
            f"- 最佳验证损失: {best['val_loss']:.6f} (epoch {best['epoch']})",
            "- 产出: models/bilstm.pt, metrics/bilstm_train_history.csv, predictions/bilstm_test.csv",
            "=" * 60,
        ],
    )
    logger.info("EXP-P02-BiLSTM 结束")


if __name__ == "__main__":
    main()
