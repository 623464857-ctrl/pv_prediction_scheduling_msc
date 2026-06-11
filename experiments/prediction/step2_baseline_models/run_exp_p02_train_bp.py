"""
实验编号: EXP-P02-BP
实验名称: 基础预测模型对比 — BP 神经网络
实验目的: 构建浅层前馈神经网络基线，预测单步 power_pu
运行方式: python experiments/prediction/step2_baseline_models/run_exp_p02_train_bp.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

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
from exp_p02_torch_utils import BPNet, get_device, make_loader, predict, train_with_early_stop  # noqa: E402

LOG_NAME = "EXP-P02_BP.log"


def main() -> None:
    logger = setup_logger("EXP-P02-BP", LOG_NAME)
    cfg = load_config()
    set_seed(cfg["random_seed"])
    ensure_dirs()

    data = load_samples()
    in_dim = data["X_train_flat"].shape[1]
    device = get_device()
    logger.info("设备: %s | 输入维度: %d", device, in_dim)

    train_loader = make_loader(data["X_train_flat"], data["y_train"], batch_size=256, shuffle=True)
    val_loader = make_loader(data["X_val_flat"], data["y_val"], batch_size=256, shuffle=False)

    model, history = train_with_early_stop(
        BPNet(in_dim),
        train_loader,
        val_loader,
        lr=1e-3,
        max_epochs=50,
        patience=8,
        device=device,
    )

    model_path = MODELS_DIR / "bp.pt"
    import torch

    torch.save(model.state_dict(), model_path)
    hist_path = save_train_history("bp", history)
    y_pred = predict(model, data["X_test_flat"], device)
    pred_path = save_predictions("bp", data["y_test"], y_pred)

    best = min(history, key=lambda x: x["val_loss"])
    logger.info("最佳 epoch=%d | train_loss=%.6f | val_loss=%.6f", best["epoch"], best["train_loss"], best["val_loss"])
    logger.info("模型: %s | 预测: %s", model_path.name, pred_path.name)

    append_log_summary(
        LOG_NAME,
        [
            "=" * 60,
            "【EXP-P02-BP 摘要】",
            f"- 结构: 208->128->64->1, Dropout=0.2, Adam lr=1e-3",
            f"- 最佳验证损失: {best['val_loss']:.6f} (epoch {best['epoch']})",
            f"- 产出: models/bp.pt, metrics/bp_train_history.csv, predictions/bp_test.csv",
            "=" * 60,
        ],
    )
    logger.info("EXP-P02-BP 结束")


if __name__ == "__main__":
    main()
