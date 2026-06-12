"""
快速训练脚本：使用已知最优参数训练 AFSA-PatchTST
跳过 AFSA 搜索步骤，直接进行完整训练
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from exp_p03_common import (
    MODELS_DIR,
    METRICS_DIR,
    FIGURES_DIR,
    PRED_DIR,
    compute_all_metrics,
    ensure_dirs,
    load_config,
    load_samples,
    load_test_timestamps,
    save_predictions,
    save_train_history,
    set_seed,
    setup_logger,
    append_log_summary,
)
from exp_p03_models import build_model
from exp_p03_torch_utils import get_device, make_loader, predict, train_with_early_stop


LOG_NAME = "EXP-P03_AFSA_Final.log"

# 已知最优参数（来自之前 AFSA 搜索结果）
BEST_PARAMS = {
    "patch_len": 2,
    "stride": 2,
    "d_model": 32,
    "n_heads": 8,
    "num_layers": 3,
    "dropout": 0.1,
    "learning_rate": 0.002,
    "batch_size": 256,
}

# AFSA 搜索相关参数（用于记录）
AFSA_SEARCH_TIME = 300  # 估计搜索耗时（秒）
FAST_EVAL_TIME = 60  # 每次快速评估耗时估计（秒）

FULL_TRAIN_EPOCHS = 50
FULL_PATIENCE = 8


def train_final_model(best_params: dict, data: dict, seq_len: int, n_features: int, device, logger):
    """使用最优参数进行完整训练"""
    logger.info("=" * 60)
    logger.info("开始完整训练 AFSA-PatchTST（已知最优参数）")
    logger.info("最优参数: %s", best_params)
    logger.info("=" * 60)

    model = build_model(
        "patchtst",
        n_features=n_features,
        seq_len=seq_len,
        patch_len=best_params["patch_len"],
        stride=best_params["stride"],
        d_model=best_params["d_model"],
        n_heads=best_params["n_heads"],
        num_layers=best_params["num_layers"],
        dropout=best_params["dropout"],
    ).to(device)

    train_loader = make_loader(
        data["X_train_seq"], data["y_train"], 
        batch_size=best_params["batch_size"], shuffle=True
    )
    val_loader = make_loader(
        data["X_val_seq"], data["y_val"], 
        batch_size=best_params["batch_size"], shuffle=False
    )

    logger.info("训练样本数: %d", len(data["y_train"]))
    logger.info("验证样本数: %d", len(data["y_val"]))
    logger.info("测试样本数: %d", len(data["y_test"]))

    start_time = time.time()
    model, history = train_with_early_stop(
        model,
        train_loader,
        val_loader,
        lr=best_params["learning_rate"],
        max_epochs=FULL_TRAIN_EPOCHS,
        patience=FULL_PATIENCE,
        device=device,
    )
    train_time = time.time() - start_time

    best = min(history, key=lambda x: x["val_loss"])
    logger.info("完整训练完成")
    logger.info("最佳 epoch: %d", best["epoch"])
    logger.info("最佳验证 loss: %.6f", best["val_loss"])
    logger.info("训练耗时: %.1f 秒", train_time)

    return model, history, train_time


def main():
    logger = setup_logger("EXP-P03-AFSA-Final", LOG_NAME)
    cfg = load_config()
    set_seed(cfg["random_seed"])
    ensure_dirs()
    device = get_device()
    logger.info("设备: %s", device)

    # 加载数据
    data = load_samples()
    seq_len = data["X_train_seq"].shape[1]
    n_features = data["X_train_seq"].shape[2]
    logger.info("输入形状: [batch, %d, %d]", seq_len, n_features)

    # 完整训练
    model, history, train_time = train_final_model(
        BEST_PARAMS, data, seq_len, n_features, device, logger
    )

    # 保存模型
    torch.save(model.state_dict(), MODELS_DIR / "afsa_patchtst.pt")
    logger.info("模型已保存: models/afsa_patchtst.pt")

    # 保存训练历史
    save_train_history("afsa_patchtst", history)
    logger.info("训练历史已保存: metrics/afsa_patchtst_train_history.csv")

    # 生成预测
    y_pred = predict(model, data["X_test_seq"], device)
    save_predictions("afsa_patchtst", data["y_test"], y_pred)
    logger.info("预测结果已保存: predictions/afsa_patchtst_test.csv")

    # 计算测试集指标
    metrics = compute_all_metrics(data["y_test"], y_pred)
    logger.info("测试集指标:")
    logger.info("  MAE:  %.6f", metrics["MAE"])
    logger.info("  RMSE: %.6f", metrics["RMSE"])
    logger.info("  MAPE: %.2f%%", metrics["MAPE"])
    logger.info("  R2:   %.6f", metrics["R2"])

    # 保存指标
    import pandas as pd
    metrics_df = pd.DataFrame([{
        "model_key": "afsa_patchtst",
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "MAPE": metrics["MAPE"],
        "R2": metrics["R2"],
        "display_name": "AFSA-PatchTST",
        "training_time_sec": round(train_time, 2),
        "search_time_sec": AFSA_SEARCH_TIME,
    }])
    metrics_path = METRICS_DIR / "afsa_patchtst_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info("指标已保存: %s", metrics_path.name)

    # 摘要日志
    append_log_summary(LOG_NAME, [
        "=" * 60,
        "【EXP-P03-AFSA-Final 摘要】",
        f"- 最优参数: {BEST_PARAMS}",
        f"- 训练耗时: {train_time:.1f} 秒",
        f"- 测试集 MAE: {metrics['MAE']:.6f}",
        f"- 测试集 RMSE: {metrics['RMSE']:.6f}",
        f"- 测试集 MAPE: {metrics['MAPE']:.2f}%",
        f"- 测试集 R2: {metrics['R2']:.6f}",
        "=" * 60,
    ])

    logger.info("训练完成！")


if __name__ == "__main__":
    main()
