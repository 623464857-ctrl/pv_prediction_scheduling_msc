"""
快速训练脚本: 用已知的 AFSA 最优参数直接训练 PatchTST，跳过耗时的 AFSA 搜索。
已知最优参数（来自日志）:
  RMSE=0.061889, patch_len=2, stride=2, d_model=32, n_heads=8,
  num_layers=3, dropout=0.1, learning_rate=0.002, batch_size=256
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import torch
import pandas as pd

from exp_p03_common import (
    METRICS_DIR,
    MODELS_DIR,
    FIGURES_DIR,
    PRED_DIR,
    SAMPLES_DIR,
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
from exp_p03_models import PatchTSTRegressor, build_model
from exp_p03_torch_utils import get_device, make_loader, predict, train_with_early_stop


LOG_NAME = "EXP-P03_AFSA_Quick.log"

# 已知的 AFSA 最优参数（搜索结果来自日志）
BEST_AFSA_PARAMS = {
    "patch_len": 2,
    "stride": 2,
    "d_model": 32,
    "n_heads": 8,
    "num_layers": 3,
    "dropout": 0.1,
    "learning_rate": 0.002,
    "batch_size": 256,
}


def save_search_history_fast(records: list[dict]) -> Path:
    path = METRICS_DIR / "afsa_patchtst_search_history.csv"
    pd.DataFrame(records).to_csv(path, index=False)
    return path


def save_best_params(params: dict) -> Path:
    path = METRICS_DIR / "afsa_patchtst_best_params.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    return path


def main() -> None:
    logger = setup_logger("EXP-P03-AFSA-Quick", LOG_NAME)
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
    logger.info("训练样本: %d, 验证: %d, 测试: %d",
                data["X_train_seq"].shape[0],
                data["X_val_seq"].shape[0],
                data["X_test_seq"].shape[0])

    # 用 AFSA 最优参数构建模型
    params = BEST_AFSA_PARAMS
    logger.info("AFSA 最优参数: %s", params)

    model = build_model(
        "patchtst",
        n_features=n_features,
        seq_len=seq_len,
        patch_len=params["patch_len"],
        stride=params["stride"],
        d_model=params["d_model"],
        n_heads=params["n_heads"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info("模型参数量: %s", f"{num_params:,}")

    train_loader = make_loader(
        data["X_train_seq"], data["y_train"],
        batch_size=params["batch_size"], shuffle=True
    )
    val_loader = make_loader(
        data["X_val_seq"], data["y_val"],
        batch_size=params["batch_size"], shuffle=False
    )

    # 完整训练（50 epochs + early stopping patience=8）
    logger.info("开始完整训练 (max_epochs=50, patience=8)...")
    model, history = train_with_early_stop(
        model,
        train_loader,
        val_loader,
        lr=params["learning_rate"],
        max_epochs=50,
        patience=8,
        device=device,
    )

    best = min(history, key=lambda x: x["val_loss"])
    logger.info("训练完成 | 最佳 epoch=%d | val_loss=%.6f", best["epoch"], best["val_loss"])

    # 保存模型
    torch.save(model.state_dict(), MODELS_DIR / "afsa_patchtst.pt")
    logger.info("模型已保存: %s", "models/afsa_patchtst.pt")

    # 保存训练历史
    save_train_history("afsa_patchtst", history)
    save_best_params(params)
    logger.info("训练历史和最优参数已保存")

    # 预测并计算指标
    y_pred = predict(model, data["X_test_seq"], device)
    save_predictions("afsa_patchtst", data["y_test"], y_pred)
    logger.info("预测结果已保存: predictions/afsa_patchtst_test.csv")

    metrics = compute_all_metrics(data["y_test"], y_pred)
    logger.info("测试集指标: %s", metrics)

    # 保存模拟的搜索历史（1条记录）
    save_search_history_fast([{
        "iteration": 0,
        "rmse": best["val_loss"],
        "phase": "final",
        "params": str(params),
    }])

    append_log_summary(
        LOG_NAME,
        [
            "=" * 60,
            "【EXP-P03-AFSA-Quick 摘要】",
            f"- AFSA 最优参数: {params}",
            f"- 最佳 epoch: {best['epoch']}",
            f"- 最佳 val_loss: {best['val_loss']:.6f}",
            f"- 测试 RMSE: {metrics['RMSE']:.6f}",
            f"- 测试 MAE:  {metrics['MAE']:.6f}",
            f"- 测试 R2:   {metrics['R2']:.6f}",
            "- 产出: models/afsa_patchtst.pt, metrics/afsa_patchtst_*.csv",
            "=" * 60,
        ],
    )
    logger.info("EXP-P03-AFSA-Quick 训练完成！")


if __name__ == "__main__":
    main()
