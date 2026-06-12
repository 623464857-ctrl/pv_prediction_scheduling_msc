"""EXP-P03 人工鱼群算法（AFSA）用于 PatchTST 超参数搜索。"""

from __future__ import annotations

import itertools
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from exp_p03_common import (
    MODELS_DIR,
    METRICS_DIR,
    FIGURES_DIR,
    append_log_summary,
    compute_all_metrics,
    ensure_dirs,
    load_config,
    load_samples,
    load_test_timestamps,
    save_predictions,
    save_train_history,
    set_seed,
    setup_logger,
)
from exp_p03_models import PatchTSTRegressor, build_model
from exp_p03_torch_utils import get_device, make_loader, predict, train_with_early_stop


LOG_NAME = "EXP-P03_AFSA.log"

# 搜索空间
SEARCH_SPACE = {
    "patch_len": [2, 4, 8],
    "stride": [1, 2, 4],
    "d_model": [32, 64, 128],
    "n_heads": [2, 4, 8],
    "num_layers": [1, 2, 3],
    "dropout": [0.1, 0.2, 0.3],
    "learning_rate": [0.0005, 0.001, 0.002],
    "batch_size": [128, 256],
}

DEFAULT_AFSA_PARAMS = {
    "fish_num": 6,
    "max_iter": 5,
    "try_number": 5,
    "visual": 3,
    "step": 1.0,
    "crowd_factor": 0.6,
}

FAST_TRAIN_EPOCHS = 10
FULL_TRAIN_EPOCHS = 50
FAST_PATIENCE = 4
FULL_PATIENCE = 8


def sample_params(rng: random.Random) -> dict:
    return {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}


def params_to_key(params: dict) -> tuple:
    return (
        params["patch_len"],
        params["stride"],
        params["d_model"],
        params["n_heads"],
        params["num_layers"],
        params["dropout"],
        params["learning_rate"],
        params["batch_size"],
    )


def evaluate_params(params: dict, data: dict, seq_len: int, n_features: int, device, logger) -> float:
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

    train_loader = make_loader(data["X_train_seq"], data["y_train"], batch_size=params["batch_size"], shuffle=True)
    val_loader = make_loader(data["X_val_seq"], data["y_val"], batch_size=params["batch_size"], shuffle=False)

    _, history = train_with_early_stop(
        model,
        train_loader,
        val_loader,
        lr=params["learning_rate"],
        max_epochs=FAST_TRAIN_EPOCHS,
        patience=FAST_PATIENCE,
        device=device,
    )
    best = min(history, key=lambda x: x["val_loss"])
    val_loss = best["val_loss"]
    val_rmse = float(np.sqrt(val_loss))
    return val_rmse


def afsa_search(
    data: dict,
    seq_len: int,
    n_features: int,
    afsa_params: dict,
    rng: random.Random,
    device,
    logger,
) -> tuple[dict, float, list[dict]]:
    fish_num = afsa_params["fish_num"]
    max_iter = afsa_params["max_iter"]
    try_number = afsa_params["try_number"]
    visual = afsa_params["visual"]
    step_size = afsa_params["step"]
    crowd_factor = afsa_params["crowd_factor"]

    # 初始化鱼群
    fishes = [sample_params(rng) for _ in range(fish_num)]
    fitness = []
    history_records = []

    logger.info("AFSA 初始化鱼群: fish_num=%d, max_iter=%d", fish_num, max_iter)

    for idx, p in enumerate(fishes):
        rmse = evaluate_params(p, data, seq_len, n_features, device, logger)
        fitness.append(rmse)
        logger.info("  初始化鱼 %d: RMSE=%.6f | %s", idx, rmse, p)

    best_idx = int(np.argmin(fitness))
    global_best = fishes[best_idx].copy()
    global_best_fitness = fitness[best_idx]

    for idx, p in enumerate(fishes):
        history_records.append({
            "iteration": 0,
            "fish_id": idx,
            "val_rmse": fitness[idx],
            "best_val_rmse": global_best_fitness,
            **p,
        })

    for it in range(1, max_iter + 1):
        logger.info("迭代 %d/%d", it, max_iter)
        for i in range(fish_num):
            p = fishes[i].copy()
            # 找 visual 范围内的邻居
            neighbors = []
            for j in range(fish_num):
                if i == j:
                    continue
                key_i = params_to_key(p)
                key_j = params_to_key(fishes[j])
                dist = sum(a != b for a, b in zip(key_i, key_j))
                if dist <= visual:
                    neighbors.append((j, fitness[j], fishes[j]))

            improved = False

            # 追尾：向最佳邻居移动
            if neighbors:
                best_n = min(neighbors, key=lambda x: x[1])
                if best_n[1] < fitness[i]:
                    new_p = _move_towards(p, best_n[2], step_size, rng)
                    new_rmse = evaluate_params(new_p, data, seq_len, n_features, device, logger)
                    if new_rmse < fitness[i]:
                        fishes[i] = new_p
                        fitness[i] = new_rmse
                        improved = True

            # 聚群
            if not improved and neighbors:
                center = _compute_center(neighbors, fishes)
                center_fitness = evaluate_params(center, data, seq_len, n_features, device, logger)
                if center_fitness < crowd_factor * fitness[i]:
                    new_p = _move_towards(p, center, step_size, rng)
                    new_rmse = evaluate_params(new_p, data, seq_len, n_features, device, logger)
                    if new_rmse < fitness[i]:
                        fishes[i] = new_p
                        fitness[i] = new_rmse
                        improved = True

            # 觅食
            if not improved:
                for _ in range(try_number):
                    trial = sample_params(rng)
                    trial_rmse = evaluate_params(trial, data, seq_len, n_features, device, logger)
                    if trial_rmse < fitness[i]:
                        fishes[i] = trial
                        fitness[i] = trial_rmse
                        improved = True
                        break

            # 随机游走
            if not improved:
                fishes[i] = sample_params(rng)
                fitness[i] = evaluate_params(fishes[i], data, seq_len, n_features, device, logger)

        best_idx = int(np.argmin(fitness))
        if fitness[best_idx] < global_best_fitness:
            global_best = fishes[best_idx].copy()
            global_best_fitness = fitness[best_idx]
            logger.info("  全局最优更新于迭代 %d: RMSE=%.6f", it, global_best_fitness)

        for idx in range(fish_num):
            history_records.append({
                "iteration": it,
                "fish_id": idx,
                "val_rmse": fitness[idx],
                "best_val_rmse": global_best_fitness,
                **fishes[idx],
            })

    return global_best, global_best_fitness, history_records


def _move_towards(p: dict, target: dict, step_size: float, rng: random.Random) -> dict:
    new_p = {}
    for k in p:
        choices = SEARCH_SPACE[k]
        idx_t = choices.index(target[k])
        idx_p = choices.index(p[k])
        delta = idx_t - idx_p
        # 随机步长，不超过 |delta|
        if delta == 0:
            new_idx = idx_p
        else:
            step = rng.randint(1, min(abs(int(round(step_size * delta))), abs(delta)) or 1)
            direction = 1 if delta > 0 else -1
            new_idx = max(0, min(len(choices) - 1, idx_p + direction * step))
        new_p[k] = choices[new_idx]
    return new_p


def _compute_center(neighbors: list, fishes: list) -> dict:
    center = {}
    for key in SEARCH_SPACE.keys():
        values = [fishes[j][key] for j, _, _ in neighbors]
        # 取众数/首个
        center[key] = max(set(values), key=values.count)
    return center


def save_search_history(records: list[dict]) -> Path:
    path = METRICS_DIR / "afsa_patchtst_search_history.csv"
    pd.DataFrame(records).to_csv(path, index=False)
    return path


def plot_search_curve(records: list[dict]) -> Path:
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.DataFrame(records)
    out_path = FIGURES_DIR / "afsa_patchtst_search_curve.png"
    plt.figure(figsize=(8, 4))
    plt.plot(df["iteration"], df["val_rmse"], marker="o")
    plt.xlabel("iteration")
    plt.ylabel("best validation RMSE")
    plt.title("AFSA PatchTST Search Curve")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def run_full_training(
    best_params: dict,
    data: dict,
    seq_len: int,
    n_features: int,
    device,
    logger,
) -> tuple[nn.Module, list[dict]]:
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

    train_loader = make_loader(data["X_train_seq"], data["y_train"], batch_size=best_params["batch_size"], shuffle=True)
    val_loader = make_loader(data["X_val_seq"], data["y_val"], batch_size=best_params["batch_size"], shuffle=False)

    model, history = train_with_early_stop(
        model,
        train_loader,
        val_loader,
        lr=best_params["learning_rate"],
        max_epochs=FULL_TRAIN_EPOCHS,
        patience=FULL_PATIENCE,
        device=device,
    )
    return model, history


def main() -> None:
    logger = setup_logger("EXP-P03-AFSA", LOG_NAME)
    cfg = load_config()
    set_seed(cfg["random_seed"])
    ensure_dirs()
    device = get_device()
    logger.info("设备: %s", device)

    data = load_samples()
    seq_len = data["X_train_seq"].shape[1]
    n_features = data["X_train_seq"].shape[2]
    logger.info("输入: [batch, %d, %d]", seq_len, n_features)

    afsa_params = {**DEFAULT_AFSA_PARAMS}
    rng = random.Random(cfg["random_seed"])

    logger.info("开始 AFSA 搜索...")
    best_params, best_rmse, history_records = afsa_search(
        data, seq_len, n_features, afsa_params, rng, device, logger
    )
    logger.info("AFSA 搜索完成 | 最优 RMSE=%.6f | 参数=%s", best_rmse, best_params)

    # 保存搜索记录
    save_search_history(history_records)
    plot_search_curve(history_records)

    best_params_path = METRICS_DIR / "afsa_patchtst_best_params.json"
    with open(best_params_path, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2, ensure_ascii=False)
    logger.info("最优参数已保存: %s", best_params_path.name)

    # 用最优参数完整训练
    logger.info("使用最优参数完整训练 PatchTST...")
    model, history = run_full_training(best_params, data, seq_len, n_features, device, logger)
    best = min(history, key=lambda x: x["val_loss"])
    logger.info("完整训练最佳 epoch=%d | val_loss=%.6f", best["epoch"], best["val_loss"])

    torch.save(model.state_dict(), MODELS_DIR / "afsa_patchtst.pt")
    save_train_history("afsa_patchtst", history)

    y_pred = predict(model, data["X_test_seq"], device)
    save_predictions("afsa_patchtst", data["y_test"], y_pred)
    logger.info("AFSA-PatchTST 预测已保存")

    append_log_summary(
        LOG_NAME,
        [
            "=" * 60,
            "【EXP-P03-AFSA 摘要】",
            f"- 最优验证 RMSE: {best_rmse:.6f}",
            f"- 最优参数: {best_params}",
            f"- 完整训练最佳 epoch: {best['epoch']}",
            "- 产出: models/afsa_patchtst.pt, metrics/afsa_patchtst_*.csv, figures/afsa_patchtst_search_curve.png",
            "=" * 60,
        ],
    )
    logger.info("EXP-P03-AFSA 结束")


if __name__ == "__main__":
    main()
