"""
实验编号: EXP-P02-summarize
实验名称: 基础预测模型对比 — 结果汇总
实验目的: 统一计算五模型测试集误差指标，绘制损失/预测曲线，输出初步结论
运行方式: python experiments/prediction/step2_baseline_models/run_exp_p02_summarize_results.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from exp_p02_common import (  # noqa: E402
    FIGURES_DIR,
    METRICS_DIR,
    PRED_DIR,
    REPORTS_DIR,
    append_log_summary,
    ensure_dirs,
    setup_logger,
)

LOG_NAME = "EXP-P02_summarize.log"

MODELS = [
    ("bp", "BP"),
    ("svr", "SVR"),
    ("randomforest", "Random Forest"),
    ("lstm", "LSTM"),
    ("bilstm", "BiLSTM"),
]

PLOT_DAYS = 5  # 测试集展示连续 5 天


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    mask = np.abs(y_true) > 0.01
    if mask.any():
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    else:
        mape = float("nan")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


def plot_loss(history_path: Path, title: str, out_path: Path) -> None:
    if not history_path.exists():
        return
    df = pd.read_csv(history_path)
    plt.figure(figsize=(8, 4))
    plt.plot(df["epoch"], df["train_loss"], label="train_loss")
    plt.plot(df["epoch"], df["val_loss"], label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("MSE loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_single_pred(df: pd.DataFrame, title: str, out_path: Path, n_points: int) -> None:
    sub = df.iloc[:n_points]
    plt.figure(figsize=(12, 4))
    plt.plot(sub["timestamp"], sub["y_true"], label="actual", linewidth=1.2)
    plt.plot(sub["timestamp"], sub["y_pred"], label="predicted", linewidth=1.0, alpha=0.85)
    plt.xlabel("timestamp")
    plt.ylabel("power_pu")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_overlay(preds: dict[str, pd.DataFrame], n_points: int, out_path: Path) -> None:
    first = next(iter(preds.values()))
    sub_ts = first["timestamp"].iloc[:n_points]
    y_true = first["y_true"].iloc[:n_points]

    plt.figure(figsize=(14, 5))
    plt.plot(sub_ts, y_true, label="actual", color="black", linewidth=1.5)
    colors = ["C0", "C1", "C2", "C3", "C4"]
    for (key, label), color in zip(MODELS, colors):
        if key not in preds:
            continue
        sub = preds[key].iloc[:n_points]
        plt.plot(sub_ts, sub["y_pred"], label=label, alpha=0.8, color=color)
    plt.xlabel("timestamp")
    plt.ylabel("power_pu")
    plt.title("Test set predictions — all models (same window)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def write_conclusion(metrics_df: pd.DataFrame, out_path: Path) -> None:
    best_rmse = metrics_df.loc[metrics_df["RMSE"].idxmin()]
    best_mae = metrics_df.loc[metrics_df["MAE"].idxmin()]
    best_r2 = metrics_df.loc[metrics_df["R2"].idxmax()]

    lines = [
        "# EXP-P02 基础模型对比 — 初步结论",
        "",
        "## 1. 是否达成实验目标",
        "",
        "五类模型（BP、SVR、Random Forest、LSTM、BiLSTM）已在 Site_1 统一样本与测试集上完成单步 `power_pu` 预测，",
        "并输出训练损失曲线（BP/LSTM/BiLSTM）、测试集预测曲线与误差指标表，满足本周第一批基线对比实验交付要求。",
        "",
        "## 2. 当前最优基础模型",
        "",
        f"- **RMSE 最低**：{best_rmse['display_name']}（RMSE={best_rmse['RMSE']:.6f}）",
        f"- **MAE 最低**：{best_mae['display_name']}（MAE={best_mae['MAE']:.6f}）",
        f"- **R² 最高**：{best_r2['display_name']}（R²={best_r2['R2']:.6f}）",
        "",
        "综合误差指标与预测曲线，建议以 RMSE/MAE 与峰值段拟合情况共同判断最优基线。",
        "",
        "## 3. 模型差异与后续基线建议",
        "",
        "- **BP**：浅层前馈网络，训练快但时序建模能力有限，适合作为传统 NN 参照。",
        "- **SVR**：核方法，预测曲线可能偏平滑，需关注峰值段是否低估。",
        "- **Random Forest**：非线性树模型，需观察是否存在阶梯式预测。",
        "- **LSTM / BiLSTM**：显式时序建模；若 BiLSTM 相对 LSTM 有稳定收益，可作为深度学习改进基线。",
        "",
        "下一阶段复杂模型应在本实验最优基线之上做对照，而非直接对比原始数据或未统一划分的样本。",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    logger = setup_logger("EXP-P02-summarize", LOG_NAME)
    ensure_dirs()
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    preds: dict[str, pd.DataFrame] = {}
    rows = []

    for key, display in MODELS:
        path = PRED_DIR / f"{key}_test.csv"
        if not path.exists():
            logger.error("缺少预测文件: %s", path.name)
            sys.exit(1)
        df = pd.read_csv(path, parse_dates=["timestamp"])
        preds[key] = df
        m = compute_metrics(df["y_true"].to_numpy(), df["y_pred"].to_numpy())
        rows.append({"model": key, "display_name": display, **m})
        logger.info("%s | MAE=%.6f RMSE=%.6f MAPE=%.2f%% R2=%.6f", display, m["MAE"], m["RMSE"], m["MAPE"], m["R2"])

    metrics_df = pd.DataFrame(rows)
    metrics_path = METRICS_DIR / "baseline_comparison_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    n_points = min(PLOT_DAYS * 96, len(next(iter(preds.values()))))
    logger.info("预测曲线展示窗口: 前 %d 个测试点 (约 %d 天)", n_points, PLOT_DAYS)

    for key, display in MODELS:
        plot_single_pred(preds[key], f"{display} — test predictions", FIGURES_DIR / f"pred_{key}.png", n_points)

    plot_overlay(preds, n_points, FIGURES_DIR / "pred_all_models_overlay.png")

    for key, display in [("bp", "BP"), ("lstm", "LSTM"), ("bilstm", "BiLSTM")]:
        plot_loss(METRICS_DIR / f"{key}_train_history.csv", f"{display} training loss", FIGURES_DIR / f"loss_{key}.png")

    # 指标柱状图
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    x = np.arange(len(metrics_df))
    for ax, col in zip(axes, ["MAE", "RMSE", "R2"]):
        ax.bar(x, metrics_df[col])
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_df["display_name"], rotation=20, ha="right")
        ax.set_title(col)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "metrics_bar_comparison.png", dpi=150)
    plt.close()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    conclusion_path = REPORTS_DIR / "EXP-P02_preliminary_conclusion.md"
    write_conclusion(metrics_df, conclusion_path)

    best = metrics_df.loc[metrics_df["RMSE"].idxmin()]
    append_log_summary(
        LOG_NAME,
        [
            "=" * 60,
            "【EXP-P02-summarize 摘要】",
            f"- 五模型误差表: metrics/baseline_comparison_metrics.csv",
            f"- RMSE 最优: {best['display_name']} ({best['RMSE']:.6f})",
            f"- 图表: figures/loss_*.png, pred_*.png, metrics_bar_comparison.png",
            f"- 结论: reports/EXP-P02_preliminary_conclusion.md",
            "=" * 60,
        ],
    )
    logger.info("汇总完成 | 指标表: %s", metrics_path.name)
    logger.info("EXP-P02-summarize 结束")


if __name__ == "__main__":
    main()
