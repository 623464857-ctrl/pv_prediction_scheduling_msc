"""
实验编号: EXP-P03-summarize
实验名称: 混合深度学习模型对比 — 汇总与可视化
实验目的: 汇总所有模型测试指标、生成对比图表、输出 Markdown 报告
运行方式: python experiments/prediction/step3_hybrid_models/run_exp_p03_summarize.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from exp_p03_common import (
    MODEL_DISPLAY_NAMES,
    MODEL_ORDER,
    append_log_summary,
    compute_all_metrics,
    ensure_dirs,
    load_config,
    load_test_timestamps,
    plot_loss_curve,
    plot_overlay,
    plot_pred_curve,
    plot_metrics_bar,
    plot_training_time_comparison,
    save_predictions,
    set_seed,
    setup_logger,
)

LOG_NAME = "EXP-P03_summarize.log"


def gather_metrics(preds_dir) -> pd.DataFrame:
    rows = []
    for key, label in MODEL_DISPLAY_NAMES.items():
        pred_file = preds_dir / f"{key}_test.csv"
        if not pred_file.exists():
            continue
        df = pd.read_csv(pred_file)
        y_true = df["y_true"].values
        y_pred = df["y_pred"].values
        metrics = compute_all_metrics(y_true, y_pred)
        metrics["model_key"] = key
        metrics["display_name"] = label
        rows.append(metrics)
    return pd.DataFrame(rows).set_index("model_key")


def gather_training_times(metrics_dir) -> pd.Series:
    times = {}
    for key in MODEL_DISPLAY_NAMES:
        hist_file = metrics_dir / f"{key}_train_history.csv"
        if not hist_file.exists():
            continue
        df = pd.read_csv(hist_file)
        times[key] = df["epoch"].iloc[-1]
    return pd.Series(times, name="epochs")


def write_report(metrics_df: pd.DataFrame, out_path: Path) -> None:
    lines = [
        "# EXP-P03 混合深度学习模型对比 — 实验报告",
        "",
        "## 1. 实验概述",
        "",
        "- 实验编号: EXP-P03",
        "- 站点: Site_1",
        "- 任务: 光伏功率归一化值预测（power_pu）",
        "- 划分: 时序 70% 训练 / 14% 验证 / 30% 测试",
        "- 输入窗口: lookback=16, horizon=1",
        "- 特征数: 13",
        "",
        "## 2. 模型汇总",
        "",
    ]
    for key in MODEL_ORDER:
        if key not in metrics_df.index:
            continue
        row = metrics_df.loc[key]
        lines.extend([
            f"### {row['display_name']}",
            "",
            f"- MAE: {row['MAE']:.6f}",
            f"- RMSE: {row['RMSE']:.6f}",
            f"- MAPE: {row['MAPE']:.2f}%",
            f"- R²: {row['R2']:.6f}",
            "",
        ])

    best = metrics_df["RMSE"].idxmin()
    lines.extend([
        "## 3. 结论",
        "",
        f"- 测试集 RMSE 最低模型: **{metrics_df.loc[best, 'display_name']}** (RMSE={metrics_df.loc[best, 'RMSE']:.6f})",
        f"- 测试集 R² 最高模型: **{metrics_df.loc[metrics_df['R2'].idxmax(), 'display_name']}** (R²={metrics_df.loc[metrics_df['R2'].idxmax(), 'R2']:.6f})",
        "",
        "## 4. 产出文件",
        "",
        "- `data/prediction/step3_hybrid_models/metrics/*.csv`",
        "- `data/prediction/step3_hybrid_models/predictions/*.csv`",
        "- `data/prediction/step3_hybrid_models/figures/*.png`",
        "- `data/prediction/step3_hybrid_models/models/*.pt`",
        "",
    ])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    logger = setup_logger("EXP-P03-summarize", LOG_NAME)
    cfg = load_config()
    set_seed(cfg["random_seed"])
    ensure_dirs()

    from exp_p03_common import (
        METRICS_DIR,
        PRED_DIR,
        FIGURES_DIR,
        REPORTS_DIR,
        MODELS_DIR,
    )

    logger.info("开始汇总...")

    metrics_df = gather_metrics(PRED_DIR)
    metrics_path = METRICS_DIR / "exp_p03_model_comparison.csv"
    metrics_df.to_csv(metrics_path)
    logger.info("指标汇总: %s", metrics_path.name)

    plot_metrics_bar(metrics_df.reset_index(), FIGURES_DIR / "metrics_comparison.png")
    logger.info("指标对比图已生成")

    epochs = gather_training_times(METRICS_DIR)
    plot_training_time_comparison(metrics_df.join(epochs.rename("training_time_sec")), FIGURES_DIR / "training_time_comparison.png")
    logger.info("训练时间对比图已生成")

    preds = {}
    for key in MODEL_ORDER:
        pred_file = PRED_DIR / f"{key}_test.csv"
        if pred_file.exists():
            preds[key] = pd.read_csv(pred_file)
    if preds:
        plot_overlay(preds, n_points=5 * 96, out_path=FIGURES_DIR / "prediction_overlay_all_models.png")
        logger.info("预测曲线叠加图已生成")

    write_report(metrics_df, REPORTS_DIR / "exp_p03_report.md")
    logger.info("报告已生成: %s", REPORTS_DIR / "exp_p03_report.md")

    logger.info("汇总完成")
    logger.info("最佳模型: %s (RMSE=%.6f)", metrics_df.loc[metrics_df["RMSE"].idxmin(), "display_name"], metrics_df["RMSE"].min())


if __name__ == "__main__":
    main()
