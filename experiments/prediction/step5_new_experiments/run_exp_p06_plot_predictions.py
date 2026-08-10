"""
Step 6: 绘制多Horizon预测曲线对比图 (H1, H4, H16)
为每个预测尺度生成一张图，每个图包含多个模型的预测曲线。

输出:
    data/prediction/step5_new_experiments/figures/{h1,h4,h16}/multi_horizon_predictions.png
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Use non-interactive backend
matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.prediction.step5_new_experiments.exp_p05_common import (
    FIGURES_DIR,
    PRED_DIR,
    MODEL_DISPLAY_NAMES,
    setup_logger,
)

# Configure Chinese font on Windows
_chinese_font = "C:\\Windows\\Fonts\\simhei.ttf"
if Path(_chinese_font).exists():
    fm.fontManager.addfont(_chinese_font)
    matplotlib.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

# 预测尺度标签
HORIZON_LABELS = {
    1: "15min (H1)",
    4: "1h (H4)",
    16: "4h (H16)"
}

# 要展示的模型（选择有代表性的模型，避免图太拥挤）
DISPLAY_MODELS = [
    "xgboost",
    "lightgbm",
    "cnn_bilstm_residual_optuna",
]

# 每个模型的显示名称
MODEL_DISPLAY_NAMES.update({
    "cnn_bilstm_residual_optuna": "CNN-BiLSTM (Residual+Opt)",
})

# 模型颜色配置
MODEL_COLORS = {
    "xgboost": "#d62728",              # 红色
    "lightgbm": "#9467bd",             # 紫色
    "cnn_bilstm_residual_optuna": "#bcbd22", # 黄绿色
    "ridge": "#2ca02c",
}


def load_predictions(horizon: int) -> dict:
    """加载指定horizon的所有预测数据"""
    hdir = PRED_DIR / f"h{horizon}"
    if not hdir.exists():
        return {}

    predictions = {}
    for model_key in DISPLAY_MODELS:
        # 尝试不同的文件名模式
        for pattern in [
            f"{model_key}_test.csv",
            f"{model_key.replace('_optuna', '')}_optuna_test.csv",
        ]:
            path = hdir / pattern
            if path.exists():
                try:
                    df = pd.read_csv(path)
                    if "timestamp" in df.columns:
                        df["timestamp"] = pd.to_datetime(df["timestamp"])
                        df = df.sort_values("timestamp")
                    predictions[model_key] = df
                    break
                except Exception:
                    continue
    return predictions


def plot_single_horizon(horizon: int, n_points: int = 500) -> Path | None:
    """为一个预测尺度绘制预测曲线图"""
    predictions = load_predictions(horizon)

    if not predictions:
        print(f"[H{horizon}] 警告: 没有找到预测数据")
        return None

    # 找真值（从第一个模型的DataFrame获取y_true）
    true_df = None
    for model_key, df in predictions.items():
        if "y_true" in df.columns:
            true_df = df
            break

    if true_df is None:
        print(f"[H{horizon}] 警告: 没有找到真值数据")
        return None

    # 创建图形
    fig, ax = plt.subplots(figsize=(16, 6))

    # 绘制真值曲线（黑色，加粗）
    sub_true = true_df.iloc[:n_points]
    ax.plot(
        range(len(sub_true)),
        sub_true["y_true"].values,
        label="True Value",
        color="black",
        linewidth=2.5,
        alpha=0.95,
        zorder=10
    )

    # 绘制各模型预测曲线
    for model_key in DISPLAY_MODELS:
        if model_key not in predictions:
            continue

        df = predictions[model_key]
        if "y_pred" not in df.columns:
            continue

        sub = df.iloc[:n_points]
        color = MODEL_COLORS.get(model_key, "#888888")

        ax.plot(
            range(len(sub)),
            sub["y_pred"].values,
            label=MODEL_DISPLAY_NAMES.get(model_key, model_key),
            color=color,
            linewidth=1.3,
            alpha=0.8
        )

    # 设置坐标轴
    ax.set_xlabel("Sample Index", fontsize=12)
    ax.set_ylabel("Normalized Power", fontsize=12)
    ax.set_title(
        f"Multi-Model Prediction Comparison - Horizon {horizon} ({HORIZON_LABELS[horizon]})",
        fontsize=14,
        fontweight="bold"
    )

    # 图例放在右下角，分两列
    ax.legend(
        fontsize=9,
        loc="lower left",
        ncol=4,
        framealpha=0.9
    )

    ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()

    # 保存图片
    out_dir = FIGURES_DIR / f"h{horizon}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"multi_horizon_predictions_H{horizon}.png"

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[H{horizon}] 保存: {out_path}")
    return out_path


def plot_all_horizons(n_points: int = 400) -> list[Path]:
    """为所有预测尺度绘制预测曲线图"""
    results = []

    for horizon in [1, 4, 16]:
        path = plot_single_horizon(horizon, n_points=n_points)
        if path:
            results.append(path)

    return results


def plot_combined_view(n_points: int = 300) -> Path | None:
    """创建一个组合视图，三张子图垂直排列"""
    fig, axes = plt.subplots(3, 1, figsize=(16, 14))

    horizons = [1, 4, 16]

    for idx, horizon in enumerate(horizons):
        ax = axes[idx]
        predictions = load_predictions(horizon)

        if not predictions:
            ax.set_title(f"H{horizon} ({HORIZON_LABELS[horizon]}) - No Data")
            continue

        # 找真值
        true_df = None
        for model_key, df in predictions.items():
            if "y_true" in df.columns:
                true_df = df
                break

        if true_df is None:
            ax.set_title(f"H{horizon} ({HORIZON_LABELS[horizon]}) - No Ground Truth")
            continue

        # 绘制真值
        sub_true = true_df.iloc[:n_points]
        ax.plot(
            range(len(sub_true)),
            sub_true["y_true"].values,
            label="True Value",
            color="black",
            linewidth=2.2,
            alpha=0.95,
            zorder=10
        )

        # 绘制各模型预测
        for model_key in DISPLAY_MODELS:
            if model_key not in predictions:
                continue

            df = predictions[model_key]
            if "y_pred" not in df.columns:
                continue

            sub = df.iloc[:n_points]
            color = MODEL_COLORS.get(model_key, "#888888")

            ax.plot(
                range(len(sub)),
                sub["y_pred"].values,
                label=MODEL_DISPLAY_NAMES.get(model_key, model_key),
                color=color,
                linewidth=1.2,
                alpha=0.8
            )

        ax.set_xlabel("Sample Index", fontsize=10)
        ax.set_ylabel("Power", fontsize=10)
        ax.set_title(
            f"Horizon {horizon} ({HORIZON_LABELS[horizon]})",
            fontsize=12,
            fontweight="bold"
        )
        ax.legend(fontsize=7, loc="lower left", ncol=4)
        ax.grid(alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()

    # 保存组合图
    out_path = FIGURES_DIR / "combined_multi_horizon_predictions.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"保存组合图: {out_path}")
    return out_path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(__name__)

    print("=" * 60)
    print("开始绘制多Horizon预测曲线对比图")
    print("=" * 60)

    # 绘制单独的三张图
    print("\n[1/2] 绘制单独的H1, H4, H16预测曲线图...")
    individual_paths = plot_all_horizons(n_points=400)

    # 绘制组合图
    print("\n[2/2] 绘制组合视图...")
    combined_path = plot_combined_view(n_points=300)

    print("\n" + "=" * 60)
    print("绘图完成!")
    print(f"生成图片数量: {len(individual_paths) + (1 if combined_path else 0)}")
    print("=" * 60)

    for p in individual_paths:
        print(f"  - {p}")
    if combined_path:
        print(f"  - {combined_path}")
