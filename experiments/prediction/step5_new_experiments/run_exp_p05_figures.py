"""
EXP-P05 图表生成（合并版）：参考 Step4 生成精简的代表性图表。
python experiments/prediction/step5_new_experiments/run_exp_p05_figures.py --horizon 1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
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
    BENCHMARK_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    MODEL_DISPLAY_NAMES,
    PRED_DIR,
    REPORTS_DIR,
    ensure_dirs,
    load_config,
    setup_logger,
)

# Configure Chinese font on Windows
_chinese_font = "C:\\Windows\\Fonts\\simhei.ttf"
if Path(_chinese_font).exists():
    fm.fontManager.addfont(_chinese_font)
    matplotlib.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

HORIZON_LABELS = {1: "15min", 4: "1h", 16: "4h"}

ALL_KEYS = [
    "persistence", "moving_average", "ridge", "xgboost", "lightgbm",
    "lstm_residual", "bilstm_residual", "cnn_lstm_residual",
    "cnn_bilstm_residual", "patchtst_residual",
]

MODEL_COLORS = {
    "persistence": "#1f77b4",
    "moving_average": "#ff7f0e",
    "ridge": "#2ca02c",
    "xgboost": "#d62728",
    "lightgbm": "#9467bd",
    "lstm_residual": "#8c564b",
    "bilstm_residual": "#e377c2",
    "cnn_lstm_residual": "#7f7f7f",
    "cnn_bilstm_residual": "#bcbd22",
    "patchtst_residual": "#17becf",
}


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_predictions(horizon: int) -> list[dict]:
    """收集所有预测 CSV，返回 [{key, df}]"""
    hdir = PRED_DIR / f"h{horizon}"
    records = []
    for p in sorted(hdir.glob("*_test.csv")) if hdir.exists() else []:
        key = p.name.replace("_test.csv", "")
        try:
            df = pd.read_csv(p)
            records.append({"key": key, "df": df})
        except Exception:
            continue
    return records


def _build_metrics_df(horizon: int) -> pd.DataFrame:
    """构建指标 DataFrame：daytime-only RMSE 优先排名"""
    hdir = METRICS_DIR / f"h{horizon}"
    segmented = load_json(hdir / "segmented_metrics.json")
    baseline = load_json(hdir / "baseline_metrics.json")
    residual = load_json(hdir / "residual_metrics.json")

    rows = []
    for key, m in (baseline or {}).items():
        seg = (segmented or {}).get(key, {})
        src = seg.get("daytime_only", m)
        rows.append({
            "key": key,
            "Model": MODEL_DISPLAY_NAMES.get(key, key),
            "RMSE": src.get("RMSE", np.nan),
            "MAE": src.get("MAE", np.nan),
            "MAPE": src.get("MAPE", np.nan),
            "R2": src.get("R2", np.nan),
        })
    for key, m in (residual or {}).items():
        seg = (segmented or {}).get(key, {})
        src = seg.get("daytime_only", m)
        rows.append({
            "key": key,
            "Model": MODEL_DISPLAY_NAMES.get(key, key),
            "RMSE": src.get("RMSE", np.nan),
            "MAE": src.get("MAE", np.nan),
            "MAPE": src.get("MAPE", np.nan),
            "R2": src.get("R2", np.nan),
        })
    return pd.DataFrame(rows)


# ─── 1. 综合指标对比（4-panel） ───────────────────────────────
def plot_metrics_comparison(horizon: int, logger: logging.Logger) -> Path | None:
    df = _build_metrics_df(horizon)
    if df.empty:
        logger.warning("无指标数据，跳过")
        return None
    df = df.sort_values(["RMSE", "MAE"], na_position="last").reset_index(drop=True)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    fig.suptitle(f"EXP-P05 指标对比 (h{horizon} {HORIZON_LABELS[horizon]})",
                 fontsize=13, fontweight="bold")
    for ax, col, ylabel in zip(axes, ["RMSE", "MAE", "MAPE", "R2"],
                               ["RMSE ↓", "MAE ↓", "MAPE ↓", "R2 ↑"]):
        vals = df[col].fillna(0).values
        bars = ax.bar(df["Model"], vals, color="#4c78a8", alpha=0.85)
        ax.set_title(ylabel, fontsize=11)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df["Model"], rotation=35, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, vals):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + max(vals) * 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = FIGURES_DIR / f"h{horizon}" / f"h{horizon}_metrics_comparison.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("保存 %s", out.name)
    return out


# ─── 2. 多模型叠加预测曲线 ───────────────────────────────────
def plot_overlay(horizon: int, logger: logging.Logger, n_points: int = 600) -> Path | None:
    records = _collect_predictions(horizon)
    if not records:
        logger.warning("无预测数据，跳过")
        return None

    # 找第一个有 y_true/y_pred 的记录作为真值来源
    true_df = None
    for rec in records:
        df = rec["df"]
        if "y_true" in df.columns and "y_pred" in df.columns:
            if "timestamp" in df.columns:
                try:
                    df = df.copy()
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df = df.sort_values("timestamp")
                except Exception:
                    pass
            true_df = df
            break

    if true_df is None:
        logger.warning("无真值数据，跳过叠加图")
        return None

    # 按 ALL_KEYS 顺序过滤可用的模型
    available = {rec["key"].replace("_test", ""): rec["df"] for rec in records}
    available_raw = {rec["key"]: rec["df"] for rec in records}
    ordered_keys = [k for k in ALL_KEYS if k in available or (k + "_test") in available_raw]

    fig, ax = plt.subplots(figsize=(14, 5))
    sub_true = true_df.iloc[:n_points]
    ax.plot(range(len(sub_true)), sub_true["y_true"].values,
            label="True", color="black", linewidth=1.8, alpha=0.95, zorder=10)

    for key in ordered_keys:
        df = available.get(key)
        if df is None:
            df = available_raw.get(key + "_test")
        if df is None:
            continue
        if "timestamp" in df.columns:
            try:
                df = df.copy()
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.sort_values("timestamp")
            except Exception:
                pass
        if "y_pred" not in df.columns:
            continue
        sub = df.iloc[:n_points]
        ax.plot(range(len(sub)), sub["y_pred"].values,
                label=MODEL_DISPLAY_NAMES.get(key, key),
                color=MODEL_COLORS.get(key), linewidth=1.2, alpha=0.85)

    ax.set_xlabel("Sample Index", fontsize=10)
    ax.set_ylabel("Power (normalized)", fontsize=10)
    ax.set_title(f"EXP-P05 预测曲线叠加 (h{horizon} {HORIZON_LABELS[horizon]})",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out = FIGURES_DIR / f"h{horizon}" / f"h{horizon}_prediction_overlay.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("保存 %s", out.name)
    return out


# ─── 3. 残差预测 vs 直接预测 ────────────────────────────────
def plot_residual_comparison(horizon: int, logger: logging.Logger) -> Path | None:
    path = METRICS_DIR / f"h{horizon}" / "residual_metrics.json"
    if not path.exists():
        logger.warning("无残差指标，跳过")
        return None
    metrics = load_json(path)
    if not metrics:
        return None

    residual_keys = list(metrics.keys())
    names = [MODEL_DISPLAY_NAMES.get(k.replace("_residual", ""), k) for k in residual_keys]
    rmse_residual = [metrics[k].get("RMSE", np.nan) for k in residual_keys]

    # 尝试获取直接预测指标（从 baseline）
    baseline = load_json(METRICS_DIR / f"h{horizon}" / "baseline_metrics.json")
    rmse_direct = []
    for key in residual_keys:
        base = key.replace("_residual", "")
        if baseline and base in baseline:
            rmse_direct.append(baseline[base].get("RMSE", np.nan))
        else:
            rmse_direct.append(np.nan)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(names))
    width = 0.35
    has_direct = any(not np.isnan(v) for v in rmse_direct)

    if has_direct:
        ax.bar(x - width / 2, rmse_direct, width, label="Direct", color="#4c78a8", alpha=0.85)
        ax.bar(x + width / 2, rmse_residual, width, label="Residual", color="#f58518", alpha=0.85)
        ymax = max(v for v in rmse_direct + rmse_residual if not np.isnan(v))
    else:
        ax.bar(names, rmse_residual, color="#54a24b", alpha=0.85)
        ymax = max(v for v in rmse_residual if not np.isnan(v))

    ax.set_title(f"EXP-P05 直接预测 vs 残差预测 (h{horizon} {HORIZON_LABELS[horizon]})",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("RMSE ↓", fontsize=10)
    if has_direct:
        ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    offset = ymax * 0.01
    for i, (d, r) in enumerate(zip(rmse_direct, rmse_residual)):
        if has_direct:
            if not np.isnan(d):
                ax.text(i - width / 2, d + offset, f"{d:.4f}", ha="center", va="bottom", fontsize=8)
            ax.text(i + width / 2, r + offset, f"{r:.4f}", ha="center", va="bottom", fontsize=8)
        else:
            ax.text(i, r + offset, f"{r:.4f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out = FIGURES_DIR / f"h{horizon}" / f"h{horizon}_residual_comparison.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("保存 %s", out.name)
    return out


# ─── 4. 推理效率对比 ────────────────────────────────────────
def plot_inference_benchmark(horizon: int, logger: logging.Logger) -> Path | None:
    path = BENCHMARK_DIR / f"h{horizon}" / "inference_benchmark.json"
    if not path.exists():
        logger.warning("无 benchmark 数据，跳过")
        return None
    bench = load_json(path)
    if not bench:
        return None

    items = []
    for key, m in bench.items():
        items.append({
            "Model": MODEL_DISPLAY_NAMES.get(key, key),
            "ms/sample": m.get("ms_per_sample", np.nan),
            "samples/s": m.get("samples_per_sec", np.nan),
            "params_m": m.get("params", 0) / 1e6,
        })
    if not items:
        return None

    df = pd.DataFrame(items).sort_values("ms/sample", na_position="last").reset_index(drop=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(f"EXP-P05 推理效率 (h{horizon} {HORIZON_LABELS[horizon]})",
                 fontsize=13, fontweight="bold")
    for ax, col, ylabel in zip(axes, ["ms/sample", "samples/s", "params_m"],
                               ["ms/sample ↓", "samples/s ↑", "Params (M)"]):
        vals = df[col].fillna(0).values
        bars = ax.bar(df["Model"], vals, color="#72b7b2", alpha=0.85)
        ax.set_title(ylabel, fontsize=11)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df["Model"], rotation=30, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, vals):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + max(vals) * 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = FIGURES_DIR / f"h{horizon}" / f"h{horizon}_inference_benchmark.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("保存 %s", out.name)
    return out


# ─── 5. 训练损失曲线（多模型合并） ───────────────────────
def plot_loss_overview(horizon: int, logger: logging.Logger) -> Path | None:
    hdir = METRICS_DIR / f"h{horizon}"
    history_files = sorted(hdir.glob("*_train_history.csv"))
    if not history_files:
        logger.warning("无训练历史，跳过")
        return None

    fig, axes = plt.subplots(1, len(history_files), figsize=(4 * len(history_files), 4))
    if len(history_files) == 1:
        axes = [axes]
    fig.suptitle(f"EXP-P05 训练损失曲线 (h{horizon} {HORIZON_LABELS[horizon]})",
                 fontsize=12, fontweight="bold")

    for ax, hist_path in zip(axes, history_files):
        stem = hist_path.name.replace("_train_history.csv", "")
        try:
            df = pd.read_csv(hist_path)
        except Exception:
            ax.set_title(stem)
            continue
        loss_col = "loss" if "loss" in df.columns else ("train_loss" if "train_loss" in df.columns else None)
        val_col = "val_loss" if "val_loss" in df.columns else None
        if loss_col:
            ax.plot(df[loss_col].values, label="Train Loss", color="#4c78a8")
        if val_col and val_col in df.columns:
            ax.plot(df[val_col].values, label="Val Loss", color="#e45756")
        display = MODEL_DISPLAY_NAMES.get(stem, stem)
        ax.set_title(display, fontsize=10)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("Loss", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out = FIGURES_DIR / f"h{horizon}" / f"h{horizon}_loss_overview.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("保存 %s", out.name)
    return out


# ─── 6b. 跨 Horizon 残差预测 RMSE 汇总（单图，3 柱/模型） ──
def plot_cross_horizon_residual(logger: logging.Logger) -> Path | None:
    horizons = [1, 4, 16]
    hlabels_short = ["h1", "h4", "h16"]
    horizon_colors = ["#4c78a8", "#f58518", "#54a24b"]  # Residual

    # 收集数据：RMSE_drop = residual RMSE
    all_residual = {}
    for h in horizons:
        res_path = METRICS_DIR / f"h{h}" / "residual_metrics.json"
        residual = load_json(res_path) if res_path.exists() else {}
        for key in residual:
            all_residual.setdefault(key, {})[h] = residual[key].get("RMSE", np.nan)

    if not all_residual:
        logger.warning("无残差模型数据，跳过")
        return None

    model_keys = sorted(all_residual.keys())
    model_names = [MODEL_DISPLAY_NAMES.get(k.replace("_residual", ""), k) for k in model_keys]
    n_models = len(model_keys)

    # 每个模型 3 根柱紧挨着，无 gap
    bw = 0.28
    offsets = [-bw, 0, bw]

    x_base = np.arange(n_models)
    fig, ax = plt.subplots(figsize=(5 + n_models * 1.3, 5.2))
    fig.suptitle("EXP-P05 残差预测 RMSE（跨 Horizon）",
                 fontsize=14, fontweight="bold", y=0.97)

    ymax_overall = 0.0
    for i, (h, hl, hc, off) in enumerate(zip(horizons, hlabels_short, horizon_colors, offsets)):
        vals = [all_residual.get(mk, {}).get(h, np.nan) for mk in model_keys]
        bars = ax.bar(x_base + off, vals, bw, label=f"{hl}", color=hc, alpha=0.85, edgecolor="white", lw=0.5)
        all_v = [v for v in vals if not np.isnan(v)]
        if all_v:
            ymax_overall = max(ymax_overall, max(all_v))
        for bar, val in zip(bars, vals):
            if np.isnan(val):
                continue
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ymax_overall * 0.012,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=7.5, rotation=30)

    ax.set_xticks(x_base)
    ax.set_xticklabels(model_names, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("RMSE ↓", fontsize=11)
    ax.set_ylim(0, ymax_overall * 1.25)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlim(x_base[0] - bw * 2, x_base[-1] + bw * 2)

    plt.tight_layout()
    out = FIGURES_DIR / "residual_comparison_all_horizons.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("保存 %s", out.name)
    return out


# ─── 7. 跨 Horizon 推理效率汇总（3 子图，分组柱状图） ──
def plot_cross_horizon_benchmark(logger: logging.Logger) -> Path | None:
    horizons = [1, 4, 16]
    hlabels = ["15min (h1)", "1h (h4)", "4h (h16)"]
    horizon_colors = ["#4c78a8", "#f58518", "#54a24b"]  # h1=蓝 h4=橙 h16=绿
    model_keys = ["lstm_residual", "bilstm_residual", "cnn_lstm_residual",
                  "cnn_bilstm_residual", "patchtst_residual"]

    # 收集数据: model_key -> {h -> {metric -> val}}
    all_data = {}
    for h in horizons:
        path = BENCHMARK_DIR / f"h{h}" / "inference_benchmark.json"
        if not path.exists():
            continue
        bench = load_json(path)
        for mk in model_keys:
            if mk not in all_data:
                all_data[mk] = {}
            if mk in bench:
                all_data[mk][h] = {
                    "ms/sample": bench[mk].get("ms_per_sample", np.nan),
                    "samples/s": bench[mk].get("samples_per_sec", np.nan),
                    "params_m": bench[mk].get("params", 0) / 1e6,
                }

    if not all_data:
        logger.warning("无 benchmark 数据，跳过")
        return None

    model_names = [MODEL_DISPLAY_NAMES.get(mk.replace("_residual", ""), mk) for mk in all_data]
    n_models = len(model_names)
    x = np.arange(n_models)
    bar_width = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("EXP-P05 跨 Horizon 推理效率对比", fontsize=14, fontweight="bold", y=1.01)

    configs = [
        ("ms/sample", "ms/sample ↓"),
        ("samples/s", "samples/s ↑"),
        ("params_m", "Params (M)"),
    ]

    for ax, (metric, ylabel) in zip(axes, configs):
        for i, (h, hl, hc) in enumerate(zip(horizons, hlabels, horizon_colors)):
            vals = []
            for mk in all_data:
                v = all_data[mk].get(h, {}).get(metric, np.nan)
                vals.append(v)
            offset = (i - 1) * bar_width
            bars = ax.bar(x + offset, vals, bar_width, label=hl, color=hc, alpha=0.85)
            # 标注数值
            ymax = max((v for v in vals if not np.isnan(v)), default=0.001)
            for bar, val in zip(bars, vals):
                if np.isnan(val):
                    continue
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + ymax * 0.015,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, rotation=0)

        ax.set_title(ylabel, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=20, ha="right", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = FIGURES_DIR / "inference_benchmark_all_horizons.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("保存 %s", out.name)
    return out


# ─── 7. 跨 Horizon 综合对比 ────────────────────────────────
def plot_cross_horizon_summary(logger: logging.Logger) -> tuple[Path | None, Path | None]:
    horizons = [1, 4, 16]
    hlabels = ["15min", "1h", "4h"]
    df_all = None
    for h in horizons:
        df = _build_metrics_df(h)
        if not df.empty:
            df["horizon"] = h
            df_all = pd.concat([df_all, df], ignore_index=True) if df_all is not None else df

    if df_all is None or df_all.empty:
        logger.warning("跨 horizon 数据不足，跳过综合图")
        return None, None

    # 图1: 4-panel 综合（RMSE 柱状 + 趋势线 + 热力图 + 最佳推荐表）
    residual_df = df_all[df_all["key"].str.endswith("_residual")].copy()
    if residual_df.empty:
        logger.warning("无残差模型数据，跳过综合图")
        return None, None

    model_keys = residual_df["key"].unique()
    display_names = [MODEL_DISPLAY_NAMES.get(k.replace("_residual", ""), k) for k in model_keys]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("EXP-P05 跨 Horizon 综合分析", fontsize=15, fontweight="bold", y=0.98)

    # Panel A: RMSE 柱状对比
    ax = axes[0, 0]
    x = np.arange(len(horizons))
    w = 0.7 / len(display_names)
    for i, (mk, dn) in enumerate(zip(model_keys, display_names)):
        vals = []
        for h in horizons:
            sub = residual_df[(residual_df["key"] == mk) & (residual_df["horizon"] == h)]
            vals.append(sub["RMSE"].values[0] if not sub.empty else np.nan)
        offset = x + i * w - 0.7 / 2 + w / 2
        ax.bar(offset, vals, w, label=dn, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"h{h} ({l})" for h, l in zip(horizons, hlabels)], fontsize=10)
    ax.set_ylabel("RMSE ↓", fontsize=10)
    ax.set_title("A. RMSE 对比", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    # Panel B: R2 柱状对比
    ax = axes[0, 1]
    for i, (mk, dn) in enumerate(zip(model_keys, display_names)):
        vals = []
        for h in horizons:
            sub = residual_df[(residual_df["key"] == mk) & (residual_df["horizon"] == h)]
            vals.append(sub["R2"].values[0] if not sub.empty else np.nan)
        offset = x + i * w - 0.7 / 2 + w / 2
        ax.bar(offset, vals, w, label=dn, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"h{h} ({l})" for h, l in zip(horizons, hlabels)], fontsize=10)
    ax.set_ylabel("R2 ↑", fontsize=10)
    ax.set_title("B. R2 对比", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(axis="y", alpha=0.3)

    # Panel C: RMSE 趋势线
    ax = axes[1, 0]
    for mk, dn in zip(model_keys, display_names):
        means, stds = [], []
        for h in horizons:
            sub = residual_df[(residual_df["key"] == mk) & (residual_df["horizon"] == h)]
            means.append(sub["RMSE"].values[0] if not sub.empty else np.nan)
            stds.append(np.nan)
        ax.plot(horizons, means, marker="o", markersize=6, label=dn, linewidth=2,
                 color=MODEL_COLORS.get(mk, None))
    ax.set_xscale("log")
    ax.set_xticks(horizons)
    ax.set_xticklabels(["1", "4", "16"])
    ax.set_xlabel("Horizon (steps)", fontsize=10)
    ax.set_ylabel("RMSE ↓", fontsize=10)
    ax.set_title("C. RMSE 随 Horizon 变化趋势", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.grid(alpha=0.3)

    # Panel D: RMSE 热力图
    ax = axes[1, 1]
    heat_data = []
    for mk in model_keys:
        row = []
        for h in horizons:
            sub = residual_df[(residual_df["key"] == mk) & (residual_df["horizon"] == h)]
            row.append(sub["RMSE"].values[0] if not sub.empty else np.nan)
        heat_data.append(row)
    heat_data = np.array(heat_data)
    im = ax.imshow(heat_data, cmap="RdYlGn_r", aspect="auto")
    plt.colorbar(im, ax=ax, label="RMSE")
    ax.set_xticks(range(len(horizons)))
    ax.set_xticklabels([f"h{h}" for h in horizons])
    ax.set_yticks(range(len(display_names)))
    ax.set_yticklabels(display_names)
    ax.set_title("D. RMSE 热力图", fontsize=11, fontweight="bold")
    for i in range(len(model_keys)):
        for j in range(len(horizons)):
            val = heat_data[i, j]
            color = "white" if not np.isnan(val) and val > heat_data[~np.isnan(heat_data)].mean() else "black"
            ax.text(j, i, f"{val:.4f}" if not np.isnan(val) else "-",
                    ha="center", va="center", color=color, fontsize=9, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_summary = FIGURES_DIR / "comparison_summary.png"
    fig.savefig(out_summary, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("保存 %s", out_summary.name)

    # 图2: 最佳模型推荐表
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.axis("off")
    ax2.set_title("EXP-P05 各 Horizon 最佳模型推荐", fontsize=14, fontweight="bold", pad=12)

    col_labels = ["Horizon", "最佳模型", "RMSE ↓", "MAE ↓", "R2 ↑"]
    table_data = []
    for h, hl in zip(horizons, hlabels):
        h_df = residual_df[residual_df["horizon"] == h].sort_values("RMSE")
        if h_df.empty:
            continue
        best = h_df.iloc[0]
        table_data.append([
            f"h{h} ({hl})",
            best["Model"],
            f"{best['RMSE']:.4f}",
            f"{best['MAE']:.4f}",
            f"{best['R2']:.4f}",
        ])

    table = ax2.table(cellText=table_data, colLabels=col_labels,
                      loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(len(table_data)):
        if i == 0:
            table[i + 1, 1].set_facecolor("#d4edda")

    out_table = FIGURES_DIR / "comparison_best_model.png"
    fig2.tight_layout()
    fig2.savefig(out_table, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    logger.info("保存 %s", out_table.name)

    return out_summary, out_table


# ─── 更新报告 ─────────────────────────────────────────────
def update_report(horizon: int, logger: logging.Logger) -> None:
    report_path = REPORTS_DIR / f"EXP-P05_h{horizon}_report.md"
    if not report_path.exists():
        logger.warning("报告不存在: %s", report_path)
        return

    hdir = FIGURES_DIR / f"h{horizon}"
    figures = []
    for name in [
        f"h{horizon}_metrics_comparison.png",
        f"h{horizon}_prediction_overlay.png",
        f"h{horizon}_residual_comparison.png",
        f"h{horizon}_inference_benchmark.png",
        f"h{horizon}_loss_overview.png",
    ]:
        p = hdir / name
        if p.exists():
            rel = p.relative_to(PROJECT_ROOT)
            figures.append(f"![{p.stem}]({rel.as_posix()})")

    text = report_path.read_text(encoding="utf-8")
    vis = "\n\n## 可视化\n\n" + "\n\n".join(figures) + "\n"
    if "## 可视化" in text:
        text = text.split("## 可视化")[0].rstrip() + vis
    else:
        text = text.rstrip() + vis
    report_path.write_text(text, encoding="utf-8")
    logger.info("已更新报告: %s", report_path.name)


def update_cross_report(logger: logging.Logger) -> None:
    """更新各 horizon 报告，追加跨 horizon 对比图"""
    cross_figures = []
    for name in ["comparison_summary.png", "comparison_best_model.png",
                 "residual_comparison_all_horizons.png",
                 "inference_benchmark_all_horizons.png"]:
        p = FIGURES_DIR / name
        if p.exists():
            rel = p.relative_to(PROJECT_ROOT)
            cross_figures.append(f"![{p.stem}]({rel.as_posix()})")

    for h in [1, 4, 16]:
        report_path = REPORTS_DIR / f"EXP-P05_h{h}_report.md"
        if not report_path.exists():
            continue
        text = report_path.read_text(encoding="utf-8")
        cross_section = "\n\n## 跨 Horizon 综合对比\n\n" + "\n\n".join(cross_figures) + "\n"
        if "## 跨 Horizon 综合对比" not in text:
            text = text.rstrip() + cross_section
            report_path.write_text(text, encoding="utf-8")
            logger.info("已追加跨 horizon 图到报告: %s", report_path.name)


# ─── 主流程 ───────────────────────────────────────────────
def generate_figures(horizon: int, logger: logging.Logger):
    logger.info("生成 horizon=%d 图表...", horizon)
    plot_metrics_comparison(horizon, logger)
    plot_overlay(horizon, logger)
    plot_residual_comparison(horizon, logger)
    plot_inference_benchmark(horizon, logger)
    plot_loss_overview(horizon, logger)
    update_report(horizon, logger)


def main():
    parser = argparse.ArgumentParser(description="EXP-P05 图表生成（合并版）")
    parser.add_argument("--horizon", type=int, choices=[1, 4, 16], required=True)
    parser.add_argument("--cross", action="store_true", help="生成跨 horizon 综合对比图")
    args = parser.parse_args()

    ensure_dirs(FIGURES_DIR / f"h{args.horizon}", REPORTS_DIR)
    logger = setup_logger("figures", f"EXP-P05_h{args.horizon}_figures.log")
    logger.info("EXP-P05 图表生成 horizon=%d", args.horizon)

    t0 = time.time()
    generate_figures(args.horizon, logger)
    logger.info("horizon=%d 完成，耗时 %.2fs", args.horizon, time.time() - t0)

    if args.cross:
        logger.info("生成跨 horizon 综合对比图...")
        t1 = time.time()
        plot_cross_horizon_summary(logger)
        plot_cross_horizon_residual(logger)
        plot_cross_horizon_benchmark(logger)
        update_cross_report(logger)
        logger.info("跨 horizon 图完成，耗时 %.2fs", time.time() - t1)


if __name__ == "__main__":
    main()
