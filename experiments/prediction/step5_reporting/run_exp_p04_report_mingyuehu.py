"""
明月湖数据集实验报告生成脚本
python -m experiments.prediction.step5_reporting.run_exp_p04_report_mingyuehu --horizon 1
生成详细实验报告（Markdown + 图表）。
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

# 优先使用中文字体（Windows），缺失时回退 DejaVu Sans
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.prediction.step2_hyperparameter_search.exp_p04_common import (
    FIGURES_DIR,
    METRICS_DIR,
    MODEL_DISPLAY_NAMES,
    PRED_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
    setup_logger,
)
from experiments.prediction.step5_reporting.exp_p04_step_audit import (
    record_step_failure,
    record_step_result,
)

# =============================================================================
# 明月湖配置
# =============================================================================
HORIZON_CONFIGS = {
    1: {"horizon": 1, "lookback": 16, "horizon_label": "H1 (15min)"},
    4: {"horizon": 4, "lookback": 48, "horizon_label": "H4 (1h)"},
    16: {"horizon": 16, "lookback": 96, "horizon_label": "H16 (4h)"},
}

MODEL_ORDER = ["cnn_bilstm"]


def load_mingyuehu_sample_dir(horizon: int, lookback: int = None) -> Path:
    """返回明月湖样本目录"""
    if lookback is None:
        lookback = HORIZON_CONFIGS[horizon]["lookback"]
    from experiments.prediction.step2_hyperparameter_search.exp_p04_common import SAMPLES_DIR
    return SAMPLES_DIR / f"mingyuehu_h{horizon}_lb{lookback}"


def load_mingyuehu_test_timestamps(horizon: int, lookback: int = None) -> pd.Series:
    """加载明月湖测试集时间戳"""
    hdir = load_mingyuehu_sample_dir(horizon, lookback)
    return pd.read_csv(hdir / "test_timestamps.csv", parse_dates=["timestamp"])["timestamp"]


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Convert DataFrame to markdown table string (no tabulate dependency)."""
    if df.empty:
        return "*无数据*"
    lines = []
    col_widths = {c: len(str(c)) for c in df.columns}
    for _, row in df.iterrows():
        for c in df.columns:
            col_widths[c] = max(col_widths[c], len(str(row[c])))
    header = "| " + " | ".join(str(c).ljust(col_widths[c]) for c in df.columns) + " |"
    sep = "| " + " | ".join("-" * col_widths[c] for c in df.columns) + " |"
    lines.append(header)
    lines.append(sep)
    for _, row in df.iterrows():
        line = "| " + " | ".join(str(row[c]).ljust(col_widths[c]) for c in df.columns) + " |"
        lines.append(line)
    return "\n".join(lines)


def _load_reproduce(horizon: int, model: str) -> dict | None:
    p = METRICS_DIR / f"mingyuehu_h{horizon}" / f"mingyuehu_{model}_reproduce.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def _load_test_metrics(horizon: int, model: str) -> dict | None:
    p = METRICS_DIR / f"mingyuehu_h{horizon}" / f"mingyuehu_{model}_test_metrics.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def _load_optuna(horizon: int, model: str) -> dict | None:
    p = METRICS_DIR / f"mingyuehu_h{horizon}" / f"mingyuehu_{model}_optuna.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def _rank_key(m: dict) -> tuple:
    """Ranking: RMSE (primary, lower better) > MAE (secondary) > R2 (tertiary, higher better)."""
    return (
        m["mean"].get("RMSE", float("inf")),
        m["mean"].get("MAE", float("inf")),
        -m["mean"].get("R2", 0),
    )


def _summary_table(metrics_list: list[dict], horizon_label: str) -> pd.DataFrame:
    rows = []
    for m in metrics_list:
        mean = m.get("mean", {})
        std = m.get("std", {})
        row = {
            "Model": MODEL_DISPLAY_NAMES.get(m["model"], m["model"]),
            "MAE": f"{mean.get('MAE', 0):.4f} ± {std.get('MAE', 0):.4f}",
            "RMSE": f"{mean.get('RMSE', 0):.4f} ± {std.get('RMSE', 0):.4f}",
            "MAPE(%)": f"{mean.get('MAPE', 0):.2f} ± {std.get('MAPE', 0):.2f}",
            "R²": f"{mean.get('R2', 0):.4f} ± {std.get('R2', 0):.4f}",
            "Time(s)": f"{mean.get('training_time_sec', 0):.1f}",
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Add rank column: sort by RMSE primary > MAE secondary > R2 tertiary
    sorted_metrics = sorted(metrics_list, key=_rank_key)
    rank_map = {MODEL_DISPLAY_NAMES.get(m["model"], m["model"]): i + 1
                for i, m in enumerate(sorted_metrics)}
    df.insert(1, "Rank", df["Model"].map(rank_map))
    df = df.sort_values("Rank")

    return df


def _build_metrics_list(horizon: int, all_models: list[str]) -> list[dict]:
    metrics_list = []
    for mname in all_models:
        rep = _load_reproduce(horizon, mname)
        if rep:
            metrics_list.append(rep)
    return metrics_list


def _load_horizon_pred_df(horizon: int, model_name: str) -> pd.DataFrame | None:
    """加载某 horizon 的测试集预测，多步预测仅保留第 1 步（与 H1 对齐）。"""
    for fname in (f"{model_name}_seed42_test.csv", f"{model_name}_test.csv"):
        path = PRED_DIR / f"mingyuehu_h{horizon}" / fname
        if path.exists():
            df = pd.read_csv(path, parse_dates=["timestamp"])
            if horizon > 1:
                df = df.iloc[0::horizon].reset_index(drop=True)
            return df
    return None


def _time_axis_locator(t0: pd.Timestamp, t1: pd.Timestamp) -> mdates.HourLocator:
    """按时间跨度选择合适的刻度间隔。"""
    span_hours = (t1 - t0).total_seconds() / 3600
    if span_hours <= 48:
        interval = 6
    elif span_hours <= 120:
        interval = 12
    else:
        interval = 24
    return mdates.HourLocator(interval=interval)


def _plot_predictions_all_horizons(
    model_name: str,
    n_points: int,
    logger,
) -> Path | None:
    """H1/H4/H16 同图展示：一行一个 horizon，共用同一时间窗口。"""
    horizon_specs = [
        (1, "H1 (15min)"),
        (4, "H4 (1h)"),
        (16, "H16 (4h)"),
    ]
    pred_colors = {1: "#1f77b4", 4: "#ff7f0e", 16: "#2ca02c"}

    loaded: dict[int, pd.DataFrame] = {}
    for h_int, _ in horizon_specs:
        df = _load_horizon_pred_df(h_int, model_name)
        if df is None or df.empty:
            logger.warning("缺少 mingyuehu h%d 预测文件，跳过跨 horizon 预测曲线", h_int)
            return None
        loaded[h_int] = df

    n = min(n_points, *(len(loaded[h]) for h in loaded))
    ref_ts = loaded[1]["timestamp"].iloc[:n]
    t0 = pd.Timestamp(ref_ts.iloc[0])
    t1 = pd.Timestamp(ref_ts.iloc[-1])
    t0_str = t0.strftime("%Y-%m-%d %H:%M")
    t1_str = t1.strftime("%Y-%m-%d %H:%M")
    duration_str = str(t1 - t0).split(".")[0]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for ax, (h_int, label) in zip(axes, horizon_specs):
        sub = loaded[h_int].iloc[:n].copy()
        sub["timestamp"] = pd.to_datetime(ref_ts.values)

        ax.plot(
            sub["timestamp"], sub["y_true"],
            color="black", linewidth=1.5, linestyle="-", label="Actual",
        )
        ax.plot(
            sub["timestamp"], sub["y_pred"],
            color=pred_colors[h_int], linewidth=1.2, linestyle="--", label="Predicted",
        )
        ax.set_ylabel("power_pu")
        ax.set_title(label)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(t0, t1)

    locator = _time_axis_locator(t0, t1)
    formatter = mdates.DateFormatter("%m-%d %H:%M")
    for ax in axes:
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.tick_params(axis="x", rotation=30)

    # 均匀刻度，强制包含起止时间
    n_ticks = min(9, n)
    tick_idx = np.unique(np.round(np.linspace(0, n - 1, n_ticks)).astype(int))
    tick_times = pd.to_datetime(ref_ts.iloc[tick_idx].values)
    tick_labels = [pd.Timestamp(t).strftime("%m-%d\n%H:%M") for t in tick_times]
    for ax in axes:
        ax.set_xticks(tick_times)
        ax.set_xticklabels(tick_labels)

    axes[-1].set_xlabel(
        f"选用时间范围：{t0_str}  →  {t1_str}  "
        f"（时长 {duration_str}，共 {n} 个 15min 采样点）",
        fontsize=11,
    )
    fig.suptitle("明月湖 Test Set Predictions — H1 / H4 / H16", y=1.01, fontsize=13)
    plt.tight_layout()

    out = FIGURES_DIR / "mingyuehu_predictions_h1_h4_h16_combined.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("已保存跨 horizon 预测曲线: %s", out.name)
    return out


def _plot_comparison_horizons(logger):
    """绘制三个 horizon 的 MAE/RMSE/R² 对比柱状图。"""
    horizons = [(1, "15min"), (4, "1h"), (16, "4h")]
    all_models = ["cnn_bilstm"]

    mae_data, rmse_data, r2_data = {}, {}, {}
    for h, hlabel in horizons:
        mae_data[hlabel], rmse_data[hlabel], r2_data[hlabel] = {}, {}, {}
        for mname in all_models:
            rep = _load_reproduce(h, mname)
            if rep:
                mae_data[hlabel][MODEL_DISPLAY_NAMES.get(mname, mname)] = rep["mean"].get("MAE", 0)
                rmse_data[hlabel][MODEL_DISPLAY_NAMES.get(mname, mname)] = rep["mean"].get("RMSE", 0)
                r2_data[hlabel][MODEL_DISPLAY_NAMES.get(mname, mname)] = rep["mean"].get("R2", 0)

    display_names = []
    for m in all_models:
        dn = MODEL_DISPLAY_NAMES.get(m, m)
        if any(dn in mae_data[h] for h in ["15min", "1h", "4h"]):
            display_names.append(dn)

    x = np.arange(len(display_names))
    width = 0.25
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, data, metric_name in zip(
        axes, [mae_data, rmse_data, r2_data], ["MAE", "RMSE", "R²"]
    ):
        for i, hlabel in enumerate(["15min", "1h", "4h"]):
            vals = [data[hlabel].get(dn, 0) for dn in display_names]
            ax.bar(x + i * width - width, vals, width, label=hlabel)
        ax.set_xticks(x)
        ax.set_xticklabels(display_names, rotation=25, ha="right")
        ax.set_title(metric_name)
        ax.legend()
    plt.tight_layout()
    out = FIGURES_DIR / "mingyuehu_comparison_all_horizons.png"
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info("已保存跨 horizon 对比图: %s", out.name)


def _plot_loss_curve(history_path, title: str, out_path) -> None:
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


def _plot_pred_curve(df: pd.DataFrame, title: str, out_path, n_points: int = 500) -> None:
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


def _plot_overlay(preds: dict, n_points: int, out_path, horizon: str) -> None:
    first_key = next(iter(preds))
    first = preds[first_key]
    sub_ts = first["timestamp"].iloc[:n_points]
    y_true = first["y_true"].iloc[:n_points]

    plt.figure(figsize=(14, 5))
    plt.plot(sub_ts, y_true, label="actual", color="black", linewidth=1.5)
    colors = ["C0", "C1", "C2", "C3", "C4", "C5"]
    for (key, label), color in zip(
        [(k, MODEL_DISPLAY_NAMES.get(k, k)) for k in MODEL_ORDER if k in preds], colors
    ):
        sub = preds[key].iloc[:n_points]
        plt.plot(sub_ts, sub["y_pred"], label=label, alpha=0.8, color=color)
    plt.xlabel("timestamp")
    plt.ylabel("power_pu")
    plt.title(f"明月湖 Test set predictions — all models ({horizon})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_training_time_comparison(metrics_df: pd.DataFrame, out_path) -> None:
    sub = metrics_df.dropna(subset=["training_time_sec"]).copy()
    if sub.empty:
        return
    sub = sub.sort_values("training_time_sec")
    plt.figure(figsize=(8, 4))
    bars = plt.barh(sub["display_name"], sub["training_time_sec"], color="steelblue")
    plt.xlabel("Training time (seconds)")
    plt.title("明月湖 Training Time Comparison")
    for bar, val in zip(bars, sub["training_time_sec"]):
        plt.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}s",
            va="center",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _gen_markdown_report(horizon: int, horizon_label: str, all_models: list[str],
                         logger) -> str:
    """生成 Markdown 报告全文。"""
    lookback = HORIZON_CONFIGS[horizon]["lookback"]
    hdir = load_mingyuehu_sample_dir(horizon, lookback)

    if not hdir.exists():
        logger.warning("明月湖样本目录不存在: %s", hdir)
        return ""

    meta = json.loads((hdir / "meta.json").read_text(encoding="utf-8"))

    metrics_list = _build_metrics_list(horizon, all_models)
    df_summary = _summary_table(metrics_list, horizon_label)

    # Optuna 参数表
    optuna_rows = []
    for mname in all_models:
        opt = _load_optuna(horizon, mname)
        if opt:
            params = opt.get("best_params", {})
            row = {
                "Model": MODEL_DISPLAY_NAMES.get(mname, mname),
                "Best Params": ", ".join(f"{k}={v}" for k, v in params.items()) if params else "-",
                "Val Loss": f"{opt.get('best_value', 0):.6f}",
            }
            optuna_rows.append(row)

    lines = []
    lines.append(f"# 明月湖数据集 EXP-P04 Optuna 调参详细实验汇报\n")
    lines.append(f"**数据集: 明月湖光伏电站 (5 MW)**\n")
    lines.append(f"**Horizon: {horizon_label} ({horizon} 步)**\n")
    lines.append(f"**生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}**\n\n")
    lines.append(f"---\n\n")

    # 1. 实验配置
    lines.append("## 1. 实验配置\n\n")
    lines.append("| 配置项 | 值 |\n|---|---|\n")
    lines.append(f"| 数据集 | 明月湖光伏电站 |\n")
    lines.append(f"| 电站容量 | {meta.get('capacity_kw', CAPACITY_KW := 5000):.0f} kW |\n")
    lines.append(f"| Lookback | {meta['lookback']} |\n")
    lines.append(f"| Horizon | {horizon} |\n")
    lines.append(f"| 特征数量 | {meta['n_features']} |\n")
    lines.append(f"| 训练样本数 | {meta['n_train']:,} |\n")
    lines.append(f"| 验证样本数 | {meta['n_val']:,} |\n")
    lines.append(f"| 测试样本数 | {meta['n_test']:,} |\n")
    lines.append(f"| 数据来源 | {meta.get('source_csv', 'N/A')} |\n")
    lines.append(f"| 预测模式 | {meta.get('prediction_mode', 'residual')} |\n")
    lines.append("\n")

    # 2. 特征列详情
    feat_cols = meta.get("feature_cols", [])
    lines.append("## 2. 特征工程\n\n")
    groups = {
        "气象/辐照": [c for c in feat_cols if any(x in c for x in ["irradiance", "temperature", "atmosphere", "humidity", "ghi", "dni", "dhi", "flag"])],
        "时间周期": [c for c in feat_cols if "sin_" in c or "cos_" in c],
        "功率 lag": [c for c in feat_cols if "_lag_" in c],
        "多尺度 ramp": [c for c in feat_cols if "_ramp_" in c],
        "滚动统计": [c for c in feat_cols if "_roll_" in c],
    }
    groups["其他"] = [c for c in feat_cols if c not in sum([groups[k] for k in groups], [])]
    for group, cols in groups.items():
        if cols:
            lines.append(f"**{group}**：`{'`, `'.join(cols)}`\n\n")

    # 3. 最优参数
    lines.append("## 3. Optuna 最优参数\n\n")
    if optuna_rows:
        lines.append(_df_to_markdown(pd.DataFrame(optuna_rows)))
        lines.append("\n")
    else:
        lines.append("*无 Optuna 结果*\n\n")

    # 4. 多 Seed 结果汇总表
    lines.append("## 4. 多 Seed 复现结果（Mean ± Std）\n\n")
    if not df_summary.empty:
        lines.append(_df_to_markdown(df_summary))
        lines.append("\n\n")
    else:
        lines.append("*无复现结果*\n\n")

    # 5. 关键发现
    lines.append("## 5. 关键发现\n\n")
    if metrics_list:
        sorted_metrics = sorted(metrics_list, key=_rank_key)
        best = sorted_metrics[0]
        worst = sorted_metrics[-1]
        lines.append(f"- **最优模型**: {MODEL_DISPLAY_NAMES.get(best['model'], best['model'])} "
                    f"(RMSE={best['mean'].get('RMSE', 0):.4f}, "
                    f"MAE={best['mean'].get('MAE', 0):.4f} ± {best['std'].get('MAE', 0):.4f}, "
                    f"R²={best['mean'].get('R2', 0):.4f})\n")
        lines.append(f"- **最差模型**: {MODEL_DISPLAY_NAMES.get(worst['model'], worst['model'])} "
                    f"(RMSE={worst['mean'].get('RMSE', 0):.4f})\n")
        lines.append("\n")
    else:
        lines.append("*等待实验完成*\n\n")

    # 6. 图表
    lines.append("## 6. 可视化\n\n")
    lines.append("### 6.1 跨 Horizon 预测曲线（H1/H4/H16）\n\n")
    fig_cross = FIGURES_DIR / "mingyuehu_predictions_h1_h4_h16_combined.png"
    if fig_cross.exists():
        rel_cross = fig_cross.relative_to(PROJECT_ROOT)
        lines.append(f"![跨Horizon预测曲线]({rel_cross})\n\n")
        lines.append("*真实值：黑色实线；预测值：彩色虚线（H1 蓝 / H4 橙 / H16 绿）。三行子图共用同一时间窗口，x 轴标注起止时间与采样点数。*\n\n")

    lines.append("### 6.2 指标对比\n\n")
    fig_h = FIGURES_DIR / f"mingyuehu_h{horizon}"
    if fig_h.exists():
        for fpath in fig_h.glob("*.png"):
            rel = fpath.relative_to(PROJECT_ROOT)
            lines.append(f"![{fpath.stem}]({rel})\n\n")

    lines.append("---\n\n")
    lines.append(f"*报告由 run_exp_p04_report_mingyuehu.py 自动生成*\n")

    return "\n".join(lines)


def generate_figures(horizon: int, horizon_label: str, all_models: list[str], logger):
    """生成所有图表。"""
    lookback = HORIZON_CONFIGS[horizon]["lookback"]
    hdir = load_mingyuehu_sample_dir(horizon, lookback)
    fig_h = FIGURES_DIR / f"mingyuehu_h{horizon}"
    fig_h.mkdir(parents=True, exist_ok=True)

    if not (hdir / "meta.json").exists():
        logger.warning("样本 meta.json 不存在，跳过绘图")
        return

    logger.info("生成图表...")
    ts_path = hdir / "test_timestamps.csv"
    if not ts_path.exists():
        logger.warning("test_timestamps.csv 不存在，跳过时序图")
        ts = None
    else:
        ts = pd.read_csv(ts_path, parse_dates=["timestamp"])["timestamp"]

    # 加载复现结果中 seed=42 的预测
    preds = {}
    for mname in all_models:
        pred_path = PRED_DIR / f"mingyuehu_h{horizon}" / f"{mname}_seed42_test.csv"
        if pred_path.exists():
            df = pd.read_csv(pred_path)
            if ts is not None and "timestamp" not in df.columns:
                df.insert(0, "timestamp", ts.values[: len(df)])
            preds[mname] = df

    # 1. 指标柱状图
    metrics_list = _build_metrics_list(horizon, all_models)
    if metrics_list:
        display_names = [MODEL_DISPLAY_NAMES.get(m["model"], m["model"]) for m in metrics_list]
        mae_vals = [m["mean"].get("MAE", 0) for m in metrics_list]
        rmse_vals = [m["mean"].get("RMSE", 0) for m in metrics_list]
        r2_vals = [m["mean"].get("R2", 0) for m in metrics_list]

        df_metrics = pd.DataFrame({
            "display_name": display_names,
            "MAE": mae_vals,
            "RMSE": rmse_vals,
            "R2": r2_vals,
        })
        # 指标对比合并图（MAE / RMSE / R² 子图）
        out_combined = fig_h / f"mingyuehu_h{horizon}_metrics_comparison.png"
        fig_cmp, axes = plt.subplots(1, 3, figsize=(16, 4))
        for ax, (col, ylabel) in zip(axes, [("MAE", "MAE"), ("RMSE", "RMSE"), ("R2", "R²")]):
            vals = [df_metrics.loc[df_metrics["display_name"] == dn, col].values[0] for dn in display_names]
            bars = ax.bar(display_names, vals)
            ax.set_title(ylabel)
            ax.set_xticklabels(display_names, rotation=25, ha="right")
            ax.set_ylabel(ylabel)
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.01,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        fig_cmp.suptitle(f"明月湖 Metrics Comparison — {horizon_label}", y=1.02, fontsize=13)
        plt.tight_layout()
        fig_cmp.savefig(out_combined, dpi=150)
        plt.close(fig_cmp)
        logger.info("  保存 %s", out_combined.name)

        # 训练时间
        df_time = pd.DataFrame({
            "display_name": display_names,
            "training_time_sec": [m["mean"].get("training_time_sec", 0) for m in metrics_list],
        })
        out_time = fig_h / "training_time.png"
        _plot_training_time_comparison(df_time, out_time)
        logger.info("  保存 %s", out_time.name)

    # 2. 多模型时序叠加图
    if preds:
        out_overlay = fig_h / "predictions_overlay.png"
        _plot_overlay(preds, n_points=500, out_path=out_overlay, horizon=horizon_label)
        logger.info("  保存 %s", out_overlay.name)

        # 3. 单模型预测对比
        for mname, df in preds.items():
            out_single = fig_h / f"pred_{mname}.png"
            title = f"{MODEL_DISPLAY_NAMES.get(mname, mname)} — {horizon_label}"
            _plot_pred_curve(df, title, out_single, n_points=500)
            logger.info("  保存 %s", out_single.name)

    # 4. 训练 loss 曲线
    for mname in all_models:
        hist_path = METRICS_DIR / f"mingyuehu_h{horizon}" / f"mingyuehu_{mname}_seed42_train_history.csv"
        if hist_path.exists():
            out_loss = fig_h / f"loss_{mname}.png"
            _plot_loss_curve(hist_path,
                            f"{MODEL_DISPLAY_NAMES.get(mname, mname)} Loss — {horizon_label}",
                            out_loss)
            logger.info("  保存 %s", out_loss.name)

    logger.info("图表生成完成")


def run_report(horizon: int, logger, n_points: int = 500):
    horizon_label = HORIZON_CONFIGS[horizon]["horizon_label"]
    all_models = ["cnn_bilstm"]

    logger.info("=" * 60)
    logger.info("生成报告  horizon=%s (%s)", horizon, horizon_label)

    # 生成图表
    generate_figures(horizon, horizon_label, all_models, logger)

    # 跨 horizon 预测曲线 + 指标对比（h1/h4/h16 预测文件齐全时生成）
    _plot_predictions_all_horizons("cnn_bilstm", n_points, logger)
    _plot_comparison_horizons(logger)

    # 生成 Markdown
    md_text = _gen_markdown_report(horizon, horizon_label, all_models, logger)
    if md_text:
        report_path = REPORTS_DIR / f"EXP-P04_mingyuehu_h{horizon}_详细实验汇报.md"
        report_path.write_text(md_text, encoding="utf-8")
        logger.info("报告已保存: %s", report_path.relative_to(PROJECT_ROOT))
    else:
        report_path = None
        logger.warning("报告内容为空，跳过保存")

    logger.info("报告生成完成！")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="明月湖 EXP-P04 报告生成")
    parser.add_argument("--horizon", type=int, choices=[1, 4, 16], required=True)
    parser.add_argument("--n-points", type=int, default=500,
                        help="跨 horizon 预测曲线的时间窗口长度（样本点数）")
    args = parser.parse_args()

    t0 = time.time()
    horizon = args.horizon

    log_file = f"EXP-P04_mingyuehu_h{horizon}_report.log"
    logger = setup_logger("report_mingyuehu", log_file)
    logger.info("=" * 60)
    logger.info("明月湖 EXP-P04 报告生成  horizon=%d", horizon)

    report_path = run_report(horizon, logger, n_points=args.n_points)
    elapsed = time.time() - t0

    combined_fig = FIGURES_DIR / "mingyuehu_predictions_h1_h4_h16_combined.png"
    artifacts = []
    if report_path:
        artifacts.append(str(report_path.relative_to(PROJECT_ROOT)))
    if combined_fig.exists():
        artifacts.append(str(combined_fig.relative_to(PROJECT_ROOT)))

    record_step_result(
        horizon, "report_mingyuehu", "success", log_file,
        summary={
            "dataset": "mingyuehu",
            "report_file": str(report_path.relative_to(PROJECT_ROOT)) if report_path else None,
            "n_points": args.n_points,
            "elapsed_sec": round(elapsed, 1),
        },
        duration_sec=elapsed,
        artifacts=artifacts,
    )
    return horizon, log_file


if __name__ == "__main__":
    t0 = time.time()
    try:
        main()
    except Exception as e:
        record_step_failure("report_mingyuehu", t0, e)
        raise
