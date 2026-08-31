"""
python -m experiments.prediction.step4_optuna_hybrid.run_exp_p04_report --horizon 1
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

from experiments.prediction.step4_optuna_hybrid.exp_p04_common import (
    FIGURES_DIR,
    METRICS_DIR,
    MODEL_DISPLAY_NAMES,
    MODEL_ORDER,
    PRED_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
    load_config,
    load_sample_dir,
    plot_loss_curve,
    plot_metrics_bar,
    plot_overlay,
    plot_pred_curve,
    plot_training_time_comparison,
    setup_logger,
)
from experiments.prediction.step4_optuna_hybrid.exp_p04_step_audit import (
    record_step_failure,
    record_step_result,
)


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


def _load_reproduce(horizon: str, model: str) -> dict | None:
    p = METRICS_DIR / horizon / f"{model}_reproduce.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def _load_test_metrics(horizon: str, model: str) -> dict | None:
    p = METRICS_DIR / horizon / f"{model}_test_metrics.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def _load_optuna(horizon: str, model: str) -> dict | None:
    p = METRICS_DIR / horizon / f"{model}_optuna.json"
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


def _build_metrics_list(horizon: str, all_models: list[str]) -> list[dict]:
    metrics_list = []
    for mname in all_models:
        rep = _load_reproduce(horizon, mname)
        if rep:
            metrics_list.append(rep)
    return metrics_list


def _load_horizon_pred_df(horizon: int, model_name: str) -> pd.DataFrame | None:
    """加载某 horizon 的测试集预测，多步预测仅保留第 1 步（与 H1 对齐）。"""
    hs = f"h{horizon}"
    for fname in (f"{model_name}_seed42_test.csv", f"{model_name}_test.csv"):
        path = PRED_DIR / hs / fname
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
            logger.warning("缺少 h%d 预测文件，跳过跨 horizon 预测曲线", h_int)
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
    fig.suptitle("Test Set Predictions — H1 / H4 / H16", y=1.01, fontsize=13)
    plt.tight_layout()

    out = FIGURES_DIR / "predictions_h1_h4_h16_combined.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("已保存跨 horizon 预测曲线: %s", out.name)
    return out


def _plot_comparison_horizons(base_cfg: dict, logger):
    """绘制三个 horizon 的 MAE/RMSE/R² 对比柱状图。"""
    horizons = [(1, "15min"), (4, "1h"), (16, "4h")]
    all_models = ["cnn_bilstm"]

    mae_data, rmse_data, r2_data = {}, {}, {}
    for h, hlabel in horizons:
        hs = f"h{h}"
        mae_data[hlabel], rmse_data[hlabel], r2_data[hlabel] = {}, {}, {}
        for mname in all_models:
            rep = _load_reproduce(hs, mname)
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
    out = FIGURES_DIR / "comparison_all_horizons.png"
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info("已保存跨 horizon 对比图: %s", out.name)


def _gen_markdown_report(horizon: int, horizon_label: str, all_models: list[str],
                         logger) -> str:
    """生成 Markdown 报告全文。"""
    hs = f"h{horizon}"
    horizon_cfg = load_config(f"exp_p04_h{horizon}.json")

    metrics_list = _build_metrics_list(hs, all_models)
    df_summary = _summary_table(metrics_list, horizon_label)

    # 基本信息
    hdir = load_sample_dir(horizon)
    meta = json.loads((hdir / "meta.json").read_text(encoding="utf-8"))
    base_cfg = load_config("exp_p04_base.json")

    # Optuna 参数表
    optuna_rows = []
    for mname in all_models:
        opt = _load_optuna(hs, mname)
        if opt:
            params = opt.get("best_params", {})
            row = {
                "Model": MODEL_DISPLAY_NAMES.get(mname, mname),
                "Best Params": ", ".join(f"{k}={v}" for k, v in params.items()) if params else "-",
                "Val Loss": f"{opt.get('best_value', 0):.6f}",
            }
            optuna_rows.append(row)

    lines = []
    lines.append(f"# EXP-P04 Optuna 调参详细实验汇报\n")
    lines.append(f"**Horizon: {horizon_label} ({horizon} 步)**\n")
    lines.append(f"**生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}**\n\n")
    lines.append(f"---\n\n")

    # 1. 实验配置
    lines.append("## 1. 实验配置\n\n")
    lines.append("| 配置项 | 值 |\n|---|---|\n")
    lines.append(f"| Lookback | {meta['lookback']} |\n")
    lines.append(f"| Horizon | {horizon} |\n")
    lines.append(f"| 特征数量 | {meta['n_features']} |\n")
    lines.append(f"| 训练样本数 | {meta['n_train']:,} |\n")
    lines.append(f"| 验证样本数 | {meta['n_val']:,} |\n")
    lines.append(f"| 测试样本数 | {meta['n_test']:,} |\n")
    lines.append(f"| Rolling Folds | {base_cfg['n_rolling_folds']} |\n")
    lines.append(f"| Optuna Trials/模型 | {base_cfg['optuna_n_trials_per_model']} |\n")
    lines.append(f"| 最终 Max Epochs | {base_cfg['final_max_epochs']} |\n")
    lines.append(f"| 最终 Patience | {base_cfg['final_patience']} |\n")
    lines.append(f"| Reproduce Seeds | {base_cfg['reproduce_seeds']} |\n")
    lines.append(f"| 特征列 | `{meta['feature_cols']}` |\n")
    lines.append("\n")

    # 2. 特征列详情
    lines.append("## 2. 特征工程\n\n")
    lines.append("使用特征列表：\n\n")
    feat_cols = meta.get("feature_cols", [])
    groups = {
        "气象/辐照": [c for c in feat_cols if "irradiance" in c or "temperature" in c or "atmosphere" in c or "humidity" in c or "flag" in c],
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
    fig_cross = FIGURES_DIR / "predictions_h1_h4_h16_combined.png"
    if fig_cross.exists():
        rel_cross = fig_cross.relative_to(PROJECT_ROOT)
        lines.append(f"![跨Horizon预测曲线]({rel_cross})\n\n")
        lines.append("*真实值：黑色实线；预测值：彩色虚线（H1 蓝 / H4 橙 / H16 绿）。三行子图共用同一时间窗口，x 轴标注起止时间与采样点数。*\n\n")

    lines.append("### 6.2 指标对比\n\n")
    fig_combined = FIGURES_DIR / hs / f"h{horizon}_metrics_comparison.png"
    fig_time = FIGURES_DIR / hs / "training_time.png"
    fig_overlay = FIGURES_DIR / hs / "predictions_overlay.png"
    for fpath in [fig_combined, fig_time, fig_overlay]:
        if fpath.exists():
            rel = fpath.relative_to(PROJECT_ROOT)
            lines.append(f"![{fpath.stem}]({rel})\n\n")

    lines.append("---\n\n")
    lines.append("*报告由 run_exp_p04_report.py 自动生成*\n")

    return "\n".join(lines)


def generate_figures(horizon: int, horizon_label: str, all_models: list[str], logger):
    """生成所有图表。"""
    hs = f"h{horizon}"
    hdir = load_sample_dir(horizon)
    fig_h = FIGURES_DIR / hs
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
        pred_path = PRED_DIR / hs / f"{mname}_seed42_test.csv"
        if pred_path.exists():
            df = pd.read_csv(pred_path)
            if ts is not None and "timestamp" not in df.columns:
                df.insert(0, "timestamp", ts.values[: len(df)])
            preds[mname] = df

    # 1. 指标柱状图
    metrics_list = _build_metrics_list(hs, all_models)
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
        out_combined = fig_h / f"h{horizon}_metrics_comparison.png"
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
        fig_cmp.suptitle(f"Metrics Comparison — {horizon_label}", y=1.02, fontsize=13)
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
        plot_training_time_comparison(df_time, out_time)
        logger.info("  保存 %s", out_time.name)

    # 2. 多模型时序叠加图
    if preds:
        out_overlay = fig_h / "predictions_overlay.png"
        plot_overlay(preds, n_points=500, out_path=out_overlay, horizon=horizon_label)
        logger.info("  保存 %s", out_overlay.name)

        # 3. 单模型预测对比
        for mname, df in preds.items():
            out_single = fig_h / f"pred_{mname}.png"
            title = f"{MODEL_DISPLAY_NAMES.get(mname, mname)} — {horizon_label}"
            plot_pred_curve(df, title, out_single, n_points=500)
            logger.info("  保存 %s", out_single.name)

    # 4. 训练 loss 曲线
    for mname in all_models:
        hist_path = METRICS_DIR / hs / f"{mname}_final_train_history.csv"
        if hist_path.exists():
            out_loss = fig_h / f"loss_{mname}.png"
            plot_loss_curve(hist_path,
                            f"{MODEL_DISPLAY_NAMES.get(mname, mname)} Loss — {horizon_label}",
                            out_loss)
            logger.info("  保存 %s", out_loss.name)

    logger.info("图表生成完成")


def run_report(horizon: int, logger, n_points: int = 500):
    horizon_cfg = load_config(f"exp_p04_h{horizon}.json")
    horizon_label = horizon_cfg["horizon_label"]
    hs = f"h{horizon}"

    all_models = horizon_cfg["baseline_models"] + list(horizon_cfg["model_search_space"].keys())

    logger.info("=" * 60)
    logger.info("生成报告  horizon=%s (%s)", horizon, horizon_label)

    # 生成图表
    generate_figures(horizon, horizon_label, all_models, logger)

    # 跨 horizon 预测曲线 + 指标对比（h1/h4/h16 预测文件齐全时生成）
    _plot_predictions_all_horizons("cnn_bilstm", n_points, logger)
    _plot_comparison_horizons(load_config("exp_p04_base.json"), logger)

    # 生成 Markdown
    md_text = _gen_markdown_report(horizon, horizon_label, all_models, logger)
    report_path = REPORTS_DIR / f"EXP-P04_h{horizon}_详细实验汇报.md"
    report_path.write_text(md_text, encoding="utf-8")
    logger.info("报告已保存: %s", report_path.relative_to(PROJECT_ROOT))
    logger.info("报告生成完成！")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="EXP-P04 报告生成")
    parser.add_argument("--horizon", type=int, choices=[1, 4, 16], required=True)
    parser.add_argument("--n-points", type=int, default=500,
                        help="跨 horizon 预测曲线的时间窗口长度（样本点数）")
    args = parser.parse_args()

    t0 = time.time()
    horizon = args.horizon
    horizon_cfg = load_config(f"exp_p04_h{horizon}.json")
    log_file = horizon_cfg["log_file"].replace(".log", "_report.log")
    logger = setup_logger("report", log_file)
    logger.info("=" * 60)
    logger.info("EXP-P04 报告生成  horizon=%d", horizon)

    report_path = run_report(horizon, logger, n_points=args.n_points)
    elapsed = time.time() - t0
    hs = f"h{horizon}"
    combined_fig = FIGURES_DIR / hs / "predictions_h1_h4_h16_combined.png"
    artifacts = [
        str(report_path.relative_to(PROJECT_ROOT)),
    ]
    if combined_fig.exists():
        artifacts.append(str(combined_fig.relative_to(PROJECT_ROOT)))

    record_step_result(
        horizon, "report", "success", log_file,
        summary={
            "report_file": str(report_path.relative_to(PROJECT_ROOT)),
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
        record_step_failure("report", t0, e)
        raise
