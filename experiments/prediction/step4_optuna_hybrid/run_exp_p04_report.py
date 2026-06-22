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
import matplotlib.pyplot as plt
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
    REPORTS_DIR,
    SAMPLES_DIR,
    load_config,
    plot_loss_curve,
    plot_metrics_bar,
    plot_overlay,
    plot_pred_curve,
    plot_training_time_comparison,
    setup_logger,
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


def _load_old_afsa() -> dict | None:
    csv_path = PROJECT_ROOT / "data/prediction/step3_hybrid_models/metrics/afsa_patchtst_metrics.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        row = df.iloc[0]
        return {
            "MAE": row["MAE"],
            "RMSE": row["RMSE"],
            "MAPE": row["MAPE"],
            "R2": row["R2"],
        }
    return None


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
    return df


def _build_metrics_list(horizon: str, all_models: list[str]) -> list[dict]:
    metrics_list = []
    for mname in all_models:
        rep = _load_reproduce(horizon, mname)
        if rep:
            metrics_list.append(rep)
    return metrics_list


def _plot_comparison_horizons(base_cfg: dict, logger):
    """绘制三个 horizon 的 MAE/RMSE/R² 对比柱状图。"""
    horizons = [(1, "15min"), (4, "1h"), (16, "4h")]
    all_models = ["lstm", "bilstm", "cnn_lstm", "cnn_bilstm", "minipatchtst", "afsa_patchtst"]

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
    old_afsa = _load_old_afsa()

    # 基本信息
    hdir = SAMPLES_DIR / hs
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

    # 5. 旧 AFSA-PatchTST 对比
    lines.append("## 5. 与旧 AFSA-PatchTST 对比\n\n")
    if old_afsa:
        lines.append("| 指标 | 旧 AFSA-PatchTST |\n|---|---|\n")
        lines.append(f"| MAE | {old_afsa.get('MAE', 'N/A'):.4f} |\n")
        lines.append(f"| RMSE | {old_afsa.get('RMSE', 'N/A'):.4f} |\n")
        lines.append(f"| MAPE(%) | {old_afsa.get('MAPE', 'N/A'):.2f} |\n")
        lines.append(f"| R² | {old_afsa.get('R2', 'N/A'):.4f} |\n")
        lines.append(f"\n*注：旧 AFSA-PatchTST 为 horizon=1 的结果。*\n\n")
    else:
        lines.append("*旧模型结果未找到*\n\n")

    # 6. 关键发现
    lines.append("## 6. 关键发现\n\n")
    if metrics_list:
        best = min(metrics_list, key=lambda x: x["mean"].get("MAE", float("inf")))
        lines.append(f"- **最优模型**: {MODEL_DISPLAY_NAMES.get(best['model'], best['model'])} "
                     f"(MAE={best['mean'].get('MAE', 0):.4f} ± {best['std'].get('MAE', 0):.4f})\n")
        worst = max(metrics_list, key=lambda x: x["mean"].get("MAE", 0))
        lines.append(f"- **最差模型**: {MODEL_DISPLAY_NAMES.get(worst['model'], worst['model'])} "
                     f"(MAE={worst['mean'].get('MAE', 0):.4f})\n")

        if old_afsa:
            old_mae = old_afsa.get("metrics", {}).get("MAE", float("inf"))
            best_mae = best["mean"].get("MAE", float("inf"))
            improvement = (old_mae - best_mae) / old_mae * 100 if old_mae else 0
            if improvement > 0:
                lines.append(f"- **相对旧 AFSA 提升**: MAE 降低 {improvement:.1f}%\n")
            else:
                lines.append(f"- **相对旧 AFSA**: MAE 差 {abs(improvement):.1f}%\n")

        lines.append("\n")
    else:
        lines.append("*等待实验完成*\n\n")

    # 7. 图表
    lines.append("## 7. 可视化\n\n")
    lines.append("### 7.1 指标对比\n\n")
    fig_mae = FIGURES_DIR / hs / "metrics_mae_bar.png"
    fig_rmse = FIGURES_DIR / hs / "metrics_rmse_bar.png"
    fig_r2 = FIGURES_DIR / hs / "metrics_r2_bar.png"
    fig_time = FIGURES_DIR / hs / "training_time.png"
    fig_overlay = FIGURES_DIR / hs / "predictions_overlay.png"
    for fpath in [fig_mae, fig_rmse, fig_r2, fig_time, fig_overlay]:
        if fpath.exists():
            rel = fpath.relative_to(PROJECT_ROOT)
            lines.append(f"![{fpath.stem}]({rel})\n\n")

    lines.append("---\n\n")
    lines.append("*报告由 run_exp_p04_report.py 自动生成*\n")

    return "\n".join(lines)


def generate_figures(horizon: int, horizon_label: str, all_models: list[str], logger):
    """生成所有图表。"""
    hs = f"h{horizon}"
    hdir = SAMPLES_DIR / hs
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
        for col, out_name in [("MAE", "metrics_mae_bar"), ("RMSE", "metrics_rmse_bar"), ("R2", "metrics_r2_bar")]:
            out = fig_h / f"{out_name}.png"
            plt.figure(figsize=(8, 4))
            bars = plt.bar(display_names, [df_metrics.loc[df_metrics["display_name"] == dn, col].values[0] for dn in display_names])
            plt.title(f"{col} — {horizon_label}")
            plt.xticks(rotation=20, ha="right")
            for bar, val in zip(bars, [df_metrics.loc[df_metrics["display_name"] == dn, col].values[0] for dn in display_names]):
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                         f"{val:.3f}", ha="center", va="bottom", fontsize=8)
            plt.tight_layout()
            plt.savefig(out, dpi=150)
            plt.close()
            logger.info("  保存 %s", out.name)

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


def run_report(horizon: int, logger):
    horizon_cfg = load_config(f"exp_p04_h{horizon}.json")
    horizon_label = horizon_cfg["horizon_label"]
    hs = f"h{horizon}"

    all_models = horizon_cfg["baseline_models"] + list(horizon_cfg["model_search_space"].keys())

    logger.info("=" * 60)
    logger.info("生成报告  horizon=%s (%s)", horizon, horizon_label)

    # 生成图表
    generate_figures(horizon, horizon_label, all_models, logger)

    # 生成 Markdown
    md_text = _gen_markdown_report(horizon, horizon_label, all_models, logger)
    report_path = REPORTS_DIR / f"EXP-P04_h{horizon}_详细实验汇报.md"
    report_path.write_text(md_text, encoding="utf-8")
    logger.info("报告已保存: %s", report_path.relative_to(PROJECT_ROOT))

    # 跨 horizon 对比图（只在 horizon=16 时做一次）
    if horizon == 16:
        _plot_comparison_horizons(load_config("exp_p04_base.json"), logger)

    logger.info("报告生成完成！")


def main():
    parser = argparse.ArgumentParser(description="EXP-P04 报告生成")
    parser.add_argument("--horizon", type=int, choices=[1, 4, 16], required=True)
    args = parser.parse_args()

    horizon = args.horizon
    horizon_cfg = load_config(f"exp_p04_h{horizon}.json")
    log_file = horizon_cfg["log_file"].replace(".log", "_report.log")
    logger = setup_logger("report", log_file)
    logger.info("=" * 60)
    logger.info("EXP-P04 报告生成  horizon=%d", horizon)

    run_report(horizon, logger)


if __name__ == "__main__":
    main()
