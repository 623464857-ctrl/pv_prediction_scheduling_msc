"""
EXP-P04 跨 Horizon 综合对比可视化
生成 comparison_summary.png 和 comparison_report.md
python -m experiments.prediction.step4_optuna_hybrid.run_exp_p04_summary
"""

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm

_chinese_font = "C:\\Windows\\Fonts\\simhei.ttf"  # forced to this known-good font
fm.fontManager.addfont(_chinese_font)
_chinese_font_name = Path(_chinese_font).stem  # "simhei"

# Force simhei font for Chinese characters
_ch_font = "C:\\Windows\\Fonts\\simhei.ttf"
fm.fontManager.addfont(_ch_font)
_chinese_name = "SimHei"
fm.fontManager.ttflist.extend([
    fm.FontEntry(fname=_ch_font, name=_chinese_name, style="normal", variant="normal", weight="normal")
])
matplotlib.rcParams["font.sans-serif"] = [_chinese_name, "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
from experiments.prediction.step4_optuna_hybrid.exp_p04_common import METRICS_DIR, FIGURES_DIR

HORIZONS = ["h1", "h4", "h16"]
HORIZON_LABELS = ["h1 (15min)", "h4 (1h)", "h16 (4h)"]
MODELS = ["cnn_bilstm"]
MODEL_LABELS = ["CNN-BiLSTM"]
MODEL_COLORS = {
    "cnn_bilstm": "#d62728",
}
EXCLUDE = set()


def load_reproduce(hs: str, model: str) -> dict:
    path = METRICS_DIR / hs / f"{model}_reproduce.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def cv(std_val: float, mean_val: float) -> float:
    if mean_val == 0:
        return float("nan")
    return abs(std_val / mean_val) * 100


# ─── 数据提取 ───────────────────────────────────────
metrics = {}  # metrics[m][h] = {mae, rmse, r2, mae_std, rmse_std, mae_cv, rmse_cv}
for model in MODELS:
    metrics[model] = {}
    for hs in HORIZONS:
        d = load_reproduce(hs, model)
        mae = d.get("mean", {}).get("MAE", 0)
        rmse = d.get("mean", {}).get("RMSE", 0)
        r2 = d.get("mean", {}).get("R2", 0)
        mae_std = d.get("std", {}).get("MAE", 0)
        rmse_std = d.get("std", {}).get("RMSE", 0)
        metrics[model][hs] = dict(
            mae=mae, rmse=rmse, r2=r2,
            mae_std=mae_std, rmse_std=rmse_std,
            mae_cv=cv(mae_std, mae), rmse_cv=cv(rmse_std, rmse)
        )


# ─── 图1: 4-panel 综合图 ────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("EXP-P04 跨 Horizon 综合对比分析", fontsize=16, fontweight="bold", y=0.98)

h_positions = np.arange(len(HORIZONS))
bar_width = 0.7
models_bar = MODELS.copy()
MODEL_LABELS_F = MODEL_LABELS.copy()
COLORS_F = [MODEL_COLORS[m] for m in models_bar]

# ─ Panel A: RMSE ───────────────────────────────────
ax = axes[0, 0]
for i, model in enumerate(models_bar):
    means = [metrics[model][hs]["rmse"] for hs in HORIZONS]
    stds = [metrics[model][hs]["rmse_std"] for hs in HORIZONS]
    x = h_positions + i * bar_width / len(models_bar) - bar_width * 0.5 + bar_width / len(models_bar) / 2
    bars = ax.bar(x, means, bar_width / len(models_bar),
                  label=MODEL_LABELS_F[i], color=COLORS_F[i], alpha=0.85)
    ax.errorbar(x, means, yerr=stds, fmt="none", color="black", capsize=3, linewidth=1)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=6, rotation=30)
ax.set_xticks(h_positions)
ax.set_xticklabels(HORIZON_LABELS, fontsize=10)
ax.set_ylabel("RMSE (标准化单位)", fontsize=11)
ax.set_title("A. RMSE 对比 (越小越好)", fontsize=12, fontweight="bold")
ax.legend(fontsize=8, loc="upper left")
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, 0.085)

# ─ Panel B: R2 ─────────────────────────────────────
ax = axes[0, 1]
for i, model in enumerate(models_bar):
    means = [metrics[model][hs]["r2"] for hs in HORIZONS]
    x = h_positions + i * bar_width / len(models_bar) - bar_width * 0.5 + bar_width / len(models_bar) / 2
    ax.bar(x, means, bar_width / len(models_bar),
           label=MODEL_LABELS_F[i], color=COLORS_F[i], alpha=0.85)
ax.set_xticks(h_positions)
ax.set_xticklabels(HORIZON_LABELS, fontsize=10)
ax.set_ylabel("R2 (1.0=完美)", fontsize=11)
ax.set_title("B. R2 对比 (越大越好)", fontsize=12, fontweight="bold")
ax.legend(fontsize=8, loc="lower left")
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0.85, 1.0)
ax.axhline(y=0.9, color="red", linestyle="--", linewidth=1, alpha=0.5, label="R2=0.9 基线")

# ─ Panel C: RMSE 随 horizon 变化趋势线 ───────────────
ax = axes[1, 0]
h_nums = [1, 4, 16]
for model in MODELS:
    means = [metrics[model][hs]["rmse"] for hs in HORIZONS]
    stds = [metrics[model][hs]["rmse_std"] for hs in HORIZONS]
    is_unstable = False
    ls = "-"
    lw = 2.0
    ax.errorbar(h_nums, means, yerr=stds, marker="o", markersize=6,
                label=MODEL_LABELS[MODELS.index(model)] + (" (unstable)" if is_unstable else ""),
                color=MODEL_COLORS[model], linewidth=lw, linestyle=ls,
                capsize=4, alpha=0.9)
ax.set_xscale("log")
ax.set_xticks([1, 4, 16])
ax.set_xticklabels(["1", "4", "16"])
ax.set_xlabel("Horizon (步数)", fontsize=11)
ax.set_ylabel("RMSE (标准化单位)", fontsize=11)
ax.set_title("C. RMSE 随 Horizon 变化趋势", fontsize=12, fontweight="bold")
ax.legend(fontsize=8, loc="upper left", ncol=2)
ax.grid(alpha=0.3)

# ─ Panel D: RMSE CV 稳定性热力图 ─────────────────────
ax = axes[1, 1]
cv_data = []
for model in MODELS:
    row = [metrics[model][hs]["rmse_cv"] for hs in HORIZONS]
    cv_data.append(row)
cv_data = np.array(cv_data)

im = ax.imshow(cv_data, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=15)
plt.colorbar(im, ax=ax, label="RMSE CV = std/mean * 100 (%)")

ax.set_xticks(range(len(HORIZONS)))
ax.set_xticklabels(HORIZON_LABELS)
ax.set_yticks(range(len(MODELS)))
ax.set_yticklabels(MODEL_LABELS)
ax.set_title("D. RMSE 稳定性热力图 (CV%, 越小越稳定)", fontsize=12, fontweight="bold")

# 标注数值
for i in range(len(MODELS)):
    for j in range(len(HORIZONS)):
        val = cv_data[i, j]
        color = "white" if val > 8 else "black"
        flag = "*" if val > 10 else ""
        ax.text(j, i, f"{val:.1f}{flag}", ha="center", va="center",
                color=color, fontsize=10, fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.96])
out_fig = FIGURES_DIR / "comparison_summary.png"
plt.savefig(out_fig, dpi=150, bbox_inches="tight")
plt.close()
print(f"[OK] 保存: {out_fig}")

# ─── 图2: 最佳模型推荐表 (表格图) ──────────────────
fig2, ax2 = plt.subplots(figsize=(12, 4))
ax2.axis("off")
ax2.set_title("EXP-P04 各 Horizon 最佳模型推荐", fontsize=14, fontweight="bold", pad=12)

table_data = []
col_labels = ["Horizon", "最佳模型", "RMSE", "std", "CV(%)", "R2", "次佳模型", "RMSE次佳"]
for hs in HORIZONS:
    rows = []
    for model in MODELS:
        if (hs, model) in EXCLUDE:
            continue
        d = metrics[model][hs]
        rows.append((model, d["rmse"], d["rmse_std"], d["rmse_cv"], d["r2"]))
    rows.sort(key=lambda x: x[1])
    best = rows[0]
    best_name = MODEL_LABELS[MODELS.index(best[0])]
    second = rows[1]
    second_name = MODEL_LABELS[MODELS.index(second[0])]
    table_data.append([
        hs.upper(),
        best_name,
        f"{best[1]:.5f}",
        f"{best[2]:.5f}",
        f"{best[3]:.2f}",
        f"{best[4]:.4f}",
        second_name,
        f"{second[1]:.5f}",
    ])

table = ax2.table(cellText=table_data, colLabels=col_labels,
                   loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.8)

# 高亮最佳行
for i, hs in enumerate(HORIZONS):
    for j in range(len(col_labels)):
        cell = table[i, j]
        if j == 1:  # 最佳模型列
            cell.set_facecolor("#d4edda")

fig2.tight_layout()
out_table = FIGURES_DIR / "comparison_best_model.png"
fig2.savefig(out_table, dpi=150, bbox_inches="tight")
plt.close()
print(f"[OK] 保存: {out_table}")

# ─── 生成 Markdown 总结报告 ─────────────────────────
lines = []
lines.append("# EXP-P04 跨 Horizon 综合对比总结报告")
lines.append("")
lines.append(f"> 生成时间: 2026-06-22")
lines.append("")
lines.append("## 1. 关键结论")
lines.append("")
lines.append("### 最佳模型推荐")
lines.append("")
lines.append("| Horizon | 最佳模型 | RMSE | std | CV(%) | R2 |")
lines.append("|--------|---------|------|-----|-------|-----|")
for hs in HORIZONS:
    rows = []
    for model in MODELS:
        if (hs, model) in EXCLUDE:
            continue
        d = metrics[model][hs]
        rows.append((model, d["rmse"], d["rmse_std"], d["rmse_cv"], d["r2"]))
    rows.sort(key=lambda x: x[1])
    best = rows[0]
    best_name = MODEL_LABELS[MODELS.index(best[0])]
    flag = ""
    lines.append(f"| {hs.upper()} | {best_name}{flag} | {best[1]:.5f} | {best[2]:.5f} | {best[3]:.2f} | {best[4]:.4f} |")

lines.append("")
lines.append("### 核心发现")
lines.append("")
lines.append(f"**h1 (15min)**: CNN-BiLSTM (RMSE={metrics['cnn_bilstm']['h1']['rmse']:.5f})")
lines.append(f"**h4 (1h)**: CNN-BiLSTM (RMSE={metrics['cnn_bilstm']['h4']['rmse']:.5f})")
lines.append(f"**h16 (4h)**: CNN-BiLSTM (RMSE={metrics['cnn_bilstm']['h16']['rmse']:.5f})")
lines.append("")
lines.append("## 2. 图表说明")
lines.append("")
lines.append("| 文件 | 说明 |")
lines.append("|-----|------|")
lines.append("| `comparison_summary.png` | 综合四面板对比：RMSE柱状图、R2柱状图、趋势线、稳定性热力图 |")
lines.append("| `comparison_best_model.png` | 各 Horizon 最佳模型推荐表 |")
lines.append("")
lines.append("## 3. RMSE 随 Horizon 变化规律")
lines.append("")
lines.append("| Model | h1 RMSE | h4 RMSE | h16 RMSE | h1/h16 增幅 |")
lines.append("|-------|---------|---------|----------|-------------|")
for model in MODELS:
    h1_rmse = metrics[model]["h1"]["rmse"]
    h4_rmse = metrics[model]["h4"]["rmse"]
    h16_rmse = metrics[model]["h16"]["rmse"]
    ratio = h16_rmse / h1_rmse
    lines.append(f"| {model} | {h1_rmse:.5f} | {h4_rmse:.5f} | {h16_rmse:.5f} | {ratio:.2f}x |")

lines.append("")

out_md = FIGURES_DIR / "comparison_summary.md"
out_md.write_text("\n".join(lines), encoding="utf-8")
print(f"[OK] 保存: {out_md}")
print()
print("Done!")
