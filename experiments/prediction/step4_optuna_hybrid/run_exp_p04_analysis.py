"""
EXP-P04 跨 Horizon 综合对比分析 + h16 bilstm 高方差诊断
python -m experiments.prediction.step4_optuna_hybrid.run_exp_p04_analysis
"""

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
from experiments.prediction.step4_optuna_hybrid.exp_p04_common import METRICS_DIR

HORIZONS = ["h1", "h4", "h16"]
MODELS = ["lstm", "bilstm", "cnn_lstm", "cnn_bilstm", "minipatchtst"]
METRIC_NAMES = {"MAE": "MAE", "RMSE": "RMSE", "R2": "R2"}


def load_reproduce(hs: str, model: str) -> dict:
    path = METRICS_DIR / hs / f"{model}_reproduce.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_optuna(hs: str, model: str) -> dict:
    path = METRICS_DIR / hs / f"{model}_optuna.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_final_train(hs: str, model: str) -> dict:
    path = METRICS_DIR / hs / f"{model}_final_train_history.csv"
    return path


def cv_ratio(std_val: float, mean_val: float) -> float:
    """Coefficient of variation (CV = std/mean), as percentage."""
    if mean_val == 0:
        return float("nan")
    return abs(std_val / mean_val) * 100


def print_divider(title: str, width: int = 80):
    print()
    print("=" * width)
    print("  " + title)
    print("=" * width)


def print_subdivider(title: str, width: int = 80):
    print()
    print("-" * width)
    print("  " + title)
    print("-" * width)


# ─────────────────────────────────────────────
# 1. 跨 Horizon MAE 对比表
# ─────────────────────────────────────────────
print_divider("1. 跨 Horizon MAE 对比 (越小越好)")
print()
print("%-15s  %-8s  %-8s  %-8s  %-8s  %-8s  %-8s  %-8s" % (
    "Model", "h1 MAE", "h1 std", "h4 MAE", "h4 std", "h16 MAE", "h16 std", "h1/h16"
))
print("-" * 88)
for model in MODELS:
    mae_h1 = load_reproduce("h1", model)
    mae_h4 = load_reproduce("h4", model)
    mae_h16 = load_reproduce("h16", model)

    h1_mean = mae_h1.get("mean", {}).get("MAE", 0)
    h1_std = mae_h1.get("std", {}).get("MAE", 0)
    h4_mean = mae_h4.get("mean", {}).get("MAE", 0)
    h4_std = mae_h4.get("std", {}).get("MAE", 0)
    h16_mean = mae_h16.get("mean", {}).get("MAE", 0)
    h16_std = mae_h16.get("std", {}).get("MAE", 0)
    ratio = h16_mean / h1_mean if h1_mean > 0 else 0

    print("%-15s  %-8.5f  %-8.5f  %-8.5f  %-8.5f  %-8.5f  %-8.5f  %-8.2fx" % (
        model, h1_mean, h1_std, h4_mean, h4_std, h16_mean, h16_std, ratio
    ))

# ─────────────────────────────────────────────
# 2. 最佳模型横向排名
# ─────────────────────────────────────────────
print()
print_divider("2. 各 Horizon 最佳模型排名 (按 MAE)")
for hs in HORIZONS:
    rows = []
    for model in MODELS:
        data = load_reproduce(hs, model)
        if not data:
            continue
        m = data.get("mean", {}).get("MAE", 999)
        s = data.get("std", {}).get("MAE", 0)
        cv = cv_ratio(s, m)
        rows.append((model, m, s, cv))
    rows.sort(key=lambda x: x[1])

    print()
    print("  %-s" % hs.upper())
    print("  %-4s  %-18s  %-12s  %-12s  %-10s" % ("#", "Model", "MAE", "std", "CV(%)"))
    print("  " + "-" * 60)
    for i, (model, m, s, cv) in enumerate(rows, 1):
        flag = " <-- BEST" if i == 1 else ""
        print("  %-4d  %-18s  %-12.6f  %-12.6f  %-10.2f%s" % (i, model, m, s, cv, flag))

# ─────────────────────────────────────────────
# 3. R2 跨 Horizon 对比
# ─────────────────────────────────────────────
print()
print_divider("3. 跨 Horizon R2 对比 (越大越好, 1.0=完美)")
print()
print("%-15s  %-10s  %-10s  %-10s" % ("Model", "h1 R2", "h4 R2", "h16 R2"))
print("-" * 50)
for model in MODELS:
    h1 = load_reproduce("h1", model)
    h4 = load_reproduce("h4", model)
    h16 = load_reproduce("h16", model)
    r1 = h1.get("mean", {}).get("R2", 0)
    r4 = h4.get("mean", {}).get("R2", 0)
    r16 = h16.get("mean", {}).get("R2", 0)
    print("%-15s  %-10.4f  %-10.4f  %-10.4f" % (model, r1, r4, r16))

# ─────────────────────────────────────────────
# 4. h16 bilstm 高方差深度诊断
# ─────────────────────────────────────────────
print()
print_divider("4. h16 bilstm 高方差诊断")

# 4a. 对比其他模型在 h16 的 CV
print()
print("  4a. 各模型 h16 MAE 稳定性 (CV = std/mean * 100)")
print("  %-15s  %-10s  %-10s  %-10s  %-10s" % ("Model", "mean", "std", "CV(%)", "稳定评级"))
print("  " + "-" * 60)
cv_list = []
for model in MODELS:
    d = load_reproduce("h16", model)
    m = d.get("mean", {}).get("MAE", 0)
    s = d.get("std", {}).get("MAE", 0)
    cv = cv_ratio(s, m)
    cv_list.append((model, cv))
    if cv < 5:
        rating = "STABLE"
    elif cv < 10:
        rating = "MODERATE"
    else:
        rating = "UNSTABLE <--"
    print("  %-15s  %-10.6f  %-10.6f  %-10.2f  %s" % (model, m, s, cv, rating))

print()
print("  结论: bilstm h16 CV=%.1f%%, 远超其他模型 (<10%%)" % (
    dict(cv_list).get("bilstm", 0)
))

# 4b. bilstm 跨 horizon 稳定性对比
print()
print("  4b. bilstm 跨 horizon 稳定性 (CV = std/mean * 100)")
print("  %-6s  %-10s  %-10s  %-10s  %-10s" % ("Horizon", "mean MAE", "std MAE", "CV(%)", "评级"))
print("  " + "-" * 60)
for hs, hlabel in [("h1", "1-step"), ("h4", "4-step"), ("h16", "16-step")]:
    d = load_reproduce(hs, "bilstm")
    m = d.get("mean", {}).get("MAE", 0)
    s = d.get("std", {}).get("MAE", 0)
    cv = cv_ratio(s, m)
    if cv < 5:
        rating = "STABLE"
    elif cv < 10:
        rating = "MODERATE"
    else:
        rating = "UNSTABLE <--"
    print("  %-6s  %-10.6f  %-10.6f  %-10.2f  %s" % (hlabel, m, s, cv, rating))

# 4c. bilstm Optuna 参数对比
print()
print("  4c. bilstm Optuna 最优参数 (h1 vs h16)")
print("  提示: note='baseline_fixed_params' 表示使用默认参数, 未实际调参")
for hs in ["h1", "h16"]:
    d = load_optuna(hs, "bilstm")
    if d:
        print("  [%s] params=%s  note='%s'  best_value=%.6f" % (
            hs, d.get("best_params", {}), d.get("note", ""), d.get("best_value", 0)
        ))

# 4d. seed 级别数据对比 (bilstm h16)
print()
print("  4d. bilstm h16 各 seed 详细数据")
print("  %-6s  %-10s  %-10s  %-10s  %-10s  %-10s" % (
    "seed", "MAE", "RMSE", "R2", "MAPE", "train_time"))
print("  " + "-" * 70)
d = load_reproduce("h16", "bilstm")
for entry in d.get("per_seed", []):
    print("  %-6d  %-10.5f  %-10.5f  %-10.5f  %-10.2f  %-10.1fs" % (
        entry["seed"], entry["MAE"], entry["RMSE"],
        entry["R2"], entry["MAPE"], entry["training_time_sec"]))

# 4e. 根因分析
print()
print("  4e. 根因分析")
print("""
  1. h16 bilstm 的 best_params 与 h1 完全相同 (hidden=64, layers=2, lr=0.001)
     -> note='baseline_fixed_params' 说明 bilstm 未经过 Optuna 调参
     -> h16 预测 16 步 (4小时) 与 h1 预测 1 步 (15分钟) 任务难度差异巨大
     -> 相同的超参数对 h16 欠拟合, 导致训练不稳定

  2. seed=42 初始化最差 (MAE=0.0481), seed=43 最好 (MAE=0.0379)
     -> bilstm 是双向结构, 16步预测需要重建未来上下文
     -> 权重初始化差异在小模型上放大, 造成预测结果高度分散

  3. bilstm 在 h1 (CV=0.25%) vs h16 (CV=12.4%) 相差 50 倍
     -> 单向 lstm/cnn_lstm/cnn_bilstm 在 h16 均保持稳定 (CV<5%)
     -> 问题根因: bidirectional 特性在长 horizon 上的放大效应

  建议:
  - 对 h16 bilstm 重新运行 Optuna 调参 (hidden 建议 128-256, layers 建议 2-3)
  - 或将 bilstm 从 h16 对比中排除 (已有 cnn_bilstm 作为替代双向模型)
  - 增加更多 seed (如 5-10) 以平滑随机性影响
""")

# ─────────────────────────────────────────────
# 5. 训练时间对比
# ─────────────────────────────────────────────
print()
print_divider("5. 训练时间对比 (秒, 越短越好)")
print()
print("%-15s  %-12s  %-12s  %-12s  %-12s" % (
    "Model", "h1 (s)", "h4 (s)", "h16 (s)", "h1/h16 ratio"))
print("-" * 70)
for model in MODELS:
    h1 = load_reproduce("h1", model).get("mean", {}).get("training_time_sec", 0)
    h4 = load_reproduce("h4", model).get("mean", {}).get("training_time_sec", 0)
    h16 = load_reproduce("h16", model).get("mean", {}).get("training_time_sec", 0)
    ratio = h1 / h16 if h16 > 0 else 0
    print("%-15s  %-12.1f  %-12.1f  %-12.1f  %-12.2fx" % (
        model, h1, h4, h16, ratio))

# ─────────────────────────────────────────────
# 6. 最终推荐
# ─────────────────────────────────────────────
print()
print_divider("6. 各 Horizon 推荐模型")
for hs in HORIZONS:
    rows = []
    for model in MODELS:
        # 排除 bilstm h16 (不稳定)
        if model == "bilstm" and hs == "h16":
            continue
        d = load_reproduce(hs, model)
        m = d.get("mean", {}).get("MAE", 999)
        s = d.get("std", {}).get("MAE", 0)
        r = d.get("mean", {}).get("R2", 0)
        rows.append((model, m, s, r))
    rows.sort(key=lambda x: x[1])
    best = rows[0]
    print()
    print("  %s: 最佳模型 = %s  (MAE=%.5f±%.5f, R2=%.4f)" % (
        hs.upper(), best[0], best[1], best[2], best[3]))

print()
print("=" * 80)
print("  分析完成!")
print("=" * 80)
