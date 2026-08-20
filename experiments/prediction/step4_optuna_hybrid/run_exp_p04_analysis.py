"""
EXP-P04 CNN-BiLSTM 跨 Horizon 综合分析
python -m experiments.prediction.step4_optuna_hybrid.run_exp_p04_analysis
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
from experiments.prediction.step4_optuna_hybrid.exp_p04_common import METRICS_DIR

HORIZONS = ["h1", "h4", "h16"]
MODEL = "cnn_bilstm"
METRIC_NAMES = {"MAE": "MAE", "RMSE": "RMSE", "R2": "R2"}


def load_reproduce(hs: str) -> dict:
    path = METRICS_DIR / hs / f"{MODEL}_reproduce.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_optuna(hs: str) -> dict:
    path = METRICS_DIR / hs / f"{MODEL}_optuna.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def cv_ratio(std_val: float, mean_val: float) -> float:
    if mean_val == 0:
        return float("nan")
    return abs(std_val / mean_val) * 100


def print_divider(title: str, width: int = 80):
    print()
    print("=" * width)
    print("  " + title)
    print("=" * width)


# ─────────────────────────────────────────────
# 1. 跨 Horizon MAE 对比表
# ─────────────────────────────────────────────
print_divider("1. CNN-BiLSTM 跨 Horizon MAE 对比")
print()
print("%-10s  %-12s  %-12s  %-12s  %-10s" % (
    "Horizon", "MAE mean", "MAE std", "CV(%)", "评级"))
print("-" * 60)
for hs in HORIZONS:
    d = load_reproduce(hs)
    m = d.get("mean", {}).get("MAE", 0)
    s = d.get("std", {}).get("MAE", 0)
    cv = cv_ratio(s, m)
    if cv < 5:
        rating = "STABLE"
    elif cv < 10:
        rating = "MODERATE"
    else:
        rating = "UNSTABLE"
    print("%-10s  %-12.6f  %-12.6f  %-12.2f  %s" % (hs, m, s, cv, rating))

# ─────────────────────────────────────────────
# 2. 跨 Horizon RMSE / R2
# ─────────────────────────────────────────────
print()
print_divider("2. RMSE / R2 跨 Horizon")
print()
print("%-10s  %-12s  %-12s  %-10s" % ("Horizon", "RMSE mean", "RMSE std", "R2 mean"))
print("-" * 50)
for hs in HORIZONS:
    d = load_reproduce(hs)
    rmse = d.get("mean", {}).get("RMSE", 0)
    rmse_std = d.get("std", {}).get("RMSE", 0)
    r2 = d.get("mean", {}).get("R2", 0)
    print("%-10s  %-12.6f  %-12.6f  %-10.4f" % (hs, rmse, rmse_std, r2))

# ─────────────────────────────────────────────
# 3. Optuna 最优参数
# ─────────────────────────────────────────────
print()
print_divider("3. Optuna 最优超参数")
for hs in HORIZONS:
    d = load_optuna(hs)
    if d:
        print(f"  [{hs}] params={d.get('best_params', {})}  best_value={d.get('best_value', 0):.6f}")
    else:
        print(f"  [{hs}] 无 Optuna 结果")

# ─────────────────────────────────────────────
# 4. 训练时间
# ─────────────────────────────────────────────
print()
print_divider("4. 训练时间")
print()
print("%-10s  %-15s" % ("Horizon", "Mean time (s)"))
print("-" * 30)
for hs in HORIZONS:
    d = load_reproduce(hs)
    t = d.get("mean", {}).get("training_time_sec", 0)
    print("%-10s  %-15.1f" % (hs, t))

# ─────────────────────────────────────────────
# 5. 种子级别详细数据
# ─────────────────────────────────────────────
print()
print_divider("5. 各 Seed 详细数据")
for hs in HORIZONS:
    d = load_reproduce(hs)
    if not d:
        continue
    print(f"\n  {hs.upper()}")
    print("  %-6s  %-10s  %-10s  %-10s  %-10s  %-10s" % (
        "seed", "MAE", "RMSE", "R2", "MAPE", "time(s)"))
    print("  " + "-" * 60)
    for entry in d.get("per_seed", []):
        print("  %-6d  %-10.5f  %-10.5f  %-10.5f  %-10.2f  %-10.1f" % (
            entry["seed"], entry["MAE"], entry["RMSE"],
            entry["R2"], entry["MAPE"], entry["training_time_sec"]))

print()
print("=" * 80)
print("  分析完成!")
print("=" * 80)
