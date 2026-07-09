"""
EXP-P05 报告生成：汇总主实验表（RMSE 优先）
python experiments/prediction/step5_new_experiments/run_exp_p05_report.py --horizon 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.prediction.step5_new_experiments.exp_p05_common import (
    BENCHMARK_DIR,
    METRICS_DIR,
    MODEL_DISPLAY_NAMES,
    REPORTS_DIR,
    ensure_dirs,
    setup_logger,
    sort_results_by_rmse,
)


HORIZON_LABEL = {1: "15min", 4: "1h", 16: "4h"}


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_main_table(horizon: int) -> pd.DataFrame:
    hdir = METRICS_DIR / f"h{horizon}"
    bench = load_json(BENCHMARK_DIR / f"h{horizon}" / "inference_benchmark.json")
    segmented = load_json(hdir / "segmented_metrics.json")
    baseline = load_json(hdir / "baseline_metrics.json")
    residual = load_json(hdir / "residual_metrics.json")

    rows = []
    sources = [
        ("persistence", baseline.get("persistence"), "daytime"),
        ("moving_average", baseline.get("moving_average"), "daytime"),
        ("ridge", baseline.get("ridge"), "daytime"),
        ("xgboost", baseline.get("xgboost"), "daytime"),
        ("lightgbm", baseline.get("lightgbm"), "daytime"),
    ]
    for key, m, _ in sources:
        if not m:
            continue
        rows.append(_row_from_metrics(key, m, horizon, bench))

    for key, m in (residual or {}).items():
        seg = segmented.get(key, {})
        m_use = seg.get("daytime_only", m)
        rows.append(_row_from_metrics(key, m_use, horizon, bench))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values(["RMSE ↓", "MAE ↓"], ascending=[True, True]).reset_index(drop=True)
    return df


def _row_from_metrics(key: str, m: dict, horizon: int, bench: dict) -> dict:
    display = MODEL_DISPLAY_NAMES.get(key.replace("_residual", ""), key)
    if key.endswith("_residual"):
        display = MODEL_DISPLAY_NAMES.get(key, key)
    infer = bench.get(key, {})
    return {
        "Model": display,
        "Horizon": HORIZON_LABEL.get(horizon, str(horizon)),
        "RMSE ↓": f"{m.get('RMSE', float('nan')):.4f}",
        "MAE ↓": f"{m.get('MAE', float('nan')):.4f}",
        "MAPE ↓": f"{m.get('MAPE', float('nan')):.2f}%",
        "R² ↑": f"{m.get('R2', float('nan')):.4f}",
        "nRMSE ↓": f"{m.get('nRMSE', float('nan')):.4f}",
        "Params": infer.get("params", "-"),
        "Inference (ms/sample)": f"{infer.get('ms_per_sample', float('nan')):.3f}" if infer else "-",
    }


def build_residual_comparison(horizon: int) -> pd.DataFrame:
    """直接预测 vs 残差预测对比表。"""
    hdir = METRICS_DIR / f"h{horizon}"
    residual = load_json(hdir / "residual_metrics.json")
    rows = []
    for key, m in (residual or {}).items():
        base = key.replace("_residual", "")
        rows.append({
            "Model": MODEL_DISPLAY_NAMES.get(base, base),
            "Horizon": HORIZON_LABEL.get(horizon, str(horizon)),
            "RMSE ↓ (residual)": f"{m.get('RMSE', float('nan')):.4f}",
            "MAE ↓ (residual)": f"{m.get('MAE', float('nan')):.4f}",
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="EXP-P05 报告生成")
    parser.add_argument("--horizon", type=int, choices=[1, 4, 16], required=True)
    args = parser.parse_args()

    ensure_dirs(REPORTS_DIR)
    logger = setup_logger("report", f"EXP-P05_h{args.horizon}_report.log")

    main_table = build_main_table(args.horizon)
    residual_table = build_residual_comparison(args.horizon)

    main_path = REPORTS_DIR / f"EXP-P05_h{args.horizon}_main_table.csv"
    residual_path = REPORTS_DIR / f"EXP-P05_h{args.horizon}_residual_comparison.csv"
    main_table.to_csv(main_path, index=False)
    residual_table.to_csv(residual_path, index=False)

    try:
        main_md = main_table.to_markdown(index=False) if not main_table.empty else "_暂无结果_"
        residual_md = residual_table.to_markdown(index=False) if not residual_table.empty else "_暂无结果_"
    except ImportError:
        main_md = main_table.to_string(index=False) if not main_table.empty else "_暂无结果_"
        residual_md = residual_table.to_string(index=False) if not residual_table.empty else "_暂无结果_"

    md_lines = [
        f"# EXP-P05 实验报告 (h{args.horizon})",
        "",
        "## 主实验表（Daytime RMSE 优先）",
        "",
        main_md,
        "",
        "## 残差预测对比",
        "",
        residual_md,
        "",
        "> 排名标准：RMSE（主要）> MAE（次要）",
    ]
    md_path = REPORTS_DIR / f"EXP-P05_h{args.horizon}_report.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    logger.info("报告已生成: %s", md_path)
    print(main_table.to_string(index=False) if not main_table.empty else "暂无主表数据")


if __name__ == "__main__":
    main()
