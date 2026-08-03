"""EXP-P05 主入口：按实验计划编排 Step1-6（不自动运行时可分步调用）。"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.prediction.step5_new_experiments.exp_p05_common import (
    BENCHMARK_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    PRED_DIR,
    MODEL_DISPLAY_NAMES,
    compute_all_metrics,
    ensure_dirs,
    load_config,
    load_meta,
    load_samples,
    setup_logger,
)

logger = logging.getLogger(__name__)


def run_step_prepare(horizon: int) -> None:
    from experiments.prediction.step5_new_experiments.run_exp_p05_prepare_samples import main as prepare_main

    sys.argv = ["run_exp_p05_prepare_samples.py", "--horizon", str(horizon)]
    prepare_main()


def run_step_baselines(horizon: int) -> dict:
    from experiments.prediction.step5_new_experiments.run_exp_p05_baselines import main as baseline_main

    sys.argv = ["run_exp_p05_baselines.py", "--horizon", str(horizon)]
    baseline_main()
    path = METRICS_DIR / f"h{horizon}" / "baseline_metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def run_step_residual(horizon: int) -> dict:
    from experiments.prediction.step5_new_experiments.run_exp_p05_residual_train import main as residual_main

    sys.argv = ["run_exp_p05_residual_train.py", "--horizon", str(horizon)]
    residual_main()
    path = METRICS_DIR / f"h{horizon}" / "residual_metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def run_step_evaluation(horizon: int) -> None:
    from experiments.prediction.step5_new_experiments.run_exp_p05_evaluation import main as eval_main

    sys.argv = ["run_exp_p05_evaluation.py", "--horizon", str(horizon)]
    eval_main()


def run_step_hybrid(horizon: int, all_strategies: bool = False) -> None:
    from experiments.prediction.step5_new_experiments.run_exp_p05_hybrid_search import main as hybrid_main

    sys.argv = ["run_exp_p05_hybrid_search.py", "--horizon", str(horizon)]
    if all_strategies:
        sys.argv.append("--all")
    else:
        sys.argv.extend(["--strategy", "S6"])
    hybrid_main()


def run_step_report(horizon: int) -> pd.DataFrame:
    from experiments.prediction.step5_new_experiments.run_exp_p05_report import build_main_table

    return build_main_table(horizon)


def build_comparison_table(horizon: int, all_results: dict) -> pd.DataFrame:
    rows = []
    order = [
        "persistence", "moving_average", "ridge", "xgboost", "lightgbm",
        "cnn_bilstm_residual",
    ]
    for key in order:
        if key not in all_results:
            continue
        m = all_results[key]
        display = MODEL_DISPLAY_NAMES.get(key, key)
        rows.append({
            "Model": display,
            "RMSE ↓": f"{m.get('RMSE', float('nan')):.4f}",
            "MAE ↓": f"{m.get('MAE', float('nan')):.4f}",
            "MAPE ↓": f"{m.get('MAPE', float('nan')):.2f}%",
            "R² ↑": f"{m.get('R2', float('nan')):.4f}",
            "nRMSE ↓": f"{m.get('nRMSE', float('nan')):.4f}",
        })
    return pd.DataFrame(rows)


def run_pipeline(horizon: int, steps: list[str] | None = None, hybrid_all: bool = False):
    """
    执行顺序（默认全跑）：
    prepare -> baselines -> residual -> evaluation -> hybrid -> report
    """
    cfg = load_config()
    ensure_dirs(
        METRICS_DIR / f"h{horizon}",
        MODELS_DIR / f"h{horizon}",
        PRED_DIR / f"h{horizon}",
        FIGURES_DIR / f"h{horizon}",
        BENCHMARK_DIR / f"h{horizon}",
    )

    log_file = f"EXP-P05_h{horizon}_pipeline.log"
    logger_pipeline = setup_logger(f"pipeline_h{horizon}", log_file)
    logger_pipeline.info("=" * 60)
    logger_pipeline.info("EXP-P05 实验流程 horizon=%d", horizon)

    steps = steps or ["prepare", "baselines", "residual", "evaluation", "hybrid", "report"]
    all_results = {}

    logger_pipeline.info("将执行步骤: %s", " -> ".join(steps))

    if "prepare" in steps:
        logger_pipeline.info("Step2 特征增强 + 样本构造")
        run_step_prepare(horizon)

    if "baselines" in steps:
        logger_pipeline.info("Step1 强基线")
        baseline_results = run_step_baselines(horizon)
        all_results.update(baseline_results)

    if "residual" in steps:
        logger_pipeline.info("Step3 残差预测 (5 models)")
        residual_results = run_step_residual(horizon)
        all_results.update(residual_results)

    if "evaluation" in steps:
        logger_pipeline.info("Step4+5 分段评价 + 推理计时")
        run_step_evaluation(horizon)

    if "hybrid" in steps:
        logger_pipeline.info("Step6 Optuna-AFSA 混合搜索")
        run_step_hybrid(horizon, all_strategies=hybrid_all)

    comparison_df = pd.DataFrame()
    if "report" in steps:
        logger_pipeline.info("生成报告")
        from experiments.prediction.step5_new_experiments.run_exp_p05_report import main as report_main

        sys.argv = ["run_exp_p05_report.py", "--horizon", str(horizon)]
        report_main()
        comparison_df = run_step_report(horizon)

    if all_results:
        comparison_df = build_comparison_table(horizon, all_results)
        comparison_path = METRICS_DIR / f"h{horizon}" / "comparison_table.csv"
        comparison_df.to_csv(comparison_path, index=False)
        summary_path = METRICS_DIR / f"h{horizon}" / "summary.json"
        summary_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")

    logger_pipeline.info("EXP-P05 horizon=%d 完成", horizon)
    return all_results, comparison_df


def main():
    parser = argparse.ArgumentParser(description="EXP-P05 实验主流程")
    parser.add_argument("--horizon", type=int, choices=[1, 4, 16], required=True)
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["prepare", "baselines", "residual", "evaluation", "hybrid", "report"],
        default=None,
        help="指定执行的步骤，默认全部",
    )
    parser.add_argument("--hybrid-all", action="store_true", help="混合搜索运行 S1-S6 全部消融")
    args = parser.parse_args()

    all_results, comparison_df = run_pipeline(args.horizon, steps=args.steps, hybrid_all=args.hybrid_all)

    print("=" * 80)
    print(f"EXP-P05 实验结果 (Horizon=h{args.horizon})")
    print("=" * 80)
    if not comparison_df.empty:
        try:
            print(comparison_df.to_string(index=False))
        except UnicodeEncodeError:
            print(comparison_df.to_string(index=False).encode("utf-8", errors="replace").decode("utf-8", errors="replace"))
    else:
        print("（暂无对比表，请先运行 prepare/baselines/residual 步骤）")
    print("=" * 80)


if __name__ == "__main__":
    main()
