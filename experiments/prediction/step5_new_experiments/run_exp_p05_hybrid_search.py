"""
Step 6: Optuna-AFSA 混合搜索消融 (S1-S6)
python experiments/prediction/step5_new_experiments/run_exp_p05_hybrid_search.py --horizon 1 --strategy S2
python experiments/prediction/step5_new_experiments/run_exp_p05_hybrid_search.py --horizon 1 --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.prediction.step5_new_experiments.exp_p05_common import (
    METRICS_DIR,
    ensure_dirs,
    load_config,
    load_meta,
    load_samples,
    setup_logger,
)
from experiments.prediction.step5_new_experiments.exp_p05_hybrid_search import run_hybrid_ablation

STRATEGIES = ["S2", "S3", "S4", "S5", "S6"]


def main():
    parser = argparse.ArgumentParser(description="EXP-P05 Optuna-AFSA 混合搜索")
    parser.add_argument("--horizon", type=int, choices=[1, 4, 16], required=True)
    parser.add_argument("--strategy", type=str, choices=STRATEGIES, default="S2")
    parser.add_argument("--all", action="store_true", help="运行 S1-S6 全部消融")
    args = parser.parse_args()

    cfg = load_config()
    ensure_dirs(METRICS_DIR / f"h{args.horizon}")
    logger = setup_logger("hybrid_search", f"EXP-P05_h{args.horizon}_hybrid_search.log")

    samples = load_samples(args.horizon, use_step5=True)
    meta = load_meta(args.horizon, use_step5=True)
    model_name = cfg["hybrid_search"]["target_model"]

    strategies = STRATEGIES if args.all else [args.strategy]
    all_results = {}
    for s in strategies:
        logger.info("运行策略 %s ...", s)
        result = run_hybrid_ablation(s, model_name, samples, meta, args.horizon, cfg)
        # 去掉不可序列化的 model_state
        best = {k: v for k, v in result["best"].items() if k != "model_state"}
        all_results[s] = {"strategy": s, "trials": result["trials"], "best": best}

    out = METRICS_DIR / f"h{args.horizon}" / "hybrid_search_ablation.json"
    out.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("混合搜索完成: %s", out)


if __name__ == "__main__":
    main()
