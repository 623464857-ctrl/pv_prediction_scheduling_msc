"""
Step 6: Optuna-AFSA 混合搜索（全模型 × 全 horizon × 全策略，对齐 Step4 搜索策略）

与 Step4 策略完全一致：
    Step 1: 用训练集后 1/3 做 quick subset 快速筛选参数（trial 级别）
    Step 2: 用最优参数在完整 3-fold CV 上评估，得到稳健的 val_RMSE

保留 S1-S6 混合消融实验。

执行方式：
    # 单模型 + 单策略
    python experiments/prediction/step5_new_experiments/run_exp_p06_hybrid_search.py --horizon 1 --strategy S2 --model cnn_lstm
    # 单 horizon + 全策略（S1-S6）
    python experiments/prediction/step5_new_experiments/run_exp_p06_hybrid_search.py --horizon 1 --all-strategies
    # 全 horizon + 全模型 + 全策略
    python experiments/prediction/step5_new_experiments/run_exp_p06_hybrid_search.py --all-horizons --all-strategies
    # 关闭 quick subset（使用完整训练数据搜索）
    python experiments/prediction/step5_new_experiments/run_exp_p06_hybrid_search.py --horizon 1 --no-quick --all-strategies

输出：
    data/prediction/step5_new_experiments/metrics/h{1,4,16}/hybrid_search_full.json
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
    setup_logger,
)
from experiments.prediction.step5_new_experiments.exp_p05_residual import (
    compute_residual_targets,
    fit_residual_scaler,
    inverse_transform_residual,
    reconstruct_from_residual,
    transform_residual,
)
from experiments.prediction.step5_new_experiments.exp_p06_hybrid_search import run_hybrid_ablation

STRATEGIES = ["S1", "S2", "S3", "S4", "S5", "S6"]
MODELS = ["cnn_bilstm"]


def load_config() -> dict:
    path = PROJECT_ROOT / "data" / "prediction" / "step5_new_experiments" / "config" / "exp_p06_config.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_samples(horizon: int) -> dict:
    """加载 Step5 的样本数据（含残差目标所需的 y_last）。"""
    from experiments.prediction.step5_new_experiments.exp_p05_common import SAMPLES_DIR

    hdir = SAMPLES_DIR / f"h{horizon}"
    data = {
        "X_train_seq": _np_load(hdir / "X_train_seq.npy"),
        "X_val_seq": _np_load(hdir / "X_val_seq.npy"),
        "X_test_seq": _np_load(hdir / "X_test_seq.npy"),
        "y_train": _np_load(hdir / "y_train.npy"),
        "y_val": _np_load(hdir / "y_val.npy"),
        "y_test": _np_load(hdir / "y_test.npy"),
    }
    for split in ("train", "val", "test"):
        p = hdir / f"y_{split}_raw.npy"
        if p.exists():
            data[f"y_{split}_raw"] = _np_load(p)
    for split in ("train", "val"):
        p = hdir / f"y_last_{split}.npy"
        if p.exists():
            data[f"y_last_{split}"] = _np_load(p)
    p = hdir / "y_last_test.npy"
    if p.exists():
        data["y_last_test"] = _np_load(p)
    return data


def _np_load(path: Path) -> "np.ndarray":
    import numpy as np
    return np.load(path)


def load_meta(horizon: int) -> dict:
    from experiments.prediction.step5_new_experiments.exp_p05_common import SAMPLES_DIR

    hdir = SAMPLES_DIR / f"h{horizon}"
    with open(hdir / "meta.json", encoding="utf-8") as f:
        return json.load(f)


def prepare_residual_data(samples: dict, horizon: int):
    """
    将样本数据转换为残差目标（Δy = y_future - y_last），用于混合搜索训练。
    替换 y_train / y_val / y_test 为残差目标，返回 (samples, res_scaler)。
    """
    y_res_train = compute_residual_targets(samples["y_train_raw"], samples["y_last_train"])
    y_res_val = compute_residual_targets(samples["y_val_raw"], samples["y_last_val"])
    res_scaler = fit_residual_scaler(y_res_train)

    samples["y_train"] = transform_residual(res_scaler, y_res_train)
    samples["y_val"] = transform_residual(res_scaler, y_res_val)
    # y_test 不需要 transform，_train_and_score 只用 y_test 做评估
    return samples, res_scaler


def run_search(
    model_name: str,
    horizon: int,
    strategy: str,
    cfg: dict,
    logger,
    use_quick_subset: bool = True,
    n_folds: int = 3,
    train_frac: float = 0.7,
) -> dict:
    """对单个 (model, horizon, strategy) 运行一次混合搜索（对齐 Step4 策略）。"""
    logger.info("-" * 60)
    logger.info("模型=%s  Horizon=H%d  策略=%s  QuickSubset=%s",
                model_name, horizon, strategy, use_quick_subset)

    samples = load_samples(horizon)
    meta = load_meta(horizon)
    samples, res_scaler = prepare_residual_data(samples, horizon)

    result = run_hybrid_ablation(
        strategy=strategy,
        model_name=model_name,
        data=samples,
        meta=meta,
        horizon=horizon,
        cfg=cfg,
        use_quick_subset=use_quick_subset,
        n_folds=n_folds,
        train_frac=train_frac,
        logger=logger,
    )

    # 去掉不可序列化的字段
    best_serializable = {k: v for k, v in result["quick_best"].items() if k != "model_state"}
    full_eval_serializable = {k: v for k, v in result["full_eval"].items() if k != "best_model_state"}
    out = {
        "strategy": result["strategy"],
        "model": model_name,
        "horizon": horizon,
        "trials": result["trials"],
        "quick_best": best_serializable,
        "full_eval": full_eval_serializable,
    }
    logger.info(
        "策略 %s 完成: quick_RMSE=%.6f  3fold_val_RMSE=%.6f  3fold_TEST_RMSE=%.6f",
        strategy,
        best_serializable.get("RMSE", -1),
        full_eval_serializable.get("avg_val_RMSE", -1),
        full_eval_serializable.get("avg_test_RMSE", -1),
    )
    return out


def main():
    parser = argparse.ArgumentParser(
        description="EXP-P06 Optuna-AFSA 混合搜索（S1-S6 消融，对齐 Step4 搜索策略）"
    )
    parser.add_argument("--horizon", type=int, choices=[1, 4, 16], default=None,
                        help="指定 horizon（与 --all-horizons 互斥）")
    parser.add_argument("--model", type=str, choices=MODELS, default=None,
                        help="指定模型（默认全部 5 个模型）")
    parser.add_argument("--strategy", type=str, choices=STRATEGIES, default="S2",
                        help="指定搜索策略（默认 S2）")
    parser.add_argument("--all-horizons", action="store_true",
                        help="同时运行 H1 / H4 / H16")
    parser.add_argument("--all-strategies", action="store_true",
                        help="同时运行 S1-S6 全部策略")
    parser.add_argument("--no-quick", action="store_true",
                        help="关闭 quick subset 模式，使用完整训练数据搜索（Step4 对齐模式默认开启）")
    parser.add_argument("--n-folds", type=int, default=3,
                        help="3-fold 评估的 fold 数（默认 3）")
    parser.add_argument("--train-frac", type=float, default=0.667,
                        help="滚动窗口训练集比例（默认 0.667，与 P04 对齐）")
    args = parser.parse_args()

    if not args.all_horizons and args.horizon is None:
        parser.error("需要指定 --horizon 或使用 --all-horizons")

    cfg = load_config()
    horizons = [args.horizon] if args.horizon else [1, 4, 16]
    strategies = STRATEGIES if args.all_strategies else [args.strategy]
    models = [args.model] if args.model else MODELS
    use_quick = not args.no_quick

    logger = setup_logger("hybrid_search_p06", "EXP-P06_hybrid_search.log")

    all_results = {}
    for h in horizons:
        ensure_dirs(METRICS_DIR / f"h{h}")
        h_results = {}
        for s in strategies:
            for m in models:
                try:
                    key = f"{m}_{s}"
                    h_results[key] = run_search(
                        m, h, s, cfg, logger,
                        use_quick_subset=use_quick,
                        n_folds=args.n_folds,
                        train_frac=args.train_frac,
                    )
                except Exception as e:
                    logger.error("搜索失败 model=%s horizon=H%d strategy=%s: %s", m, h, s, e, exc_info=True)
                    h_results[key] = {"error": str(e)}

        all_results[f"h{h}"] = h_results

        out = METRICS_DIR / f"h{h}" / "hybrid_search_full.json"
        out.write_text(json.dumps(h_results, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("结果已保存: %s", out.relative_to(PROJECT_ROOT))

    logger.info("=" * 60)
    logger.info("EXP-P06 混合搜索全部完成！")
    for h in horizons:
        logger.info("  Horizon H%d: %s", h, METRICS_DIR / f"h{h}" / "hybrid_search_full.json")


if __name__ == "__main__":
    main()
