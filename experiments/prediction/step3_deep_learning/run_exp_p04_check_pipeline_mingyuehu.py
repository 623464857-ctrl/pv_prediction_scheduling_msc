"""
检查明月湖数据集 P04 流水线各步骤的审计记录与日志，用于逐步排查问题。

python -m experiments.prediction.step3_deep_learning.run_exp_p04_check_pipeline_mingyuehu
python -m experiments.prediction.step3_deep_learning.run_exp_p04_check_pipeline_mingyuehu --horizon 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.prediction.step2_hyperparameter_search.exp_p04_common import (
    LOG_DIR,
    METRICS_DIR,
    MODELS_DIR,
    PRED_DIR,
    SAMPLES_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
)

# 明月湖流水线的步骤定义
MINGYUEHU_PIPELINE_STEPS = [
    ("prepare_samples_mingyuehu", "样本准备"),
    ("optuna_mingyuehu", "Optuna搜索"),
    ("train_final_mingyuehu", "最终训练"),
    ("reproduce_mingyuehu", "多Seed复现"),
    ("report_mingyuehu", "报告生成"),
]

_CHECKS: list[dict] = []


def _check(name: str, horizon: str, passed: bool, detail: str, severity: str = "ERROR") -> None:
    status = "PASS" if passed else severity
    _CHECKS.append({"name": name, "horizon": horizon, "status": status, "detail": detail})
    tag = " PASS " if passed else f"[{severity}]"
    print(f"  {tag} [{horizon}] {name}: {detail}")


def _artifact_exists(rel_path: str) -> bool:
    return (PROJECT_ROOT / rel_path).exists()


def _load_mingyuehu_sample_dir(horizon: int, lookback: int = None) -> Path:
    """返回明月湖样本目录"""
    from experiments.prediction.step2_hyperparameter_search.exp_p04_common import load_config
    if lookback is None:
        lookback_map = {1: 16, 4: 48, 16: 96}
        lookback = lookback_map.get(horizon, 16)
    return SAMPLES_DIR / f"mingyuehu_h{horizon}_lb{lookback}"


def _load_step_record(horizon: int, step: str) -> dict | None:
    """加载审计记录"""
    record_path = PROJECT_ROOT / "logs" / "prediction" / "step2_hyperparameter_search" / "audit" / f"h{horizon}" / f"{step}.json"
    if record_path.exists():
        return json.loads(record_path.read_text(encoding="utf-8"))
    return None


def check_horizon(horizon: int) -> None:
    hs = f"h{horizon}"
    print(f"\n{'=' * 60}")
    print(f"明月湖流水线检查  horizon={horizon}")
    print(f"{'=' * 60}")

    # 样本目录检查
    lookback_map = {1: 16, 4: 48, 16: 96}
    lookback = lookback_map.get(horizon, 16)
    sample_dir = _load_mingyuehu_sample_dir(horizon, lookback)
    _check("明月湖样本目录", hs, sample_dir.exists(), str(sample_dir.relative_to(PROJECT_ROOT)))

    if sample_dir.exists():
        # 检查样本文件
        sample_files = [
            "X_train_seq.npy",
            "X_val_seq.npy",
            "X_test_seq.npy",
            "y_train.npy",
            "y_val.npy",
            "y_test.npy",
            "y_anchor_test.npy",
            "scaler_params.json",
            "meta.json",
            "test_timestamps.csv",
        ]
        for fname in sample_files:
            fpath = sample_dir / fname
            _check(f"  {fname}", hs, fpath.exists(), fpath.name)

    # 检查各步骤审计记录
    prev_ok = True
    for step, label in MINGYUEHU_PIPELINE_STEPS:
        rec = _load_step_record(horizon, step)
        if rec is None:
            _check(f"{label} 审计记录", hs, False,
                   f"缺失 audit/h{horizon}/{step}.json", "WARN" if not prev_ok else "WARN")
            prev_ok = False
            continue

        status = rec.get("status", "unknown")
        _check(f"{label} 状态", hs, status == "success", f"status={status}")
        if status != "success":
            prev_ok = False
            err = rec.get("error")
            if err:
                _check(f"{label} 错误信息", hs, False, str(err)[:200])

        # 检查日志文件
        log_file = rec.get("log_file", "")
        log_path = LOG_DIR / log_file if log_file else None
        if log_path and log_path.exists():
            size_kb = log_path.stat().st_size / 1024
            _check(f"{label} 运行日志", hs, size_kb > 0.1, f"{log_file} ({size_kb:.1f} KB)")
        else:
            _check(f"{label} 运行日志", hs, False, f"缺失 {log_file}", "WARN" if prev_ok else "ERROR")

        # 检查产出物
        for art in rec.get("artifacts", []):
            _check(f"{label} 产出", hs, _artifact_exists(art), art)

        # 步骤特定检查
        summary = rec.get("summary", {})
        if step == "prepare_samples_mingyuehu" and status == "success":
            meta_ok = _artifact_exists(f"{sample_dir.relative_to(PROJECT_ROOT)}/meta.json") if sample_dir.exists() else False
            _check(f"{label} meta.json", hs, meta_ok, str(sample_dir.relative_to(PROJECT_ROOT)) if sample_dir.exists() else "无sample_dir")

        if step == "optuna_mingyuehu" and status == "success":
            optuna_rel = f"data/prediction/step2_hyperparameter_search/metrics/mingyuehu_h{horizon}/mingyuehu_cnn_bilstm_optuna.json"
            _check(f"{label} optuna.json", hs, _artifact_exists(optuna_rel), optuna_rel)
            ablation_rel = f"data/prediction/step2_hyperparameter_search/metrics/mingyuehu_h{horizon}/mingyuehu_hybrid_search_ablation.json"
            _check(f"{label} ablation.json", hs, _artifact_exists(ablation_rel), ablation_rel)

        if step == "train_final_mingyuehu" and status == "success":
            model_rel = f"data/prediction/step3_deep_learning/models/mingyuehu_h{horizon}/mingyuehu_cnn_bilstm_final.pt"
            _check(f"{label} 模型文件", hs, _artifact_exists(model_rel), model_rel)
            metrics_rel = f"data/prediction/step4_evaluation/metrics/mingyuehu_h{horizon}/mingyuehu_cnn_bilstm_test_metrics.json"
            _check(f"{label} 测试指标", hs, _artifact_exists(metrics_rel), metrics_rel)

        if step == "reproduce_mingyuehu" and status == "success":
            rep_rel = f"data/prediction/step4_evaluation/metrics/mingyuehu_h{horizon}/mingyuehu_cnn_bilstm_reproduce.json"
            _check(f"{label} reproduce.json", hs, _artifact_exists(rep_rel), rep_rel)
            if _artifact_exists(rep_rel):
                rep = json.loads((PROJECT_ROOT / rep_rel).read_text(encoding="utf-8"))
                expected_seeds = [42, 43, 44, 45, 46]
                got = rep.get("seeds", [])
                _check(f"{label} seed 列表", hs, got == expected_seeds, f"expected={expected_seeds} got={got}")

        if step == "report_mingyuehu" and status == "success":
            report_rel = f"data/prediction/step5_reporting/reports/EXP-P04_mingyuehu_h{horizon}_详细实验汇报.md"
            _check(f"{label} Markdown", hs, _artifact_exists(report_rel), report_rel)


def write_report(horizons: list[int]) -> Path:
    out = PROJECT_ROOT / "data" / "prediction" / "step2_hyperparameter_search" / "audit" / "mingyuehu_pipeline_check_report.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"明月湖数据集 EXP-P04 Pipeline Check Report", f"horizons={horizons}", ""]
    for c in _CHECKS:
        lines.append(f"[{c['status']:>6}] [{c['horizon']}] {c['name']}: {c['detail']}")
    n_fail = sum(1 for c in _CHECKS if c["status"] != "PASS")
    lines.append("")
    lines.append(f"Total checks: {len(_CHECKS)}  FAIL/WARN: {n_fail}")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main():
    parser = argparse.ArgumentParser(description="明月湖 EXP-P04 流水线分步审计检查")
    parser.add_argument("--horizon", type=int, choices=[1, 4, 16], default=None,
                        help="指定 horizon；默认检查 h1/h4/h16")
    args = parser.parse_args()

    horizons = [args.horizon] if args.horizon else [1, 4, 16]
    for h in horizons:
        check_horizon(h)

    report_path = write_report(horizons)
    n_fail = sum(1 for c in _CHECKS if c["status"] != "PASS")
    print(f"\n{'=' * 60}")
    print(f"检查完成: {len(_CHECKS)} 项, 未通过 {n_fail} 项")
    print(f"报告: {report_path.relative_to(PROJECT_ROOT)}")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
