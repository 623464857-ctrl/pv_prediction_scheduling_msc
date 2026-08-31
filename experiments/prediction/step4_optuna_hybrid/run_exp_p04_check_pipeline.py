"""
检查 P04 流水线各步骤的审计记录与日志，用于逐步排查问题。

python -m experiments.prediction.step4_optuna_hybrid.run_exp_p04_check_pipeline
python -m experiments.prediction.step4_optuna_hybrid.run_exp_p04_check_pipeline --horizon 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.prediction.step4_optuna_hybrid.exp_p04_common import (
    LOG_DIR,
    METRICS_DIR,
    MODELS_DIR,
    PRED_DIR,
    REPORTS_DIR,
    load_config,
    load_sample_dir,
)
from experiments.prediction.step4_optuna_hybrid.exp_p04_step_audit import (
    PIPELINE_STEPS,
    load_manifest,
    load_step_record,
)

_CHECKS: list[dict] = []


def _check(name: str, horizon: str, passed: bool, detail: str, severity: str = "ERROR") -> None:
    status = "PASS" if passed else severity
    _CHECKS.append({"name": name, "horizon": horizon, "status": status, "detail": detail})
    tag = " PASS " if passed else f"[{severity}]"
    print(f"  {tag} [{horizon}] {name}: {detail}")


def _artifact_exists(rel_path: str) -> bool:
    return (PROJECT_ROOT / rel_path).exists()


def check_horizon(horizon: int) -> None:
    hs = f"h{horizon}"
    print(f"\n{'=' * 60}")
    print(f"Pipeline 检查  horizon={horizon}")
    print(f"{'=' * 60}")

    manifest = load_manifest(horizon)
    if not manifest:
        _check("manifest 存在", hs, False, "未找到 audit/pipeline_manifest.json，尚未运行任何步骤")
        return

    audit_log = LOG_DIR / f"PIPELINE_AUDIT_h{horizon}.log"
    _check("流水线审计日志", hs, audit_log.exists(),
           str(audit_log.relative_to(PROJECT_ROOT)) if audit_log.exists() else "缺失")

    prev_ok = True
    for step, label in PIPELINE_STEPS:
        rec = load_step_record(horizon, step)
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
                _check(f"{label} 错误信息", hs, False, err[:200])

        log_file = rec.get("log_file", "")
        log_path = LOG_DIR / log_file if log_file else None
        if log_path and log_path.exists():
            size_kb = log_path.stat().st_size / 1024
            _check(f"{label} 运行日志", hs, size_kb > 0.1, f"{log_file} ({size_kb:.1f} KB)")
        else:
            _check(f"{label} 运行日志", hs, False, f"缺失 {log_file}")
            prev_ok = False

        for art in rec.get("artifacts", []):
            _check(f"{label} 产出", hs, _artifact_exists(art), art)

        summary = rec.get("summary", {})
        if step == "prepare_samples" and status == "success":
            sample_dir = summary.get("sample_dir", "")
            meta_ok = _artifact_exists(f"{sample_dir}/meta.json") if sample_dir else False
            _check(f"{label} meta.json", hs, meta_ok, sample_dir or "无 sample_dir")

        if step == "optuna" and status == "success":
            optuna_rel = f"data/prediction/step4_optuna_hybrid/metrics/{hs}/cnn_bilstm_optuna.json"
            _check(f"{label} optuna.json", hs, _artifact_exists(optuna_rel), optuna_rel)

        if step == "train_final" and status == "success":
            model_rel = f"data/prediction/step4_optuna_hybrid/models/{hs}/cnn_bilstm_final.pt"
            _check(f"{label} 模型文件", hs, _artifact_exists(model_rel), model_rel)

        if step == "reproduce" and status == "success":
            rep_rel = f"data/prediction/step4_optuna_hybrid/metrics/{hs}/cnn_bilstm_reproduce.json"
            _check(f"{label} reproduce.json", hs, _artifact_exists(rep_rel), rep_rel)
            seeds = load_config("exp_p04_base.json")["reproduce_seeds"]
            rep = json.loads((PROJECT_ROOT / rep_rel).read_text(encoding="utf-8")) if _artifact_exists(rep_rel) else {}
            got = rep.get("seeds", [])
            _check(f"{label} seed 列表", hs, got == seeds, f"expected={seeds} got={got}")

        if step == "report" and status == "success":
            report_rel = f"data/prediction/step4_optuna_hybrid/reports/EXP-P04_h{horizon}_详细实验汇报.md"
            _check(f"{label} Markdown", hs, _artifact_exists(report_rel), report_rel)

        if prev_ok and step != PIPELINE_STEPS[-1][0]:
            pass  # 顺序检查：下一步缺失时上条 WARN 已提示

    # 样本目录一致性
    try:
        sdir = load_sample_dir(horizon)
        _check("样本目录可解析", hs, sdir.exists(), str(sdir.relative_to(PROJECT_ROOT)))
    except Exception as e:
        _check("样本目录可解析", hs, False, str(e))


def write_report(horizons: list[int]) -> Path:
    out = PROJECT_ROOT / "data/prediction/step4_optuna_hybrid/audit/pipeline_check_report.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"EXP-P04 Pipeline Check Report", f"horizons={horizons}", ""]
    for c in _CHECKS:
        lines.append(f"[{c['status']:>6}] [{c['horizon']}] {c['name']}: {c['detail']}")
    n_fail = sum(1 for c in _CHECKS if c["status"] != "PASS")
    lines.append("")
    lines.append(f"Total checks: {len(_CHECKS)}  FAIL/WARN: {n_fail}")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main():
    parser = argparse.ArgumentParser(description="EXP-P04 流水线分步审计检查")
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
