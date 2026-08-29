"""EXP-P04 分步实验审计：每步运行后写入结构化记录，供后续排查。"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.prediction.step4_optuna_hybrid.exp_p04_common import (
    LOG_DIR,
    PROJECT_ROOT,
    STEP4_ROOT,
)

AUDIT_DIR = STEP4_ROOT / "audit"

PIPELINE_STEPS: list[tuple[str, str]] = [
    ("prepare_samples", "Step1 样本构造"),
    ("optuna", "Step2 混合搜索"),
    ("train_final", "Step3 最终训练"),
    ("reproduce", "Step4 多Seed复现"),
    ("report", "Step5 报告生成"),
]

STEP_SCRIPT = {
    "prepare_samples": "run_exp_p04_prepare_samples.py",
    "optuna": "run_exp_p04_optuna.py",
    "train_final": "run_exp_p04_train_final.py",
    "reproduce": "run_exp_p04_reproduce.py",
    "report": "run_exp_p04_report.py",
}


def _audit_dir(horizon: int) -> Path:
    d = AUDIT_DIR / f"h{horizon}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %z")


STEP_LOG_SUFFIX = {
    "prepare_samples": "_prepare_samples.log",
    "optuna": "_optuna.log",
    "train_final": "_final_train.log",
    "reproduce": "_reproduce.log",
    "report": "_report.log",
}


def parse_horizon_from_argv(default: int = 0) -> int:
    """从命令行 --horizon 解析 horizon，失败回调时用于写入正确审计路径。"""
    import sys

    if "--horizon" in sys.argv:
        idx = sys.argv.index("--horizon")
        if idx + 1 < len(sys.argv):
            try:
                return int(sys.argv[idx + 1])
            except ValueError:
                pass
    return default


def default_log_file(step: str, horizon: int = 0, *, lookback: int = 16) -> str:
    """按步骤与 horizon 推断默认日志文件名。"""
    if not horizon:
        return f"EXP-P04_{step}.log"
    if step == "prepare_samples":
        return f"EXP-P04_h{horizon}_lb{lookback}_prepare_samples.log"
    suffix = STEP_LOG_SUFFIX.get(step, ".log")
    return f"EXP-P04_h{horizon}{suffix}"


def record_step_failure(
    step: str,
    t0: float,
    error: Exception | str,
    *,
    horizon: int | None = None,
    log_file: str | None = None,
) -> Path:
    """步骤异常退出时写入 failed 审计记录。"""
    h = horizon if horizon is not None else parse_horizon_from_argv(0)
    lf = log_file or default_log_file(step, h)
    return record_step_result(
        h, step, "failed", lf,
        summary={}, duration_sec=time.time() - t0, error=str(error),
    )


def record_step_result(
    horizon: int,
    step: str,
    status: str,
    log_file: str,
    summary: dict[str, Any] | None = None,
    *,
    duration_sec: float | None = None,
    error: str | None = None,
    artifacts: list[str] | None = None,
) -> Path:
    """写入单步审计 JSON、更新 manifest、追加流水线日志。"""
    if step not in dict(PIPELINE_STEPS):
        raise ValueError(f"未知步骤: {step}")

    summary = summary or {}
    artifacts = artifacts or []
    record = {
        "step": step,
        "step_label": dict(PIPELINE_STEPS)[step],
        "horizon": horizon,
        "status": status,
        "finished_at": _now_iso(),
        "duration_sec": round(duration_sec, 2) if duration_sec is not None else None,
        "log_file": log_file,
        "log_path": str((LOG_DIR / log_file).relative_to(PROJECT_ROOT)),
        "script": STEP_SCRIPT[step],
        "summary": summary,
        "artifacts": artifacts,
        "error": error,
    }

    step_path = _audit_dir(horizon) / f"{step}.json"
    step_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest_path = _audit_dir(horizon) / "pipeline_manifest.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[step] = {
        "status": status,
        "finished_at": record["finished_at"],
        "duration_sec": record["duration_sec"],
        "log_file": log_file,
        "audit_file": str(step_path.relative_to(PROJECT_ROOT)),
    }
    if status == "success" and summary:
        manifest[step]["summary_keys"] = list(summary.keys())
    if error:
        manifest[step]["error"] = error
    manifest["_updated_at"] = record["finished_at"]
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    audit_log = LOG_DIR / f"PIPELINE_AUDIT_h{horizon}.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "=" * 72,
        f"[{record['finished_at']}] {record['step_label']} ({step})  status={status}",
        f"  horizon=h{horizon}  duration={record['duration_sec']}s",
        f"  log: logs/prediction/step4_optuna_hybrid/{log_file}",
        f"  audit: {step_path.relative_to(PROJECT_ROOT)}",
    ]
    for key, val in summary.items():
        lines.append(f"  {key}: {val}")
    if artifacts:
        lines.append("  artifacts:")
        for art in artifacts:
            lines.append(f"    - {art}")
    if error:
        lines.append(f"  ERROR: {error}")
    with open(audit_log, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return step_path


def load_step_record(horizon: int, step: str) -> dict | None:
    path = _audit_dir(horizon) / f"{step}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifest(horizon: int) -> dict:
    path = _audit_dir(horizon) / "pipeline_manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
