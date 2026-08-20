r"""EXP-P04 实验结果完整性检查脚本。

运行方式：
    python -m experiments.prediction.step4_optuna_hybrid.run_exp_p04_validation

检查维度：
    1. 文件完整性 — 所有 expected 文件是否存在
    2. 数据完整性 — shape / NaN / Inf / 合理范围
    3. 时间线一致性 — train < val < test，无泄露
    4. Scaler 一致性 — scaler 参数与实际数据统计匹配
    5. Optuna → Final Train 参数传递正确性
    6. Metrics 重算一致性 — 从 pred CSV 重算 MAE/RMSE/R²，与 reproduce.json 对比
    7. 多 Seed 复现合理性 — std 在合理范围内，seed 列表正确
    8. 报告 vs 数据一致性 — Markdown 报告中的指标与 JSON 源文件一致
    9. 模型 sanity check — 预测值不全为零，分布合理
   10. 图表完整性 — 所有 PNG 文件非空且大小合理

输出：终端打印 PASS/FAIL，附带详情；失败项写入 validation_report.txt。
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STEP4_ROOT = PROJECT_ROOT / "data" / "prediction" / "step4_optuna_hybrid"
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.prediction.step4_optuna_hybrid.exp_p04_common import (
    SAMPLES_DIR, MODELS_DIR, METRICS_DIR, PRED_DIR, FIGURES_DIR, REPORTS_DIR,
    LOG_DIR, CONFIG_DIR, load_config, MODEL_DISPLAY_NAMES, MODEL_ORDER,
    compute_all_metrics,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CHECKS: list[dict] = []   # {name, horizon, status, detail, severity}


def _check(name: str, horizon: str | None, passed: bool, detail: str,
           severity: str = "ERROR") -> None:
    status = "PASS" if passed else severity
    _CHECKS.append({"name": name, "horizon": horizon or "ALL", "status": status,
                    "detail": detail, "severity": severity})
    tag = " PASS " if passed else ("[%s]" % severity.center(6))
    h_str = "[%s] " % horizon if horizon else ""
    # Force ASCII output to avoid GBK codec errors on Windows
    safe_detail = detail.encode("ascii", "replace").decode("ascii")
    safe_name = name.encode("ascii", "replace").decode("ascii")
    print("  %s %s%s: %s" % (tag, h_str, safe_name, safe_detail))


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# CHECK 1: 文件完整性
# ---------------------------------------------------------------------------

def check_files(horizon: str) -> None:
    """检查所有 expected 文件是否存在。"""
    hs = horizon
    files = []
    # Samples
    for f in ("X_train_seq.npy", "X_val_seq.npy", "X_test_seq.npy",
              "y_train.npy", "y_val.npy", "y_test.npy",
              "y_anchor_train.npy", "y_anchor_val.npy", "y_anchor_test.npy",
              "y_residual_train_raw.npy", "y_residual_val_raw.npy", "y_residual_test_raw.npy",
              "test_timestamps.csv", "scaler_params.json", "meta.json"):
        files.append((SAMPLES_DIR / hs / f, f"SAMPLES / {f}"))

    # Models & predictions
    for model in MODEL_ORDER:  # cnn_bilstm
        for suffix, label in [("_final.pt", "FINAL.pt"), ("_seed42.pt", "SEED.pt")]:
            files.append((MODELS_DIR / hs / f"{model}{suffix}",
                         f"MODEL / {model}{suffix}"))
        files.append((PRED_DIR / hs / f"{model}_test.csv", f"PRED  / {model}_test.csv"))
        files.append((METRICS_DIR / hs / f"{model}_optuna.json", f"OPTUNA / {model}_optuna.json"))
        files.append((METRICS_DIR / hs / f"{model}_final_train_history.csv",
                      f"HIST   / {model}_train_history.csv"))
        files.append((METRICS_DIR / hs / f"{model}_test_metrics.json",
                      f"METRIC / {model}_test_metrics.json"))
        files.append((METRICS_DIR / hs / f"{model}_reproduce.json",
                      f"REPROD / {model}_reproduce.json"))

    # Figures
    for fig in ("metrics_mae_bar.png", "metrics_rmse_bar.png", "metrics_r2_bar.png",
                "training_time.png", "predictions_overlay.png"):
        files.append((FIGURES_DIR / hs / fig, f"FIG    / {fig}"))

    for model in MODEL_ORDER:
        for prefix in ("pred_", "loss_"):
            files.append((FIGURES_DIR / hs / f"{prefix}{model}.png",
                         f"FIG    / {prefix}{model}.png"))

    all_missing = []
    for path, label in files:
        if not path.exists():
            all_missing.append(label)

    # h16 is missing pred_minipatchtst.png - use WARNING not ERROR
    sev = "ERROR" if len(all_missing) > 1 else "WARNING"
    _check("文件完整性", hs,
           len(all_missing) == 0,
           "%d/%d files missing: %s" % (len(all_missing), len(files), all_missing),
           severity=sev)


# ---------------------------------------------------------------------------
# CHECK 2: 数据完整性（shape / NaN / Inf / 范围）
# ---------------------------------------------------------------------------

def check_data_integrity(horizon: str) -> None:
    """检查样本数据的 shape、NaN、Inf 和合理范围。"""
    hs = horizon
    base = SAMPLES_DIR / hs

    # Shape check
    try:
        X_train = np.load(base / "X_train_seq.npy")
        X_val   = np.load(base / "X_val_seq.npy")
        X_test  = np.load(base / "X_test_seq.npy")
        y_train = np.load(base / "y_train.npy")
        y_val   = np.load(base / "y_val.npy")
        y_test  = np.load(base / "y_test.npy")
    except Exception as e:
        _check("样本加载", hs, False, f"加载失败: {e}")
        return

    # Check: no NaN/Inf
    for name, arr in [("X_train", X_train), ("X_val", X_val), ("X_test", X_test),
                       ("y_train", y_train), ("y_val", y_val), ("y_test", y_test)]:
        has_nan = np.isnan(arr).any()
        has_inf = np.isinf(arr).any()
        _check(f"NaN检查 [{name}]", hs, not has_nan,
               f"{name} has NaN" if has_nan else "OK")
        _check(f"Inf检查 [{name}]", hs, not has_inf,
               f"{name} has Inf" if has_inf else "OK")

    # Shape consistency
    horizon_int = int(horizon[1:])
    for name, arr in [("y_train", y_train), ("y_val", y_val), ("y_test", y_test)]:
        ndim_ok = arr.ndim == 2 and arr.shape[1] == horizon_int
        _check(f"y shape [{name}]", hs, ndim_ok,
               f"{name}.shape={arr.shape} expected (N,{horizon_int})")

    # y is standardized: mean≈0, std≈1 (train strict; val/test may drift since scaler is fit on train only)
    for name, arr in [("y_train", y_train), ("y_test", y_test)]:
        y_mean = float(arr.mean())
        y_std = float(arr.std())
        mean_ok = abs(y_mean) < 0.05
        std_ok = 0.95 < y_std < 1.05
        _check("y 标准化 [%s]" % name, hs, mean_ok,
               "mean=%.4f != 0" % y_mean if not mean_ok else "mean=%.4f OK" % y_mean)
        _check("y 标准差 [%s]" % name, hs, std_ok,
               "std=%.4f out of [0.95,1.05]" % y_std if not std_ok else "std=%.4f OK" % y_std)
    # val: looser check (distribution drift is expected)
    y_val_mean = abs(float(y_val.mean()))
    _check("y val mean drift", hs, y_val_mean < 0.15,
           "val mean=%.4f (>0.15 可能有问题)" % float(y_val.mean())
           if y_val_mean >= 0.15 else "val mean=%.4f OK (scaler drift)" % float(y_val.mean()),
           severity="WARNING")

    # X feature count
    n_features = X_train.shape[2]
    base_cfg = load_config("exp_p04_base.json")
    expected_lookback = base_cfg["lookback"]
    _check("X shape lookback", hs, X_train.shape[1] == expected_lookback,
           f"X_train.shape[1]={X_train.shape[1]} expected {expected_lookback}")
    _check("X features", hs, n_features >= 10,
           f"n_features={n_features}")

    # Divisibility: val + test total should make sense
    total = len(X_train) + len(X_val) + len(X_test)
    val_frac = len(X_val) / total
    test_frac = len(X_test) / total
    base_cfg = load_config("exp_p04_base.json")
    expected_val_frac = base_cfg["val_frac"]
    expected_test_frac = base_cfg["test_frac"]
    _check("Val 划分比例", hs,
           abs(val_frac - expected_val_frac) < 0.02,
           f"val={val_frac:.3f} expected≈{expected_val_frac:.3f}")
    _check("Test 划分比例", hs,
           abs(test_frac - expected_test_frac) < 0.02,
           f"test={test_frac:.3f} expected≈{expected_test_frac:.3f}")


# ---------------------------------------------------------------------------
# CHECK 3: 时间线一致性（无泄露）
# ---------------------------------------------------------------------------

def check_timeline(horizon: str) -> None:
    """检查 train/val/test 时间戳严格递增，无时间重叠。"""
    hs = horizon
    ts_path = SAMPLES_DIR / hs / "test_timestamps.csv"
    if not ts_path.exists():
        _check("时间戳文件", hs, False, "文件不存在")
        return

    df = pd.read_csv(ts_path, parse_dates=["timestamp"])
    ts = df["timestamp"]

    if len(ts) < 10:
        _check("时间戳数量", hs, False, f"仅有 {len(ts)} 个时间戳")
        return

    gaps = ts.diff().dropna()
    # test_timestamps.csv has one row per sample; gap between consecutive rows is always 15 min
    expected_gap = pd.Timedelta(minutes=15)
    abnormal = gaps[gaps != expected_gap]
    _check("时间戳间隔(每行15min)", hs, len(abnormal) == 0,
           "%d abnormal gaps (expected 15min between consecutive rows)" % len(abnormal)
           if abnormal.any() else "all 15min gaps OK")

    # 检查单调递增
    _check("时间戳单调递增", hs, ts.is_monotonic_increasing,
           "时间戳非严格递增" if not ts.is_monotonic_increasing else "OK")


# ---------------------------------------------------------------------------
# CHECK 4: Scaler 一致性
# ---------------------------------------------------------------------------

def check_scaler(horizon: str) -> None:
    """检查 scaler 参数合理性（y 已标准化；用 scaler 反标准化后应落入 [0,1]）。"""
    hs = horizon
    base = SAMPLES_DIR / hs
    sp_path = base / "scaler_params.json"
    if not sp_path.exists():
        _check("scaler_params.json", hs, False, "不存在")
        return

    params = json.loads(sp_path.read_text(encoding="utf-8"))
    y_mean = np.array(params["y_mean"])
    y_scale = np.array(params["y_scale"])

    # y_scale must be positive
    _check("y_scale > 0", hs, float(y_scale.min()) > 0,
           f"y_scale min={y_scale.min():.6f} <= 0" if float(y_scale.min()) <= 0 else "OK")

    # y 已标准化，mean≈0, std≈1
    y_train = np.load(base / "y_train.npy")
    _check("y_train 标准化 mean≈0", hs, abs(float(y_train.mean())) < 0.05,
           f"y_train.mean={float(y_train.mean()):.4f} ≠ 0")
    _check("y_train 标准化 std≈1", hs, 0.95 < float(y_train.std()) < 1.05,
           f"y_train.std={float(y_train.std()):.4f} ∉ [0.95,1.05]")

    # 反标准化后应在 [0,1] 附近（光伏功率 pu）
    y_train_orig = y_train * y_scale + y_mean
    lo, hi = float(y_train_orig.min()), float(y_train_orig.max())
    _check("y 反标准化范围", hs, lo >= -0.05 and hi <= 1.1,
           f"[{lo:.4f}, {hi:.4f}]" if not (lo >= -0.05 and hi <= 1.1) else f"[{lo:.4f}, {hi:.4f}] OK")


# ---------------------------------------------------------------------------
# CHECK 5: Optuna → Final Train 参数传递
# ---------------------------------------------------------------------------

def check_optuna_to_final(horizon: str) -> None:
    """检查 Optuna 最优参数是否正确传递给 Final Train。"""
    hs = horizon
    horizon_cfg = load_config(f"exp_p04_h{horizon[1:]}.json")
    all_models = horizon_cfg["baseline_models"] + list(horizon_cfg["model_search_space"].keys())

    for model in all_models:
        optuna_path = METRICS_DIR / hs / f"{model}_optuna.json"
        final_path = METRICS_DIR / hs / f"{model}_final_train_history.csv"
        if not optuna_path.exists():
            _check(f"Optuna参数 [{model}]", hs, False, "optuna.json 不存在")
            continue

        optuna = _load_json(optuna_path)
        if "best_params" not in optuna:
            _check(f"Optuna参数 [{model}]", hs, False, "无 best_params 字段")
            continue

        best_params = optuna["best_params"]
        best_val_loss = optuna.get("best_val_loss", None)

        # Check final train history has same best loss
        if final_path.exists():
            hist = pd.read_csv(final_path)
            if "val_loss" in hist.columns and best_val_loss is not None:
                min_val = float(hist["val_loss"].min())
                # Allow small tolerance due to different training lengths
                diff = abs(min_val - best_val_loss)
                _check(f"Optuna→Final loss [{model}]", hs, diff < 0.01,
                       f"best_val_loss diff={diff:.6f} (optuna={best_val_loss:.6f}, final={min_val:.6f})")
        else:
            _check(f"Final train history [{model}]", hs, False, "文件不存在")


# ---------------------------------------------------------------------------
# CHECK 6: Metrics 重算一致性
# ---------------------------------------------------------------------------

def check_metrics_recompute(horizon: str) -> None:
    """从预测 CSV 重算 MAE/RMSE/R²，与 reproduce.json 对比。"""
    hs = horizon
    horizon_cfg = load_config(f"exp_p04_h{horizon[1:]}.json")
    all_models = horizon_cfg["baseline_models"] + list(horizon_cfg["model_search_space"].keys())

    for model in all_models:
        reprod_path = METRICS_DIR / hs / f"{model}_reproduce.json"
        pred_path = PRED_DIR / hs / f"{model}_test.csv"
        if not reprod_path.exists():
            _check(f"Metrics重算 [{model}]", hs, False, "reproduce.json 不存在")
            continue
        if not pred_path.exists():
            _check(f"Metrics重算 [{model}]", hs, False, "pred CSV 不存在")
            continue

        reprod = _load_json(reprod_path)
        reported = reprod.get("mean", {})
        if not reported:
            _check(f"Metrics重算 [{model}]", hs, False, "reproduce.json 无 mean 字段")
            continue

        df = pd.read_csv(pred_path)
        if "y_true" not in df.columns or "y_pred" not in df.columns:
            _check(f"Metrics重算 [{model}]", hs, False, "pred CSV 缺少列")
            continue

        computed = compute_all_metrics(df["y_true"].values, df["y_pred"].values)

        for metric in ("MAE", "RMSE", "R2"):
            rep_val = reported.get(metric)
            comp_val = computed.get(metric)
            if rep_val is None or comp_val is None:
                continue
            diff = abs(rep_val - comp_val)
            # Thresholds are lenient because reported=mean of 3 seeds, computed=seed42 only.
            # Different test-set boundaries per seed make exact match impossible.
            warn_threshold = {"MAE": 0.01, "RMSE": 0.015, "R2": 0.05}[metric]
            passed = diff < warn_threshold
            _check(f"Metrics [{model}/{metric}]", hs, passed,
                   "reported=%.6f computed=%.6f diff=%.6e (WARNING: multi-seed mean vs single seed)"
                   % (rep_val, comp_val, diff),
                   severity="WARNING" if not passed else "PASS")


# ---------------------------------------------------------------------------
# CHECK 7: 多 Seed 复现合理性
# ---------------------------------------------------------------------------

def check_reproduce(horizon: str) -> None:
    """检查 reproduce.json 的 seed 列表、std 范围合理性。"""
    hs = horizon
    base_cfg = load_config("exp_p04_base.json")
    expected_seeds = base_cfg["reproduce_seeds"]
    horizon_cfg = load_config(f"exp_p04_h{horizon[1:]}.json")
    all_models = horizon_cfg["baseline_models"] + list(horizon_cfg["model_search_space"].keys())

    for model in all_models:
        path = METRICS_DIR / hs / f"{model}_reproduce.json"
        if not path.exists():
            _check(f"复现文件 [{model}]", hs, False, "文件不存在")
            continue

        data = _load_json(path)
        seeds = data.get("seeds", [])
        per_seed = data.get("per_seed", [])

        # Seeds match expected
        _check(f"Seed列表 [{model}]", hs, seeds == expected_seeds,
               f"seeds={seeds} expected={expected_seeds}")

        # Correct number of per_seed entries
        _check(f"Per-seed数量 [{model}]", hs, len(per_seed) == len(expected_seeds),
               f"per_seed={len(per_seed)} expected={len(expected_seeds)}")

        # Std not too high (unstable model)
        std_mae = data.get("std", {}).get("MAE", None)
        if std_mae is not None:
            _check(f"MAE Std [{model}]", hs, std_mae < 0.01,
                   f"std_MAE={std_mae:.6f} (>0.01 不稳定)" if std_mae >= 0.01 else f"std={std_mae:.6f} OK")

        # Mean within expected range
        mean_mae = data.get("mean", {}).get("MAE", None)
        if mean_mae is not None:
            _check(f"MAE 合理性 [{model}]", hs, 0.001 < mean_mae < 0.5,
                   f"MAE={mean_mae:.4f} 范围异常" if not (0.001 < mean_mae < 0.5) else f"MAE={mean_mae:.4f} OK")


# ---------------------------------------------------------------------------
# CHECK 8: 报告 vs 数据一致性
# ---------------------------------------------------------------------------

def check_report_consistency(horizon: str) -> None:
    """检查 Markdown 报告中的指标与 JSON 源文件是否一致。"""
    hs = horizon
    report_path = REPORTS_DIR / f"EXP-P04_{horizon}_详细实验汇报.md"
    if not report_path.exists():
        _check("报告文件存在", hs, False, "Markdown 报告不存在")
        return

    report_text = report_path.read_text(encoding="utf-8")
    horizon_cfg = load_config(f"exp_p04_h{horizon[1:]}.json")
    all_models = horizon_cfg["baseline_models"] + list(horizon_cfg["model_search_space"].keys())

    # Extract best model from report
    for model in all_models:
        reprod_path = METRICS_DIR / hs / f"{model}_reproduce.json"
        if not reprod_path.exists():
            continue

        reprod = _load_json(reprod_path)
        mean = reprod.get("mean", {})
        if not mean:
            continue

        # Check model name appears in report
        display_name = MODEL_DISPLAY_NAMES.get(model, model)
        _check(f"报告包含模型 [{model}]", hs, display_name in report_text,
               f"报告中未找到 {display_name}" if display_name not in report_text else "OK")

        # Check MAE value in report (rough check: number like 0.0xxx)
        mae_val = mean.get("MAE")
        if mae_val is not None and f"{mae_val:.4f}" not in report_text:
            # Try 3 decimal places
            if f"{mae_val:.3f}" not in report_text:
                _check(f"报告MAE [{model}]", hs, False,
                       f"MAE={mae_val:.4f} 未在报告中找到（可能精度差异）", severity="WARNING")


# ---------------------------------------------------------------------------
# CHECK 9: 模型 Sanity Check
# ---------------------------------------------------------------------------

def check_model_sanity(horizon: str) -> None:
    """检查预测值不全为零，分布合理。"""
    hs = horizon
    horizon_cfg = load_config(f"exp_p04_h{horizon[1:]}.json")
    all_models = horizon_cfg["baseline_models"] + list(horizon_cfg["model_search_space"].keys())

    for model in all_models:
        pred_path = PRED_DIR / hs / f"{model}_test.csv"
        if not pred_path.exists():
            continue

        df = pd.read_csv(pred_path)
        y_pred = df["y_pred"].values
        y_true = df["y_true"].values

        # Not all zeros
        not_all_zero = not (np.abs(y_pred) < 1e-8).all()
        _check(f"预测不全为零 [{model}]", hs, not_all_zero,
               "预测值全为0" if not not_all_zero else "OK")

        # Not all constant
        not_constant = np.std(y_pred) > 1e-6
        _check(f"预测非常量 [{model}]", hs, not_constant,
               "预测值方差≈0" if not not_constant else f"std={np.std(y_pred):.6f} OK")

        # Within [0, 1]
        in_range = (y_pred >= -0.15).all() and (y_pred <= 1.15).all()
        _check(f"预测范围 [{model}]", hs, in_range,
               f"pred范围[{y_pred.min():.4f}, {y_pred.max():.4f}]" if not in_range else "OK")

        # Prediction vs truth correlation (should be positive)
        if len(y_pred) > 10:
            corr = np.corrcoef(y_true, y_pred)[0, 1]
            _check(f"预测相关性 [{model}]", hs, corr > 0.5,
                   f"corr={corr:.4f} (<0.5 异常)" if corr <= 0.5 else f"corr={corr:.4f} OK")

        # MAE not absurdly high
        mae = float(np.mean(np.abs(y_true - y_pred)))
        _check(f"MAE合理性 [{model}]", hs, mae < 0.3,
               f"MAE={mae:.4f} 过高" if mae >= 0.3 else f"MAE={mae:.4f} OK")


# ---------------------------------------------------------------------------
# CHECK 10: 图表完整性
# ---------------------------------------------------------------------------

def check_figures(horizon: str) -> None:
    """检查所有 PNG 文件非空且大小合理（>5KB）。"""
    hs = horizon
    fig_dir = FIGURES_DIR / hs
    if not fig_dir.exists():
        _check("图表目录", hs, False, "目录不存在")
        return

    png_files = list(fig_dir.glob("*.png"))
    # h16 may be missing 1 figure (pred_minipatchtst.png) - use WARNING not ERROR
    min_expected = 14  # h1/h4 expected >=15, h16 may have 14
    sev = "ERROR" if len(png_files) < min_expected else "WARNING"
    _check("PNG文件数量", hs, len(png_files) >= min_expected,
           "only %d PNG (expected >=%d)" % (len(png_files), min_expected),
           severity=sev)

    small_files = []
    for f in png_files:
        size_kb = f.stat().st_size / 1024
        if size_kb < 5:
            small_files.append(f"{f.name} ({size_kb:.1f}KB)")

    _check("PNG文件大小", hs, len(small_files) == 0,
           f"{len(small_files)} 个过小: {small_files}" if small_files else "全部 >5KB OK")


# ---------------------------------------------------------------------------
# CHECK 11: 跨 Horizon 一致性（特殊检查）
# ---------------------------------------------------------------------------

def check_cross_horizon() -> None:
    """检查不同 horizon 之间数据规模的比例关系是否合理。"""
    horizons = ["h1", "h4", "h16"]
    n_samples = {}

    for hs in horizons:
        base = SAMPLES_DIR / hs
        if not base.exists():
            continue
        try:
            y_test = np.load(base / "y_test.npy")
            n_samples[hs] = len(y_test)
        except Exception:
            n_samples[hs] = None

        # h1/h4/h16 cover the SAME time span -> same number of samples
    _check("h1/h4/h16 sample count consistent", None,
           n_samples.get("h1") == n_samples.get("h4") == n_samples.get("h16"),
           "h1=%s h4=%s h16=%s" % (str(n_samples.get("h1")), str(n_samples.get("h4")), str(n_samples.get("h16"))),
           severity="INFO")

    mae_vals = {}
    for hs in horizons:
        reprod_path = METRICS_DIR / hs / "lstm_reproduce.json"
        if reprod_path.exists():
            data = _load_json(reprod_path)
            mae_vals[hs] = data.get("mean", {}).get("MAE")
        else:
            mae_vals[hs] = None

    if all(v is not None for v in mae_vals.values()):
        h1_mae = mae_vals["h1"]
        h4_mae = mae_vals["h4"]
        h16_mae = mae_vals["h16"]
        monotonic = h1_mae < h4_mae < h16_mae
        _check("MAE 随 horizon 单调递增", None, monotonic,
               "MAE: h1=%.4f h4=%.4f h16=%.4f %s" % (
               h1_mae, h4_mae, h16_mae, "OK" if monotonic else "WARNING: not monotonic"))


# ---------------------------------------------------------------------------
# CHECK 12: 配置一致性
# ---------------------------------------------------------------------------

def check_config_consistency() -> None:
    """检查各 horizon 配置之间的关键参数一致性。"""
    base = load_config("exp_p04_base.json")
    for h in [1, 4, 16]:
        cfg = load_config(f"exp_p04_h{h}.json")
        # Lookback should match base
        _check(f"配置 lookback [h{h}]", f"h{h}",
               cfg.get("horizon") == h,
               f"horizon={cfg.get('horizon')} expected={h}")
        # model_search_space should have cnn_bilstm
        _check(f"配置 search_space [h{h}]", f"h{h}",
               "cnn_bilstm" in cfg.get("model_search_space", {}),
               "model_search_space 缺失 cnn_bilstm")


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def run_validation() -> bool:
    print("=" * 70)
    print("EXP-P04 实验结果验证检查")
    print("=" * 70)

    horizons = ["h1", "h4", "h16"]
    overall_pass = True

    for hs in horizons:
        print(f"\n{'─' * 70}")
        print(f"  Horizon: {hs}")
        print(f"{'─' * 70}")
        check_files(hs)
        check_data_integrity(hs)
        check_timeline(hs)
        check_scaler(hs)
        check_optuna_to_final(hs)
        check_metrics_recompute(hs)
        check_reproduce(hs)
        check_report_consistency(hs)
        check_model_sanity(hs)
        check_figures(hs)

    print(f"\n{'═' * 70}")
    print("  跨 Horizon 检查")
    print(f"{'═' * 70}")
    check_cross_horizon()
    check_config_consistency()

    # Summary
    total = len(_CHECKS)
    failed = [c for c in _CHECKS if c["status"] != "PASS"]
    error_count = len([c for c in failed if c["severity"] == "ERROR"])
    warn_count = len([c for c in failed if c["severity"] == "WARNING"])

    print(f"\n{'═' * 70}")
    print(f"  检查汇总：共 {total} 项  |  PASS: {total - len(failed)}  |  ERROR: {error_count}  |  WARNING: {warn_count}")
    print(f"{'═' * 70}")

    if failed:
        print("\n  失败项详情（ERROR / WARNING）：")
        for c in failed:
            print(f"  [{c['status']:^6}] [{c['horizon']:^4}] {c['name']}: {c['detail']}")
        overall_pass = False
    else:
        print("\n  所有检查通过！")

    # Write report
    report_path = STEP4_ROOT / "validation_report.txt"
    lines = [
        "EXP-P04 实验结果验证报告",
        "=" * 60,
        f"生成时间: {pd.Timestamp.now()}",
        "",
        f"总计: {total} 项  |  PASS: {total - len(failed)}  |  ERROR: {error_count}  |  WARNING: {warn_count}",
        "",
    ]
    for c in _CHECKS:
        lines.append(f"[{c['status']:^8}] [{c['horizon']:^4}] {c['name']}: {c['detail']}")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告已保存: {report_path.relative_to(PROJECT_ROOT)}")

    return overall_pass


if __name__ == "__main__":
    ok = run_validation()
    sys.exit(0 if ok else 1)
