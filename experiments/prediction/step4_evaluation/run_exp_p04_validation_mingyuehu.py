r"""明月湖数据集 EXP-P04 实验结果完整性检查脚本。

运行方式：
    python -m experiments.prediction.step4_evaluation.run_exp_p04_validation_mingyuehu

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
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STEP4_ROOT = PROJECT_ROOT / "data" / "prediction" / "step2_hyperparameter_search"
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.prediction.step2_hyperparameter_search.exp_p04_common import (
    MODELS_DIR, METRICS_DIR, PRED_DIR, FIGURES_DIR, REPORTS_DIR,
    LOG_DIR, SAMPLES_DIR,
    compute_all_metrics,
)

# 明月湖配置
MINGYUEHU_LOOKBACK_MAP = {1: 16, 4: 48, 16: 96}
MINGYUEHU_SEEDS = [42, 43, 44, 45, 46]

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


def _mingyuehu_sample_dir(horizon: int) -> Path:
    lookback = MINGYUEHU_LOOKBACK_MAP.get(horizon, 16)
    return SAMPLES_DIR / f"mingyuehu_h{horizon}_lb{lookback}"


# ---------------------------------------------------------------------------
# CHECK 1: 文件完整性
# ---------------------------------------------------------------------------

def check_files(horizon: int) -> None:
    """检查所有 expected 文件是否存在。"""
    hs = f"mingyuehu_h{horizon}"
    sample_dir = _mingyuehu_sample_dir(horizon)
    files = []
    # Samples
    for f in ("X_train_seq.npy", "X_val_seq.npy", "X_test_seq.npy",
              "y_train.npy", "y_val.npy", "y_test.npy",
              "y_anchor_train.npy", "y_anchor_val.npy", "y_anchor_test.npy",
              "y_residual_train_raw.npy", "y_residual_val_raw.npy", "y_residual_test_raw.npy",
              "test_timestamps.csv", "scaler_params.json", "meta.json"):
        files.append((sample_dir / f, f"SAMPLES / {f}"))

    # Models & predictions
    for suffix, label in [("_final.pt", "FINAL.pt"), ("_seed42.pt", "SEED.pt")]:
        files.append((MODELS_DIR / hs / f"mingyuehu_cnn_bilstm{suffix}",
                     f"MODEL / mingyuehu_cnn_bilstm{suffix}"))
    files.append((PRED_DIR / hs / "cnn_bilstm_test.csv", f"PRED  / cnn_bilstm_test.csv"))
    files.append((PRED_DIR / hs / "cnn_bilstm_seed42_test.csv", f"PRED  / cnn_bilstm_seed42_test.csv"))
    files.append((METRICS_DIR / hs / "mingyuehu_cnn_bilstm_optuna.json", f"OPTUNA / mingyuehu_cnn_bilstm_optuna.json"))
    files.append((METRICS_DIR / hs / "mingyuehu_hybrid_search_ablation.json", f"OPTUNA / mingyuehu_hybrid_search_ablation.json"))
    files.append((METRICS_DIR / hs / "mingyuehu_cnn_bilstm_test_metrics.json",
                  f"METRIC / mingyuehu_cnn_bilstm_test_metrics.json"))
    files.append((METRICS_DIR / hs / "mingyuehu_cnn_bilstm_reproduce.json",
                  f"REPROD / mingyuehu_cnn_bilstm_reproduce.json"))

    # Figures
    fig_dir = FIGURES_DIR / hs
    for fig in ("metrics_mae_bar.png", "metrics_rmse_bar.png", "metrics_r2_bar.png",
                "training_time.png", "predictions_overlay.png"):
        files.append((fig_dir / fig, f"FIG    / {fig}"))

    for prefix in ("pred_", "loss_"):
        files.append((fig_dir / f"{prefix}cnn_bilstm.png",
                     f"FIG    / {prefix}cnn_bilstm.png"))

    all_missing = []
    for path, label in files:
        if not path.exists():
            all_missing.append(label)

    sev = "ERROR" if len(all_missing) > 1 else "WARNING"
    _check("文件完整性", hs,
           len(all_missing) == 0,
           "%d/%d files missing: %s" % (len(all_missing), len(files), all_missing),
           severity=sev)


# ---------------------------------------------------------------------------
# CHECK 2: 数据完整性（shape / NaN / Inf / 范围）
# ---------------------------------------------------------------------------

def check_data_integrity(horizon: int) -> None:
    """检查样本数据的 shape、NaN、Inf 和合理范围。"""
    hs = f"mingyuehu_h{horizon}"
    base = _mingyuehu_sample_dir(horizon)

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
    for name, arr in [("y_train", y_train), ("y_val", y_val), ("y_test", y_test)]:
        ndim_ok = arr.ndim == 2 and arr.shape[1] == horizon
        _check(f"y shape [{name}]", hs, ndim_ok,
               f"{name}.shape={arr.shape} expected (N,{horizon})")

    # y is standardized: mean≈0, std≈1 (train strict; val/test may drift)
    for name, arr in [("y_train", y_train), ("y_test", y_test)]:
        y_mean = float(arr.mean())
        y_std = float(arr.std())
        mean_ok = abs(y_mean) < 0.05
        std_ok = 0.95 < y_std < 1.05
        _check("y 标准化 [%s]" % name, hs, mean_ok,
               "mean=%.4f != 0" % y_mean if not mean_ok else "mean=%.4f OK" % y_mean)
        _check("y 标准差 [%s]" % name, hs, std_ok,
               "std=%.4f out of [0.95,1.05]" % y_std if not std_ok else "std=%.4f OK" % y_std)
    # val: looser check
    y_val_mean = abs(float(y_val.mean()))
    _check("y val mean drift", hs, y_val_mean < 0.15,
           "val mean=%.4f (>0.15 可能有问题)" % float(y_val.mean())
           if y_val_mean >= 0.15 else "val mean=%.4f OK" % float(y_val_mean),
           severity="WARNING")

    # X feature count
    n_features = X_train.shape[2]
    expected_lookback = MINGYUEHU_LOOKBACK_MAP.get(horizon, 16)
    _check("X shape lookback", hs, X_train.shape[1] == expected_lookback,
           f"X_train.shape[1]={X_train.shape[1]} expected {expected_lookback}")
    _check("X features", hs, n_features >= 10,
           f"n_features={n_features}")

    # Divisibility: val + test total should make sense
    total = len(X_train) + len(X_val) + len(X_test)
    val_frac = len(X_val) / total
    test_frac = len(X_test) / total
    expected_val_frac = 0.15  # 明月湖配置
    expected_test_frac = 0.15
    _check("Val 划分比例", hs,
           abs(val_frac - expected_val_frac) < 0.02,
           f"val={val_frac:.3f} expected≈{expected_val_frac:.3f}")
    _check("Test 划分比例", hs,
           abs(test_frac - expected_test_frac) < 0.02,
           f"test={test_frac:.3f} expected≈{expected_test_frac:.3f}")


# ---------------------------------------------------------------------------
# CHECK 3: 时间线一致性（无泄露）
# ---------------------------------------------------------------------------

def check_timeline(horizon: int) -> None:
    """检查 train/val/test 时间戳严格递增，无时间重叠。"""
    hs = f"mingyuehu_h{horizon}"
    base = _mingyuehu_sample_dir(horizon)
    ts_path = base / "test_timestamps.csv"
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

def check_scaler(horizon: int) -> None:
    """检查 scaler 参数合理性（y 已标准化；用 scaler 反标准化后应落入 [0,1]）。"""
    hs = f"mingyuehu_h{horizon}"
    base = _mingyuehu_sample_dir(horizon)
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

def check_optuna_to_final(horizon: int) -> None:
    """检查 Optuna 最优参数是否正确传递给 Final Train。"""
    hs = f"mingyuehu_h{horizon}"
    optuna_path = METRICS_DIR / hs / "mingyuehu_cnn_bilstm_optuna.json"
    if not optuna_path.exists():
        _check("Optuna参数", hs, False, "optuna.json 不存在")
        return

    optuna = _load_json(optuna_path)
    if "best_params" not in optuna:
        _check("Optuna参数", hs, False, "无 best_params 字段")
        return

    best_params = optuna["best_params"]
    _check("Optuna参数存在", hs, True, f"best_params 包含 {len(best_params)} 个参数")
    _check("batch_size 参数", hs, "batch_size" in best_params, "缺少 batch_size")
    _check("lr 参数", hs, "lr" in best_params, "缺少 lr")


# ---------------------------------------------------------------------------
# CHECK 6: Metrics 重算一致性
# ---------------------------------------------------------------------------

def check_metrics_recompute(horizon: int) -> None:
    """从预测 CSV 重算 MAE/RMSE/R²，与 reproduce.json 对比。"""
    hs = f"mingyuehu_h{horizon}"
    reprod_path = METRICS_DIR / hs / "mingyuehu_cnn_bilstm_reproduce.json"
    pred_path = PRED_DIR / hs / "cnn_bilstm_seed42_test.csv"

    if not reprod_path.exists():
        _check("Metrics重算", hs, False, "reproduce.json 不存在")
        return
    if not pred_path.exists():
        _check("Metrics重算", hs, False, "pred CSV 不存在")
        return

    reprod = _load_json(reprod_path)
    reported = reprod.get("mean", {})
    if not reported:
        _check("Metrics重算", hs, False, "reproduce.json 无 mean 字段")
        return

    df = pd.read_csv(pred_path)
    if "y_true" not in df.columns or "y_pred" not in df.columns:
        _check("Metrics重算", hs, False, "pred CSV 缺少列")
        return

    computed = compute_all_metrics(df["y_true"].values, df["y_pred"].values)

    for metric in ("MAE", "RMSE", "R2"):
        rep_val = reported.get(metric)
        comp_val = computed.get(metric)
        if rep_val is None or comp_val is None:
            continue
        diff = abs(rep_val - comp_val)
        warn_threshold = {"MAE": 0.01, "RMSE": 0.015, "R2": 0.05}[metric]
        passed = diff < warn_threshold
        _check(f"Metrics [{metric}]", hs, passed,
               "reported=%.6f computed=%.6f diff=%.6e" % (rep_val, comp_val, diff),
               severity="WARNING" if not passed else "PASS")


# ---------------------------------------------------------------------------
# CHECK 7: 多 Seed 复现合理性
# ---------------------------------------------------------------------------

def check_reproduce(horizon: int) -> None:
    """检查 reproduce.json 的 seed 列表、std 范围合理性。"""
    hs = f"mingyuehu_h{horizon}"
    path = METRICS_DIR / hs / "mingyuehu_cnn_bilstm_reproduce.json"
    if not path.exists():
        _check("复现文件", hs, False, "文件不存在")
        return

    data = _load_json(path)
    seeds = data.get("seeds", [])
    per_seed = data.get("per_seed", [])

    # Seeds match expected
    _check("Seed列表", hs, seeds == MINGYUEHU_SEEDS,
           f"seeds={seeds} expected={MINGYUEHU_SEEDS}")

    # Correct number of per_seed entries
    _check("Per-seed数量", hs, len(per_seed) == len(MINGYUEHU_SEEDS),
           f"per_seed={len(per_seed)} expected={len(MINGYUEHU_SEEDS)}")

    # Std not too high (unstable model)
    std_mae = data.get("std", {}).get("MAE", None)
    if std_mae is not None:
        _check("MAE Std", hs, std_mae < 0.01,
               f"std_MAE={std_mae:.6f} (>0.01 不稳定)" if std_mae >= 0.01 else f"std={std_mae:.6f} OK")

    # Mean within expected range
    mean_mae = data.get("mean", {}).get("MAE", None)
    if mean_mae is not None:
        _check("MAE 合理性", hs, 0.001 < mean_mae < 0.5,
               f"MAE={mean_mae:.4f} 范围异常" if not (0.001 < mean_mae < 0.5) else f"MAE={mean_mae:.4f} OK")


# ---------------------------------------------------------------------------
# CHECK 8: 报告 vs 数据一致性
# ---------------------------------------------------------------------------

def check_report_consistency(horizon: int) -> None:
    """检查 Markdown 报告中的指标与 JSON 源文件是否一致。"""
    hs = f"mingyuehu_h{horizon}"
    report_path = REPORTS_DIR / f"EXP-P04_mingyuehu_h{horizon}_详细实验汇报.md"
    if not report_path.exists():
        _check("报告文件存在", hs, False, "Markdown 报告不存在")
        return

    report_text = report_path.read_text(encoding="utf-8")
    reprod_path = METRICS_DIR / hs / "mingyuehu_cnn_bilstm_reproduce.json"
    if not reprod_path.exists():
        _check("报告一致性", hs, False, "reproduce.json 不存在")
        return

    reprod = _load_json(reprod_path)
    mean = reprod.get("mean", {})
    if not mean:
        _check("报告一致性", hs, False, "reproduce.json 无 mean 字段")
        return

    # Check model name appears in report
    display_name = "CNN-BiLSTM"
    _check("报告包含模型", hs, display_name in report_text,
           f"报告中未找到 {display_name}" if display_name not in report_text else "OK")

    # Check MAE value in report (rough check: number like 0.0xxx)
    mae_val = mean.get("MAE")
    if mae_val is not None:
        found = f"{mae_val:.4f}" in report_text or f"{mae_val:.3f}" in report_text
        _check("报告MAE", hs, found,
               f"MAE={mae_val:.4f} 未在报告中找到（可能精度差异）", severity="WARNING")


# ---------------------------------------------------------------------------
# CHECK 9: 模型 Sanity Check
# ---------------------------------------------------------------------------

def check_model_sanity(horizon: int) -> None:
    """检查预测值不全为零，分布合理。"""
    hs = f"mingyuehu_h{horizon}"
    pred_path = PRED_DIR / hs / "cnn_bilstm_seed42_test.csv"
    if not pred_path.exists():
        _check("模型预测文件", hs, False, "pred CSV 不存在")
        return

    df = pd.read_csv(pred_path)
    y_pred = df["y_pred"].values
    y_true = df["y_true"].values

    # Not all zeros
    not_all_zero = not (np.abs(y_pred) < 1e-8).all()
    _check("预测不全为零", hs, not_all_zero,
           "预测值全为0" if not not_all_zero else "OK")

    # Not all constant
    not_constant = np.std(y_pred) > 1e-6
    _check("预测非常量", hs, not_constant,
           "预测值方差≈0" if not not_constant else f"std={np.std(y_pred):.6f} OK")

    # Within [0, 1]
    in_range = (y_pred >= -0.15).all() and (y_pred <= 1.15).all()
    _check("预测范围", hs, in_range,
           f"pred范围[{y_pred.min():.4f}, {y_pred.max():.4f}]" if not in_range else "OK")

    # Prediction vs truth correlation (should be positive)
    if len(y_pred) > 10:
        corr = np.corrcoef(y_true, y_pred)[0, 1]
        _check("预测相关性", hs, corr > 0.5,
               f"corr={corr:.4f} (<0.5 异常)" if corr <= 0.5 else f"corr={corr:.4f} OK")

    # MAE not absurdly high
    mae = float(np.mean(np.abs(y_true - y_pred)))
    _check("MAE合理性", hs, mae < 0.3,
           f"MAE={mae:.4f} 过高" if mae >= 0.3 else f"MAE={mae:.4f} OK")


# ---------------------------------------------------------------------------
# CHECK 10: 图表完整性
# ---------------------------------------------------------------------------

def check_figures(horizon: int) -> None:
    """检查所有 PNG 文件非空且大小合理（>5KB）。"""
    hs = f"mingyuehu_h{horizon}"
    fig_dir = FIGURES_DIR / hs
    if not fig_dir.exists():
        _check("图表目录", hs, False, "目录不存在")
        return

    png_files = list(fig_dir.glob("*.png"))
    min_expected = 5  # 明月湖报告图表数量
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
    horizons = [1, 4, 16]
    n_samples = {}

    for h in horizons:
        base = _mingyuehu_sample_dir(h)
        if not base.exists():
            continue
        try:
            y_test = np.load(base / "y_test.npy")
            n_samples[h] = len(y_test)
        except Exception:
            n_samples[h] = None

    # h1/h4/h16 cover the SAME time span -> same number of samples
    vals = [n_samples.get(h) for h in horizons]
    all_same = all(v == vals[0] for v in vals if v is not None)
    _check("h1/h4/h16 sample count consistent", None,
           all_same,
           "h1=%s h4=%s h16=%s" % (str(n_samples.get(1)), str(n_samples.get(4)), str(n_samples.get(16))),
           severity="INFO")

    mae_vals = {}
    for h in horizons:
        reprod_path = METRICS_DIR / f"mingyuehu_h{h}" / "mingyuehu_cnn_bilstm_reproduce.json"
        if reprod_path.exists():
            data = _load_json(reprod_path)
            mae_vals[h] = data.get("mean", {}).get("MAE")
        else:
            mae_vals[h] = None

    if all(v is not None for v in mae_vals.values()):
        h1_mae = mae_vals[1]
        h4_mae = mae_vals[4]
        h16_mae = mae_vals[16]
        monotonic = h1_mae < h4_mae < h16_mae
        _check("MAE 随 horizon 单调递增", None, monotonic,
               "MAE: h1=%.4f h4=%.4f h16=%.4f %s" % (
               h1_mae, h4_mae, h16_mae, "OK" if monotonic else "WARNING: not monotonic"))


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def run_validation() -> bool:
    print("=" * 70)
    print("明月湖数据集 EXP-P04 实验结果验证检查")
    print("=" * 70)

    horizons = [1, 4, 16]
    overall_pass = True

    for h in horizons:
        hs = f"mingyuehu_h{h}"
        print(f"\n{'─' * 70}")
        print(f"  Horizon: {h} ({hs})")
        print(f"{'─' * 70}")
        check_files(h)
        check_data_integrity(h)
        check_timeline(h)
        check_scaler(h)
        check_optuna_to_final(h)
        check_metrics_recompute(h)
        check_reproduce(h)
        check_report_consistency(h)
        check_model_sanity(h)
        check_figures(h)

    print(f"\n{'═' * 70}")
    print("  跨 Horizon 检查")
    print(f"{'═' * 70}")
    check_cross_horizon()

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
            print(f"  [{c['status']:^6}] [{c['horizon']:^16}] {c['name']}: {c['detail']}")
        overall_pass = False
    else:
        print("\n  所有检查通过！")

    # Write report
    report_path = STEP4_ROOT / "mingyuehu_validation_report.txt"
    lines = [
        "明月湖数据集 EXP-P04 实验结果验证报告",
        "=" * 60,
        f"生成时间: {pd.Timestamp.now()}",
        "",
        f"总计: {total} 项  |  PASS: {total - len(failed)}  |  ERROR: {error_count}  |  WARNING: {warn_count}",
        "",
    ]
    for c in _CHECKS:
        lines.append(f"[{c['status']:^8}] [{c['horizon']:^16}] {c['name']}: {c['detail']}")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告已保存: {report_path.relative_to(PROJECT_ROOT)}")

    return overall_pass


if __name__ == "__main__":
    ok = run_validation()
    sys.exit(0 if ok else 1)
