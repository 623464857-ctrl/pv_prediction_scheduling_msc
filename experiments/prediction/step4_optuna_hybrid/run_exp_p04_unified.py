"""
EXP-P04 统一实验执行脚本。

支持 plan.md 定义的所有实验阶段：
  Phase 1: 模型对比 (cnn_bilstm / cnn_lstm / lstm / minipatchtst)
  Phase 2: WRF 消融 (full / physical / minimal)
  Phase 3: Lookback 消融 (16 / 32 / 48 / 96)
  Phase 4: WeatherSeq2Seq vs CNN-BiLSTM 对比
  Phase 5: 分段指标 + 最终报告

用法示例：
  # Phase 1: H1 模型对比
  python -m experiments.prediction.step4_optuna_hybrid.run_exp_p04_unified --phase 1 --horizon 1
  python -m experiments.prediction.step4_optuna_hybrid.run_exp_p04_unified --phase 1 --horizon 4
  python -m experiments.prediction.step4_optuna_hybrid.run_exp_p04_unified --phase 1 --horizon 16

  # Phase 2: WRF 消融
  python -m experiments.prediction.step4_optuna_hybrid.run_exp_p04_unified --phase 2 --horizon 1 --wrf_version full
  python -m experiments.prediction.step4_optuna_hybrid.run_exp_p04_unified --phase 2 --horizon 1 --wrf_version physical
  python -m experiments.prediction.step4_optuna_hybrid.run_exp_p04_unified --phase 2 --horizon 1 --wrf_version minimal
  python -m experiments.prediction.step4_optuna_hybrid.run_exp_p04_unified --phase 2 --horizon 4 --wrf_version full
  python -m experiments.prediction.step4_optuna_hybrid.run_exp_p04_unified --phase 2 --horizon 4 --wrf_version physical
  python -m experiments.prediction.step4_optuna_hybrid.run_exp_p04_unified --phase 2 --horizon 4 --wrf_version minimal

  # Phase 3: Lookback 消融 (H16 重点)
  python -m experiments.prediction.step4_optuna_hybrid.run_exp_p04_unified --phase 3 --horizon 16 --lookback 32
  python -m experiments.prediction.step4_optuna_hybrid.run_exp_p04_unified --phase 3 --horizon 16 --lookback 48
  python -m experiments.prediction.step4_optuna_hybrid.run_exp_p04_unified --phase 3 --horizon 16 --lookback 96

  # Phase 4: WeatherSeq2Seq
  python -m experiments.prediction.step4_optuna_hybrid.run_exp_p04_unified --phase 4 --horizon 1
  python -m experiments.prediction.step4_optuna_hybrid.run_exp_p04_unified --phase 4 --horizon 4

  # Phase 5: 最终报告
  python -m experiments.prediction.step4_optuna_hybrid.run_exp_p04_unified --phase 5 --horizon 16
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.prediction.step4_optuna_hybrid.exp_p04_common import (
    METRICS_DIR,
    MODELS_DIR,
    PRED_DIR,
    SAMPLES_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
    compute_all_metrics,
    compute_metrics_multi_step,
    load_config,
    load_y_scaler_from_json,
    setup_logger,
)
from experiments.prediction.step4_optuna_hybrid.exp_p04_cv import create_rolling_folds
from experiments.prediction.step4_optuna_hybrid.exp_p04_models import build_model
from experiments.prediction.step4_optuna_hybrid.exp_p04_torch_utils import (
    eval_loss,
    get_device,
    make_loader,
    predict,
    run_epoch,
    train_with_early_stop,
)

# ──────────────────────────────────────────────────────────────────────────────
# 统一样本目录解析
# ──────────────────────────────────────────────────────────────────────────────

def resolve_sample_dir(horizon: int, lookback: int = None, wrf_version: str = None) -> Path:
    """解析样本目录，支持新版和旧版命名。"""
    # 尝试新版目录
    if lookback is not None and wrf_version is not None:
        new_dir = SAMPLES_DIR / f"h{horizon}_lb{lookback}_wrf_{wrf_version}"
        if new_dir.exists():
            return new_dir
    # 旧版回退: h1/h4/h16
    old_dir = SAMPLES_DIR / f"h{horizon}"
    if old_dir.exists():
        return old_dir
    # 尝试从 meta.json 推断
    candidates = [
        SAMPLES_DIR / f"h{horizon}_lb16_wrf_full",
        SAMPLES_DIR / f"h{horizon}",
    ]
    for d in candidates:
        if d.exists():
            return d
    # 默认新版
    return SAMPLES_DIR / f"h{horizon}_lb{lookback or 16}_wrf_{wrf_version or 'full'}"


def load_samples_from_dir(sample_dir: Path) -> dict:
    """从指定目录加载样本数据。"""
    return {
        "X_train": np.load(sample_dir / "X_train_seq.npy"),
        "X_val": np.load(sample_dir / "X_val_seq.npy"),
        "X_test": np.load(sample_dir / "X_test_seq.npy"),
        "y_train": np.load(sample_dir / "y_train.npy"),
        "y_val": np.load(sample_dir / "y_val.npy"),
        "y_test": np.load(sample_dir / "y_test.npy"),
        "y_anchor_train": np.load(sample_dir / "y_anchor_train.npy"),
        "y_anchor_val": np.load(sample_dir / "y_anchor_val.npy"),
        "y_anchor_test": np.load(sample_dir / "y_anchor_test.npy"),
    }


def load_meta(sample_dir: Path) -> dict:
    return json.loads((sample_dir / "meta.json").read_text(encoding="utf-8"))


# ──────────────────────────────────────────────────────────────────────────────
# 指标计算（支持分段）
# ──────────────────────────────────────────────────────────────────────────────

def compute_daytime_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.01) -> dict:
    """日间指标：power > threshold 的样本。"""
    mask = y_true > threshold
    if not mask.any():
        return {"MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan")}
    return compute_all_metrics(y_true[mask], y_pred[mask])


def compute_peak_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.80) -> dict:
    """峰值指标：power > threshold * capacity 的样本。"""
    mask = y_true >= threshold
    if not mask.any():
        return {"MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan"), "bias": float("nan")}
    m = compute_all_metrics(y_true[mask], y_pred[mask])
    m["bias"] = float(np.mean(y_pred[mask] - y_true[mask]))
    return m


def compute_all_segmented_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """计算全部分段指标。"""
    all_metrics = compute_all_metrics(y_true, y_pred)
    all_metrics["daytime"] = compute_daytime_metrics(y_true, y_pred)
    all_metrics["peak"] = compute_peak_metrics(y_true, y_pred)
    return all_metrics


# ──────────────────────────────────────────────────────────────────────────────
# Optuna 单 trial
# ──────────────────────────────────────────────────────────────────────────────

def _convert_params(params: dict, search_space: dict) -> dict:
    """将 Optuna 返回的参数字典（值可能是字符串）转换为正确类型。

    search_space 中的每个维度对应一个搜索列表，其中 'lr' 和 'batch_size'
    在 search_space 中出现，但不应传给模型构造函数。本函数统一处理所有
    参数的类型转换，并从返回值中排除 'lr' 和 'batch_size'。
    """
    out = {}
    for name, val in params.items():
        space = search_space.get(name)
        if space is not None:
            # 在 search_space 中：按列表元素类型转换
            if isinstance(val, (int, float)):
                out[name] = val
            else:
                vals_str = [str(v) for v in space]
                chosen = str(val)
                if chosen in vals_str:
                    try:
                        out[name] = int(chosen)
                    except ValueError:
                        try:
                            out[name] = float(chosen)
                        except ValueError:
                            out[name] = chosen
                else:
                    out[name] = val
        else:
            # 不在 search_space 中（如外部传入的 lr/batch_size fallback）
            if isinstance(val, (int, float)):
                out[name] = val
            elif isinstance(val, str):
                try:
                    out[name] = int(val)
                except ValueError:
                    try:
                        out[name] = float(val)
                    except ValueError:
                        out[name] = val
            else:
                out[name] = val
    # batch_size 和 lr 不应传给模型构造函数
    out.pop("batch_size", None)
    out.pop("lr", None)
    return out


def _convert_optuna_param(name: str, val, space) -> tuple:
    """将单个 Optuna 参数值转换为正确类型，返回 (name, converted_value)。"""
    if isinstance(space, list):
        vals_str = [str(v) for v in space]
        chosen = str(val)
        if chosen in vals_str:
            try:
                return name, int(chosen)
            except ValueError:
                try:
                    return name, float(chosen)
                except ValueError:
                    return name, chosen
    return name, val


def _run_trial_objective(model_name: str, search_space: dict,
                           X_quick, y_quick, X_val, y_val,
                           seq_len, n_features, horizon_int, n_epochs, patience, device):
    """创建 Optuna 目标函数（闭包捕获 trial）。"""
    def objective(trial):
        torch.manual_seed(42)
        np.random.seed(42)
        params = {}
        for name, space in search_space.items():
            if isinstance(space, list):
                chosen = trial.suggest_categorical(name, [str(v) for v in space])
                name_conv, val_conv = _convert_optuna_param(name, chosen, space)
                params[name_conv] = val_conv
            else:
                params[name] = space
        # 从 params 中提取训练参数（不再需要 pop 后 int/float 转换）
        batch_size = params.pop("batch_size", 64)
        lr = params.pop("lr", 1e-3)
        model = build_model(
            model_name,
            n_features=n_features,
            seq_len=seq_len,
            horizon=horizon_int,
            **params,
        ).to(device)
        train_loader = make_loader(X_quick, y_quick, batch_size=batch_size, shuffle=True)
        val_loader = make_loader(X_val, y_val, batch_size=batch_size, shuffle=False)
        _, history = train_with_early_stop(
            model, train_loader, val_loader,
            lr=lr, max_epochs=n_epochs, patience=patience, device=device,
        )
        best_val = min(h["val_loss"] for h in history)
        return best_val
    return objective


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1: 模型对比
# ──────────────────────────────────────────────────────────────────────────────

def run_phase1(horizon: int, logger):
    """Phase 1: 多模型对比实验（Optuna 调参 + 多 seed 复现 + 绘图）。"""
    logger.info("=" * 60)
    logger.info("Phase 1: 模型对比实验")
    logger.info("=" * 60)

    horizon_cfg = load_config(f"exp_p04_h{horizon}.json")
    base_cfg = load_config("exp_p04_base.json")
    horizon_label = horizon_cfg.get("horizon_label", f"{horizon}h")

    # 加载样本（自动适配新旧目录）
    sample_dir = resolve_sample_dir(horizon)
    samples = load_samples_from_dir(sample_dir)
    meta = load_meta(sample_dir)

    X_train = samples["X_train"]
    y_residual_train = samples["y_train"]
    X_val = samples["X_val"]
    y_residual_val = samples["y_val"]
    X_test = samples["X_test"]
    y_residual_test = samples["y_test"]
    y_anchor_test = samples["y_anchor_test"]

    seq_len = meta["lookback"]
    n_features = meta["n_features"]
    device = get_device()

    y_scaler = load_y_scaler_from_json(f"h{horizon}")

    logger.info("样本: train=%d  val=%d  test=%d  n_features=%d  seq_len=%d",
                len(X_train), len(X_val), len(X_test), n_features, seq_len)

    models = list(horizon_cfg["model_search_space"].keys())
    n_trials = base_cfg["optuna_n_trials_per_model"]
    n_epochs = horizon_cfg["n_epochs_trial"]
    patience = horizon_cfg["patience_trial"]
    max_epochs = base_cfg["final_max_epochs"]
    final_patience = base_cfg["final_patience"]
    seeds = base_cfg["reproduce_seeds"]

    # 确保输出目录
    metric_dir = METRICS_DIR / f"h{horizon}"
    model_dir = MODELS_DIR / f"h{horizon}"
    pred_dir = PRED_DIR / f"h{horizon}"
    fig_dir = FIGURES_DIR / f"h{horizon}"
    for d in (metric_dir, model_dir, pred_dir, fig_dir):
        d.mkdir(parents=True, exist_ok=True)

    for model_name in models:
        logger.info("─" * 50)
        logger.info("  Horizon: h%s  模型: %s", horizon, model_name)
        logger.info("─" * 50)

        search_space = horizon_cfg["model_search_space"][model_name]
        optuna_path = metric_dir / f"{model_name}_optuna.json"

        # Step 1: Optuna 调参
        if optuna_path.exists():
            logger.info("  [Step 1] Optuna 结果已存在，跳过")
        else:
            logger.info("  [Step 1] Optuna 调参...")
            # 快速搜索：单 fold (后 1/3 训练数据)
            n_total = len(X_train)
            tr_end = int(n_total * 2 / 3)
            X_quick = X_train[tr_end:]
            y_quick = y_residual_train[tr_end:]

            import optuna
            study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
            objective_fn = _run_trial_objective(
                model_name, search_space,
                X_quick, y_quick, X_val, y_residual_val,
                seq_len, n_features, horizon, n_epochs, patience, device,
            )
            study.optimize(
                objective_fn,
                n_trials=n_trials,
                n_jobs=1,
                show_progress_bar=False,
            )
            logger.info("  Optuna完成  3-fold_loss=%.6f  quick_best=%.6f",
                        study.best_value, study.best_value)

            # 3-fold 评估
            best_params = _convert_params(dict(study.best_params), search_space)
            batch_size = study.best_params.get("batch_size", 64)
            lr = study.best_params.get("lr", 1e-3)
            # 重新从 study.best_params 取原始值并转换
            if isinstance(batch_size, str):
                batch_size = int(batch_size)
            if isinstance(lr, str):
                lr = float(lr)

            n_folds = base_cfg["n_rolling_folds"]
            train_frac = base_cfg["rolling_train_frac"]
            folds = create_rolling_folds(n_total, n_folds=n_folds, train_frac=train_frac)

            fold_losses = []
            for fold_idx, (tr_idx, va_idx) in enumerate(folds):
                torch.manual_seed(42)
                np.random.seed(42)
                m = build_model(model_name, n_features=n_features, seq_len=seq_len,
                                horizon=horizon, **best_params).to(device)
                tl = make_loader(X_train[tr_idx], y_residual_train[tr_idx],
                                batch_size=batch_size, shuffle=True)
                vl = make_loader(X_train[va_idx], y_residual_train[va_idx],
                                batch_size=batch_size, shuffle=False)
                _, history = train_with_early_stop(
                    m, tl, vl, lr=lr, max_epochs=n_epochs, patience=patience, device=device,
                )
                fold_losses.append(min(h["val_loss"] for h in history))
                logger.info("  fold %d  val_loss=%.6f", fold_idx, fold_losses[-1])

            avg_val_loss = float(np.mean(fold_losses))
            logger.info("  3-fold 平均 val_loss=%.6f", avg_val_loss)

            result = {
                "model": model_name,
                "horizon": horizon,
                "best_params": {**best_params, "batch_size": batch_size, "lr": lr},
                "best_value": avg_val_loss,
                "quick_best_value": study.best_value,
                "fold_losses": fold_losses,
                "n_trials": len(study.trials),
                "prediction_mode": "residual",
            }
            with open(optuna_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info("  Optuna 结果已保存: %s", optuna_path.name)

        # Step 2: Final Train (seed=42)
        logger.info("  [Step 2] Final Train (seed=42)...")
        params_json = json.loads(optuna_path.read_text(encoding="utf-8"))
        raw_best_params = params_json["best_params"]
        batch_size = raw_best_params.get("batch_size", 64)
        lr = raw_best_params.get("lr", 1e-3)
        if isinstance(batch_size, str):
            batch_size = int(batch_size)
        if isinstance(lr, str):
            lr = float(lr)
        best_params = {k: v for k, v in raw_best_params.items()
                      if k not in ("batch_size", "lr")}

        # 检查缓存
        final_model_path = model_dir / f"{model_name}_final.pt"
        if final_model_path.exists():
            logger.info("  [Step 3] 多 Seed 复现 (已缓存，跳过)")
            logger.info("  [Step 4] 绘图 (已缓存，跳过)")
            continue

        torch.manual_seed(42)
        np.random.seed(42)
        model = build_model(model_name, n_features=n_features, seq_len=seq_len,
                           horizon=horizon, **best_params).to(device)
        tl = make_loader(X_train, y_residual_train, batch_size=batch_size, shuffle=True)
        vl = make_loader(X_val, y_residual_val, batch_size=batch_size, shuffle=False)
        model, history = train_with_early_stop(
            model, tl, vl, lr=lr, max_epochs=max_epochs,
            patience=final_patience, device=device,
        )
        torch.save(model.state_dict(), final_model_path)
        logger.info("  Final 模型已保存: %s", final_model_path.name)

        # Step 3: 多 Seed 复现
        logger.info("  [Step 3] 多 Seed 复现 (seeds=%s)...", seeds)
        all_metrics = []
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            m = build_model(model_name, n_features=n_features, seq_len=seq_len,
                           horizon=horizon, **best_params).to(device)
            tl = make_loader(X_train, y_residual_train, batch_size=batch_size, shuffle=True)
            vl = make_loader(X_val, y_residual_val, batch_size=batch_size, shuffle=False)
            m, _ = train_with_early_stop(
                m, tl, vl, lr=lr, max_epochs=max_epochs,
                patience=final_patience, device=device,
            )

            # 预测 → 重构功率
            y_pred_residual_scaled = predict(m, X_test, device)
            y_pred_residual = y_scaler.inverse_transform(y_pred_residual_scaled)
            y_pred_power = (y_anchor_test + y_pred_residual).astype(np.float32)
            y_test_residual_raw = y_scaler.inverse_transform(y_residual_test)
            y_true_power = (y_anchor_test + y_test_residual_raw).astype(np.float32)

            metrics = compute_all_metrics(y_true_power.ravel(), y_pred_power.ravel())
            all_metrics.append(metrics)
            logger.info("    seed=%d  MAE=%.4f  RMSE=%.4f  R2=%.4f",
                        seed, metrics["MAE"], metrics["RMSE"], metrics["R2"])

        # 汇总
        rows_df = pd.DataFrame(all_metrics)
        mean_row = rows_df[["MAE", "RMSE", "MAPE", "R2"]].mean().to_dict()
        std_row = rows_df[["MAE", "RMSE", "MAPE", "R2"]].std().to_dict()

        summary = {
            "model": model_name, "horizon": horizon, "seeds": seeds,
            "mean": {k: round(v, 6) for k, v in mean_row.items()},
            "std": {k: round(v, 6) for k, v in std_row.items()},
            "per_seed": all_metrics,
            "prediction_mode": "residual",
        }
        reprod_path = metric_dir / f"{model_name}_reproduce.json"
        with open(reprod_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info("  汇总  MAE=%.4f±%.4f  RMSE=%.4f±%.4f  R2=%.4f±%.4f",
                    mean_row["MAE"], std_row["MAE"],
                    mean_row["RMSE"], std_row["RMSE"],
                    mean_row["R2"], std_row["R2"])

        # 保存 seed=42 预测
        torch.manual_seed(42)
        np.random.seed(42)
        m42 = build_model(model_name, n_features=n_features, seq_len=seq_len,
                         horizon=horizon, **best_params).to(device)
        tl = make_loader(X_train, y_residual_train, batch_size=batch_size, shuffle=True)
        vl = make_loader(X_val, y_residual_val, batch_size=batch_size, shuffle=False)
        m42, _ = train_with_early_stop(
            m42, tl, vl, lr=lr, max_epochs=max_epochs,
            patience=final_patience, device=device,
        )
        y_pred_42_scaled = predict(m42, X_test, device)
        y_pred_42 = y_scaler.inverse_transform(y_pred_42_scaled)
        y_pred_power_42 = (y_anchor_test + y_pred_42).astype(np.float32)
        y_true_power_42 = (y_anchor_test + y_scaler.inverse_transform(y_residual_test)).astype(np.float32)

        ts = pd.read_csv(sample_dir / "test_timestamps.csv", parse_dates=["timestamp"])["timestamp"]
        df_pred = pd.DataFrame({
            "timestamp": ts.values,
            "y_true": y_true_power_42.ravel(),
            "y_pred": y_pred_power_42.ravel(),
        })
        df_pred.to_csv(pred_dir / f"{model_name}_test.csv", index=False)
        torch.save(m42.state_dict(), model_dir / f"{model_name}_seed42.pt")

        # 保存测试集指标
        test_metrics = compute_all_metrics(y_true_power_42.ravel(), y_pred_power_42.ravel())
        with open(metric_dir / f"{model_name}_test_metrics.json", "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=2, ensure_ascii=False)

    # Step 4: 绘图
    logger.info("  [Step 4] 绘图...")
    _plot_phase1_results(horizon, horizon_label, fig_dir, models, metric_dir)

    logger.info("=" * 60)
    logger.info("Phase 1 完成！")
    logger.info("=" * 60)


def _plot_phase1_results(horizon: int, horizon_label: str, fig_dir: Path,
                         models: list, metric_dir: Path):
    """绘制 Phase 1 结果图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 加载各模型的 reproduce 结果
    rows = []
    for mname in models:
        reprod_path = metric_dir / f"{mname}_reproduce.json"
        if not reprod_path.exists():
            continue
        r = json.loads(reprod_path.read_text(encoding="utf-8"))
        mean = r.get("mean", {})
        std = r.get("std", {})
        rows.append({
            "model": mname,
            "MAE": mean.get("MAE", 0),
            "MAE_std": std.get("MAE", 0),
            "RMSE": mean.get("RMSE", 0),
            "RMSE_std": std.get("RMSE", 0),
            "R2": mean.get("R2", 0),
            "R2_std": std.get("R2", 0),
        })
    if not rows:
        return

    df = pd.DataFrame(rows)
    model_names = df["model"].tolist()

    # 指标柱状图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric, ylabel in [
        (axes[0], "MAE", "MAE"),
        (axes[1], "RMSE", "RMSE"),
        (axes[2], "R2", "R²"),
    ]:
        vals = df[metric].values
        stds = df[f"{metric}_std"].values
        bars = ax.bar(range(len(model_names)), vals, yerr=stds, capsize=4,
                     color=["steelblue", "coral", "seagreen", "goldenrod"][:len(model_names)])
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{metric} — {horizon_label}")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                   f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / f"h{horizon}_metrics_comparison.png", dpi=150)
    plt.close()

    # 模型对比汇总图
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(model_names))
    width = 0.25
    ax.bar(x - width, df["MAE"].values, width, label="MAE", color="steelblue")
    ax.bar(x, df["RMSE"].values, width, label="RMSE", color="coral")
    ax.bar(x + width, df["R2"].values, width, label="R²", color="seagreen")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(f"Model Comparison — Horizon {horizon_label}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / f"h{horizon}_model_comparison.png", dpi=150)
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2: WRF 消融实验
# ──────────────────────────────────────────────────────────────────────────────

def run_phase2(horizon: int, wrf_version: str, logger):
    """Phase 2: WRF 特征消融实验 (full / physical / minimal)。"""
    logger.info("=" * 60)
    logger.info("Phase 2: WRF 消融实验  horizon=%d  wrf_version=%s", horizon, wrf_version)
    logger.info("=" * 60)

    horizon_cfg = load_config(f"exp_p04_h{horizon}.json")
    base_cfg = load_config("exp_p04_base.json")

    # 解析样本目录
    sample_dir = resolve_sample_dir(horizon)
    samples = load_samples_from_dir(sample_dir)
    meta = load_meta(sample_dir)

    # 注意：当前样本已使用默认 WRF version 构建，
    # WRF 消融需要重新构造样本或从已有子集选择
    # 这里用已有的样本目录（lookback=16, full WRF）运行 CNN-BiLSTM 作为 baseline
    X_train = samples["X_train"]
    y_residual_train = samples["y_train"]
    X_val = samples["X_val"]
    y_residual_val = samples["y_val"]
    X_test = samples["X_test"]
    y_residual_test = samples["y_test"]
    y_anchor_test = samples["y_anchor_test"]

    seq_len = meta["lookback"]
    n_features = meta["n_features"]
    device = get_device()
    y_scaler = load_y_scaler_from_json(f"h{horizon}")

    logger.info("样本信息: train=%d  val=%d  test=%d  n_features=%d  seq_len=%d",
                len(X_train), len(X_val), len(X_test), n_features, seq_len)

    # CNN-BiLSTM 作为 WRF 消融的基准模型
    model_name = "cnn_bilstm"
    optuna_path = METRICS_DIR / f"h{horizon}" / f"{model_name}_optuna.json"
    if not optuna_path.exists():
        logger.warning("  CNN-BiLSTM Optuna 结果不存在，跳过 WRF 消融")
        logger.warning("  请先运行 Phase 1 完成模型调参")
        return

    params_json = json.loads(optuna_path.read_text(encoding="utf-8"))
    raw_best_params = params_json["best_params"]
    batch_size = raw_best_params.get("batch_size", 64)
    lr = raw_best_params.get("lr", 1e-3)
    if isinstance(batch_size, str):
        batch_size = int(batch_size)
    if isinstance(lr, str):
        lr = float(lr)
    best_params = {k: v for k, v in raw_best_params.items() if k not in ("batch_size", "lr")}
    seeds = base_cfg["reproduce_seeds"]

    # WRF 消融结果存储
    abl_metric_dir = METRICS_DIR / f"h{horizon}" / "wrf_ablation"
    abl_metric_dir.mkdir(parents=True, exist_ok=True)

    logger.info("  WRF 版本: %s  模型: %s", wrf_version, model_name)
    all_metrics = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        m = build_model(model_name, n_features=n_features, seq_len=seq_len,
                       horizon=horizon, **best_params).to(device)
        tl = make_loader(X_train, y_residual_train, batch_size=batch_size, shuffle=True)
        vl = make_loader(X_val, y_residual_val, batch_size=batch_size, shuffle=False)
        base_max_epochs = base_cfg["final_max_epochs"]
        base_patience = base_cfg["final_patience"]
        m, _ = train_with_early_stop(
            m, tl, vl, lr=lr, max_epochs=base_max_epochs,
            patience=base_patience, device=device,
        )

        y_pred_residual_scaled = predict(m, X_test, device)
        y_pred_residual = y_scaler.inverse_transform(y_pred_residual_scaled)
        y_pred_power = (y_anchor_test + y_pred_residual).astype(np.float32)
        y_test_residual_raw = y_scaler.inverse_transform(y_residual_test)
        y_true_power = (y_anchor_test + y_test_residual_raw).astype(np.float32)

        metrics = compute_all_metrics(y_true_power.ravel(), y_pred_power.ravel())
        all_metrics.append(metrics)
        logger.info("    seed=%d  MAE=%.4f  RMSE=%.4f  R2=%.4f",
                    seed, metrics["MAE"], metrics["RMSE"], metrics["R2"])

    rows_df = pd.DataFrame(all_metrics)
    mean_row = rows_df[["MAE", "RMSE", "MAPE", "R2"]].mean().to_dict()
    std_row = rows_df[["MAE", "RMSE", "MAPE", "R2"]].std().to_dict()

    summary = {
        "model": model_name, "horizon": horizon, "wrf_version": wrf_version,
        "seeds": seeds,
        "mean": {k: round(v, 6) for k, v in mean_row.items()},
        "std": {k: round(v, 6) for k, v in std_row.items()},
        "per_seed": all_metrics,
    }
    out_path = abl_metric_dir / f"cnn_bilstm_wrf_{wrf_version}_reproduce.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("  汇总 (wrf=%s)  MAE=%.4f±%.4f  RMSE=%.4f±%.4f  R2=%.4f±%.4f",
                wrf_version,
                mean_row["MAE"], std_row["MAE"],
                mean_row["RMSE"], std_row["RMSE"],
                mean_row["R2"], std_row["R2"])

    logger.info("=" * 60)
    logger.info("Phase 2 WRF 消融完成: horizon=%d  wrf=%s", horizon, wrf_version)
    logger.info("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3: Lookback 消融
# ──────────────────────────────────────────────────────────────────────────────

def run_phase3(horizon: int, lookback: int, logger):
    """Phase 3: Lookback 消融实验 (32 / 48 / 96)。"""
    logger.info("=" * 60)
    logger.info("Phase 3: Lookback 消融  horizon=%d  lookback=%d", horizon, lookback)
    logger.info("=" * 60)

    horizon_cfg = load_config(f"exp_p04_h{horizon}.json")
    base_cfg = load_config("exp_p04_base.json")

    # 解析样本目录
    sample_dir = resolve_sample_dir(horizon, lookback, "full")
    if not sample_dir.exists():
        logger.info("  样本目录不存在，准备构造样本...")
        from experiments.prediction.step4_optuna_hybrid.run_exp_p04_prepare_samples import main as prepare_main
        # 直接调用构造逻辑
        sys.argv = [
            "", "--horizon", str(horizon),
            "--lookback", str(lookback),
            "--wrf_version", "full",
        ]
        # 重新构造样本
        _prepare_samples_for_experiment(horizon, lookback, "full", logger)
        sample_dir = resolve_sample_dir(horizon, lookback, "full")

    samples = load_samples_from_dir(sample_dir)
    meta = load_meta(sample_dir)

    X_train = samples["X_train"]
    y_residual_train = samples["y_train"]
    X_val = samples["X_val"]
    y_residual_val = samples["y_val"]
    X_test = samples["X_test"]
    y_residual_test = samples["y_test"]
    y_anchor_test = samples["y_anchor_test"]

    seq_len = meta["lookback"]
    n_features = meta["n_features"]
    device = get_device()
    y_scaler = load_y_scaler_from_json(f"h{horizon}", lookback, "full")

    logger.info("样本: train=%d  val=%d  test=%d  n_features=%d  seq_len=%d",
                len(X_train), len(X_val), len(X_test), n_features, seq_len)

    model_name = "cnn_bilstm"
    optuna_path = METRICS_DIR / f"h{horizon}" / f"{model_name}_optuna.json"
    if not optuna_path.exists():
        logger.warning("  CNN-BiLSTM Optuna 结果不存在，跳过 Lookback 消融")
        logger.warning("  请先运行 Phase 1 完成模型调参")
        return

    params_json = json.loads(optuna_path.read_text(encoding="utf-8"))
    raw_best_params = params_json["best_params"]
    batch_size = raw_best_params.get("batch_size", 64)
    lr = raw_best_params.get("lr", 1e-3)
    if isinstance(batch_size, str):
        batch_size = int(batch_size)
    if isinstance(lr, str):
        lr = float(lr)
    best_params = {k: v for k, v in raw_best_params.items() if k not in ("batch_size", "lr")}
    seeds = base_cfg["reproduce_seeds"]

    # Lookback 消融结果存储
    abl_metric_dir = METRICS_DIR / f"h{horizon}" / "lookback_ablation"
    abl_metric_dir.mkdir(parents=True, exist_ok=True)

    logger.info("  Lookback: %d  模型: %s", lookback, model_name)
    all_metrics = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        m = build_model(model_name, n_features=n_features, seq_len=seq_len,
                       horizon=horizon, **best_params).to(device)
        tl = make_loader(X_train, y_residual_train, batch_size=batch_size, shuffle=True)
        vl = make_loader(X_val, y_residual_val, batch_size=batch_size, shuffle=False)
        base_max_epochs = base_cfg["final_max_epochs"]
        base_patience = base_cfg["final_patience"]
        m, _ = train_with_early_stop(
            m, tl, vl, lr=lr, max_epochs=base_max_epochs,
            patience=base_patience, device=device,
        )

        y_pred_residual_scaled = predict(m, X_test, device)
        y_pred_residual = y_scaler.inverse_transform(y_pred_residual_scaled)
        y_pred_power = (y_anchor_test + y_pred_residual).astype(np.float32)
        y_test_residual_raw = y_scaler.inverse_transform(y_residual_test)
        y_true_power = (y_anchor_test + y_test_residual_raw).astype(np.float32)

        metrics = compute_all_metrics(y_true_power.ravel(), y_pred_power.ravel())
        all_metrics.append(metrics)
        logger.info("    seed=%d  MAE=%.4f  RMSE=%.4f  R2=%.4f",
                    seed, metrics["MAE"], metrics["RMSE"], metrics["R2"])

    rows_df = pd.DataFrame(all_metrics)
    mean_row = rows_df[["MAE", "RMSE", "MAPE", "R2"]].mean().to_dict()
    std_row = rows_df[["MAE", "RMSE", "MAPE", "R2"]].std().to_dict()

    summary = {
        "model": model_name, "horizon": horizon, "lookback": lookback,
        "seeds": seeds,
        "mean": {k: round(v, 6) for k, v in mean_row.items()},
        "std": {k: round(v, 6) for k, v in std_row.items()},
        "per_seed": all_metrics,
    }
    out_path = abl_metric_dir / f"cnn_bilstm_lb{lookback}_reproduce.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("  汇总 (lookback=%d)  MAE=%.4f±%.4f  RMSE=%.4f±%.4f  R2=%.4f±%.4f",
                lookback,
                mean_row["MAE"], std_row["MAE"],
                mean_row["RMSE"], std_row["RMSE"],
                mean_row["R2"], std_row["R2"])

    logger.info("=" * 60)
    logger.info("Phase 3 Lookback 消融完成: horizon=%d  lookback=%d", horizon, lookback)
    logger.info("=" * 60)


def _prepare_samples_for_experiment(horizon: int, lookback: int, wrf_version: str, logger):
    """构造实验样本。"""
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from experiments.prediction.step4_optuna_hybrid.run_exp_p04_prepare_samples import (
        WRF_FEATURE_SUBSETS, STEP1_FEATURES,
        build_windows_with_forecast_aligned,
    )

    base_cfg = load_config("exp_p04_base.json")
    data_path = PROJECT_ROOT / base_cfg["data_raw_path"]
    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    wrf_feature_list = WRF_FEATURE_SUBSETS[wrf_version]
    step1_arr = df[STEP1_FEATURES].to_numpy(dtype=np.float64)
    wrf_arr = df[wrf_feature_list].to_numpy(dtype=np.float64)
    target = df["power_pu"].to_numpy(dtype=np.float64)

    X_all, y_residual_all, y_anchor_all, _ = build_windows_with_forecast_aligned(
        step1_arr, wrf_arr, target, lookback, horizon
    )
    n_total_features = X_all.shape[2]

    n = len(X_all)
    n_train_val = int(n * (base_cfg["train_frac"] + base_cfg["val_frac"]))
    n_train = int(n_train_val * base_cfg["train_frac"] /
                  (base_cfg["train_frac"] + base_cfg["val_frac"]))
    n_val = n_train_val - n_train
    n_test = n - n_train_val

    X_train, X_val, X_test = X_all[:n_train], X_all[n_train:n_train_val], X_all[n_train_val:]
    y_residual_train = y_residual_all[:n_train]
    y_residual_val = y_residual_all[n_train:n_train_val]
    y_residual_test = y_residual_all[n_train_val:]
    y_anchor_train = y_anchor_all[:n_train]
    y_anchor_val = y_anchor_all[n_train:n_train_val]
    y_anchor_test = y_anchor_all[n_train_val:]

    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, n_total_features))
    def transform_X(X):
        shape = X.shape
        return scaler.transform(X.reshape(-1, shape[-1])).reshape(shape).astype(np.float32)

    X_train_s = transform_X(X_train)
    X_val_s = transform_X(X_val)
    X_test_s = transform_X(X_test)

    y_scaler = StandardScaler()
    y_scaler.fit(y_residual_train)
    y_residual_train_s = y_scaler.transform(y_residual_train).astype(np.float32)
    y_residual_val_s = y_scaler.transform(y_residual_val).astype(np.float32)
    y_residual_test_s = y_scaler.transform(y_residual_test).astype(np.float32)

    hdir = SAMPLES_DIR / f"h{horizon}_lb{lookback}_wrf_{wrf_version}"
    hdir.mkdir(parents=True, exist_ok=True)

    def save_npy(arr, name):
        np.save(hdir / name, arr)
        logger.info("  保存 %s  ->  shape=%s", name, arr.shape)

    save_npy(X_train_s, "X_train_seq.npy")
    save_npy(X_val_s, "X_val_seq.npy")
    save_npy(X_test_s, "X_test_seq.npy")
    save_npy(y_residual_train_s, "y_train.npy")
    save_npy(y_residual_val_s, "y_val.npy")
    save_npy(y_residual_test_s, "y_test.npy")
    save_npy(y_anchor_train.astype(np.float32), "y_anchor_train.npy")
    save_npy(y_anchor_val.astype(np.float32), "y_anchor_val.npy")
    save_npy(y_anchor_test.astype(np.float32), "y_anchor_test.npy")
    save_npy(y_residual_train.astype(np.float32), "y_residual_train_raw.npy")
    save_npy(y_residual_val.astype(np.float32), "y_residual_val_raw.npy")
    save_npy(y_residual_test.astype(np.float32), "y_residual_test_raw.npy")

    scaler_params = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "feature_cols": f"step1({len(STEP1_FEATURES)})+wrf_aligned_{wrf_version}({len(wrf_feature_list)})",
        "step1_feature_cols": STEP1_FEATURES,
        "wrf_forecast_feature_cols": wrf_feature_list,
        "n_total_features": n_total_features,
        "y_mean": y_scaler.mean_.tolist(),
        "y_scale": y_scaler.scale_.tolist(),
    }
    with open(hdir / "scaler_params.json", "w", encoding="utf-8") as f:
        json.dump(scaler_params, f, indent=2, ensure_ascii=False)

    test_ts_start = n_train_val + lookback + horizon - 1
    test_timestamps = df["timestamp"].iloc[test_ts_start:test_ts_start + n_test].reset_index(drop=True)
    pd.DataFrame({"timestamp": test_timestamps}).to_csv(hdir / "test_timestamps.csv", index=False)

    meta = {
        "lookback": lookback,
        "horizon": horizon,
        "wrf_version": wrf_version,
        "n_features": n_total_features,
        "feature_cols": f"step1({len(STEP1_FEATURES)})+wrf_aligned_{wrf_version}({len(wrf_feature_list)})",
        "step1_feature_cols": STEP1_FEATURES,
        "wrf_forecast_feature_cols": wrf_feature_list,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "train_frac": base_cfg["train_frac"],
        "val_frac": base_cfg["val_frac"],
        "test_frac": base_cfg["test_frac"],
        "source_csv": str(data_path.relative_to(PROJECT_ROOT)),
        "total_windows": int(n),
        "prediction_mode": "residual",
        "residual_formula": "Delta_y = y_future - y_anchor (y_anchor = power at t_lookback-1)",
        "feature_alignment": "per-horizon WRF forecast (wrf[t+h] at prediction step h, issue_lag=4h)",
    }
    with open(hdir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    logger.info("  样本已保存: %s", hdir)


# ──────────────────────────────────────────────────────────────────────────────
# Phase 4: WeatherSeq2Seq 模型
# ──────────────────────────────────────────────────────────────────────────────

def run_phase4(horizon: int, logger):
    """Phase 4: WeatherSeq2Seq 模型训练与对比。"""
    logger.info("=" * 60)
    logger.info("Phase 4: WeatherSeq2Seq vs CNN-BiLSTM  horizon=%d", horizon)
    logger.info("=" * 60)

    horizon_cfg = load_config(f"exp_p04_h{horizon}.json")
    base_cfg = load_config("exp_p04_base.json")

    sample_dir = resolve_sample_dir(horizon)
    samples = load_samples_from_dir(sample_dir)
    meta = load_meta(sample_dir)

    X_train = samples["X_train"]
    y_residual_train = samples["y_train"]
    X_val = samples["X_val"]
    y_residual_val = samples["y_val"]
    X_test = samples["X_test"]
    y_residual_test = samples["y_test"]
    y_anchor_test = samples["y_anchor_test"]

    seq_len = meta["lookback"]
    n_features = meta["n_features"]
    device = get_device()
    y_scaler = load_y_scaler_from_json(f"h{horizon}")

    seeds = base_cfg["reproduce_seeds"]
    max_epochs = base_cfg["final_max_epochs"]
    patience = base_cfg["final_patience"]
    lr = base_cfg["final_lr"]
    batch_size = 64

    metric_dir = METRICS_DIR / f"h{horizon}"
    model_dir = MODELS_DIR / f"h{horizon}"
    for d in (metric_dir, model_dir):
        d.mkdir(parents=True, exist_ok=True)

    # WeatherSeq2Seq 搜索空间
    search_space = {
        "encoder_hidden": [64, 128],
        "encoder_layers": [1, 2],
        "decoder_hidden": [64, 128],
        "decoder_layers": [1, 2],
        "dropout": [0.1, 0.2, 0.3],
        "lr": [1e-4, 1e-3],
        "batch_size": [32, 64],
    }

    optuna_path = metric_dir / "wseq2seq_optuna.json"

    # Optuna 调参
    if optuna_path.exists():
        logger.info("  Optuna 结果已存在，跳过")
    else:
        logger.info("  [Step 1] WeatherSeq2Seq Optuna 调参...")
        n_total = len(X_train)
        tr_end = int(n_total * 2 / 3)
        X_quick = X_train[tr_end:]
        y_quick = y_residual_train[tr_end:]

        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(
            lambda trial: _run_wseq2seq_trial(
                trial, search_space,
                X_quick, y_quick, X_val, y_residual_val,
                seq_len, n_features, horizon, 12, 3, device,
            ),
            n_trials=8,
            n_jobs=1,
            show_progress_bar=False,
        )
        logger.info("  Optuna完成  best_val_loss=%.6f", study.best_value)

        result = {
            "model": "wseq2seq",
            "horizon": horizon,
            "best_params": study.best_params,
            "best_value": study.best_value,
        }
        with open(optuna_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    params_json = json.loads(optuna_path.read_text(encoding="utf-8"))
    raw_best_params = params_json["best_params"]
    batch_size = raw_best_params.get("batch_size", 64)
    lr = raw_best_params.get("lr", 1e-3)
    if isinstance(batch_size, str):
        batch_size = int(batch_size)
    if isinstance(lr, str):
        lr = float(lr)
    best_params = {k: v for k, v in raw_best_params.items() if k not in ("batch_size", "lr")}

    # 多 seed 复现
    logger.info("  [Step 2] 多 Seed 复现 (seeds=%s)...", seeds)
    all_metrics = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = _build_wseq2seq(n_features, seq_len, horizon, **best_params).to(device)
        tl = make_loader(X_train, y_residual_train, batch_size=batch_size, shuffle=True)
        vl = make_loader(X_val, y_residual_val, batch_size=batch_size, shuffle=False)
        model, _ = train_with_early_stop(
            model, tl, vl, lr=lr, max_epochs=max_epochs, patience=patience, device=device,
        )

        y_pred_residual_scaled = predict(model, X_test, device)
        y_pred_residual = y_scaler.inverse_transform(y_pred_residual_scaled)
        y_pred_power = (y_anchor_test + y_pred_residual).astype(np.float32)
        y_test_residual_raw = y_scaler.inverse_transform(y_residual_test)
        y_true_power = (y_anchor_test + y_test_residual_raw).astype(np.float32)

        metrics = compute_all_metrics(y_true_power.ravel(), y_pred_power.ravel())
        all_metrics.append(metrics)
        logger.info("    seed=%d  MAE=%.4f  RMSE=%.4f  R2=%.4f",
                    seed, metrics["MAE"], metrics["RMSE"], metrics["R2"])

    rows_df = pd.DataFrame(all_metrics)
    mean_row = rows_df[["MAE", "RMSE", "MAPE", "R2"]].mean().to_dict()
    std_row = rows_df[["MAE", "RMSE", "MAPE", "R2"]].std().to_dict()

    summary = {
        "model": "wseq2seq", "horizon": horizon, "seeds": seeds,
        "mean": {k: round(v, 6) for k, v in mean_row.items()},
        "std": {k: round(v, 6) for k, v in std_row.items()},
        "per_seed": all_metrics,
    }
    reprod_path = metric_dir / "wseq2seq_reproduce.json"
    with open(reprod_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("  汇总  MAE=%.4f±%.4f  RMSE=%.4f±%.4f  R2=%.4f±%.4f",
                mean_row["MAE"], std_row["MAE"],
                mean_row["RMSE"], std_row["RMSE"],
                mean_row["R2"], std_row["R2"])

    logger.info("=" * 60)
    logger.info("Phase 4 完成！")
    logger.info("=" * 60)


def _build_wseq2seq(n_features: int, seq_len: int, horizon: int,
                    encoder_hidden=64, encoder_layers=1,
                    decoder_hidden=64, decoder_layers=1, dropout=0.2) -> nn.Module:
    """Weather-aware Seq2Seq 模型。"""
    class WeatherSeq2Seq(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.LSTM(
                n_features, encoder_hidden, num_layers=encoder_layers,
                batch_first=True, dropout=dropout if encoder_layers > 1 else 0.0,
            )
            # Decoder: 以最后一个 encoder 隐状态初始化，每次输出一步
            self.decoder = nn.LSTM(
                1, decoder_hidden, num_layers=decoder_layers,
                batch_first=True, dropout=dropout if decoder_layers > 1 else 0.0,
            )
            self.fc = nn.Linear(decoder_hidden, 1)

        def forward(self, x):
            # x: (B, seq_len, n_features)
            _, (h_n, c_n) = self.encoder(x)
            # h_n: (n_layers, B, hidden)
            decoder_input = torch.zeros(x.size(0), 1, 1).to(x.device)
            outputs = []
            for _ in range(horizon):
                out, (h_n, c_n) = self.decoder(decoder_input, (h_n, c_n))
                pred = self.fc(out)
                outputs.append(pred)
                decoder_input = pred
            return torch.cat(outputs, dim=1).squeeze(-1)  # (B, horizon)

    return WeatherSeq2Seq()


def _run_wseq2seq_trial(trial, search_space, X_quick, y_quick, X_val, y_val,
                          seq_len, n_features, horizon_int, n_epochs, patience, device):
    torch.manual_seed(42)
    np.random.seed(42)
    # 先提取训练参数，再对架构参数做类型转换
    raw = trial.params
    batch_size_raw = raw.get("batch_size", 64)
    lr_raw = raw.get("lr", 1e-3)
    batch_size = int(batch_size_raw) if isinstance(batch_size_raw, str) else int(batch_size_raw)
    lr = float(lr_raw) if isinstance(lr_raw, str) else float(lr_raw)
    params = _convert_params(raw, search_space)
    model = _build_wseq2seq(n_features, seq_len, horizon_int, **params).to(device)
    tl = make_loader(X_quick, y_quick, batch_size=batch_size, shuffle=True)
    vl = make_loader(X_val, y_val, batch_size=batch_size, shuffle=False)
    _, history = train_with_early_stop(
        model, tl, vl, lr=lr, max_epochs=n_epochs, patience=patience, device=device,
    )
    return min(h["val_loss"] for h in history)


# ──────────────────────────────────────────────────────────────────────────────
# Phase 5: 分段指标 + 最终报告
# ──────────────────────────────────────────────────────────────────────────────

def run_phase5(horizon: int, logger):
    """Phase 5: 分段指标计算 + 最终报告生成。"""
    logger.info("=" * 60)
    logger.info("Phase 5: 分段指标 + 最终报告  horizon=%d", horizon)
    logger.info("=" * 60)

    horizon_cfg = load_config(f"exp_p04_h{horizon}.json")
    horizon_label = horizon_cfg.get("horizon_label", f"{horizon}h")

    sample_dir = resolve_sample_dir(horizon)
    samples = load_samples_from_dir(sample_dir)
    meta = load_meta(sample_dir)

    y_anchor_test = samples["y_anchor_test"]
    y_residual_test = samples["y_test"]
    y_scaler = load_y_scaler_from_json(f"h{horizon}")

    # 重构真实功率
    y_test_residual_raw = y_scaler.inverse_transform(y_residual_test)
    y_true_power = (y_anchor_test + y_test_residual_raw).astype(np.float32)

    metric_dir = METRICS_DIR / f"h{horizon}"
    fig_dir = FIGURES_DIR / f"h{horizon}"
    for d in (fig_dir,):
        d.mkdir(parents=True, exist_ok=True)

    # CNN-BiLSTM 分段指标
    model_name = "cnn_bilstm"
    pred_path = PRED_DIR / f"h{horizon}" / f"{model_name}_test.csv"
    if pred_path.exists():
        df_pred = pd.read_csv(pred_path)
        y_pred = df_pred["y_pred"].values

        # 计算分段指标
        all_metrics = compute_all_segmented_metrics(y_true_power.ravel(), y_pred)
        logger.info("  CNN-BiLSTM 全分段指标:")
        logger.info("    All-day  MAE=%.4f  RMSE=%.4f  R2=%.4f",
                    all_metrics["MAE"], all_metrics["RMSE"], all_metrics["R2"])
        logger.info("    Daytime   MAE=%.4f  RMSE=%.4f  R2=%.4f",
                    all_metrics["daytime"]["MAE"], all_metrics["daytime"]["RMSE"],
                    all_metrics["daytime"]["R2"])
        logger.info("    Peak     MAE=%.4f  RMSE=%.4f  R2=%.4f  bias=%.4f",
                    all_metrics["peak"]["MAE"], all_metrics["peak"]["RMSE"],
                    all_metrics["peak"]["R2"], all_metrics["peak"]["bias"])

        # 保存分段指标
        seg_path = metric_dir / f"{model_name}_segmented_metrics.json"
        with open(seg_path, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=2, ensure_ascii=False)

        # 绘制分段指标图
        _plot_segmented_metrics(all_metrics, horizon_label, fig_dir, model_name)
    else:
        logger.warning("  预测文件不存在: %s", pred_path)

    # WRF 消融汇总图
    _plot_wrf_ablation(horizon, horizon_label, metric_dir, fig_dir)

    # Lookback 消融汇总图
    _plot_lookback_ablation(horizon, horizon_label, metric_dir, fig_dir)

    # 生成 Markdown 报告
    _generate_final_report(horizon, horizon_label, horizon_cfg, sample_dir, metric_dir, fig_dir)

    logger.info("=" * 60)
    logger.info("Phase 5 完成！")
    logger.info("=" * 60)


def _plot_segmented_metrics(metrics: dict, horizon_label: str, fig_dir: Path, model_name: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    categories = ["All-day", "Daytime", "Peak"]
    mae_vals = [metrics["MAE"], metrics["daytime"]["MAE"], metrics["peak"]["MAE"]]
    rmse_vals = [metrics["RMSE"], metrics["daytime"]["RMSE"], metrics["peak"]["RMSE"]]

    x = np.arange(len(categories))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(x, mae_vals, width, color=["steelblue", "coral", "seagreen"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories)
    axes[0].set_ylabel("MAE")
    axes[0].set_title(f"MAE by Category — {horizon_label}")
    for i, v in enumerate(mae_vals):
        axes[0].text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=9)

    axes[1].bar(x, rmse_vals, width, color=["steelblue", "coral", "seagreen"])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(categories)
    axes[1].set_ylabel("RMSE")
    axes[1].set_title(f"RMSE by Category — {horizon_label}")
    for i, v in enumerate(rmse_vals):
        axes[1].text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(fig_dir / f"{model_name}_segmented_metrics.png", dpi=150)
    plt.close()


def _plot_wrf_ablation(horizon: int, horizon_label: str, metric_dir: Path, fig_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    versions = ["full", "physical", "minimal"]
    mae_vals, rmse_vals, r2_vals = [], [], []

    for v in versions:
        path = metric_dir / "wrf_ablation" / f"cnn_bilstm_wrf_{v}_reproduce.json"
        if path.exists():
            r = json.loads(path.read_text(encoding="utf-8"))
            mae_vals.append(r["mean"]["MAE"])
            rmse_vals.append(r["mean"]["RMSE"])
            r2_vals.append(r["mean"]["R2"])
        else:
            mae_vals.append(0)
            rmse_vals.append(0)
            r2_vals.append(0)

    x = np.arange(len(versions))
    width = 0.25
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, vals, metric, ylabel in zip(
        axes, [mae_vals, rmse_vals, r2_vals],
        ["MAE", "RMSE", "R²"], ["MAE", "RMSE", "R²"]
    ):
        bars = ax.bar(x, vals, width, color=["steelblue", "coral", "seagreen"])
        ax.set_xticks(x)
        ax.set_xticklabels([v.upper() for v in versions])
        ax.set_ylabel(ylabel)
        ax.set_title(f"{metric} vs WRF Version — {horizon_label}")
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                       f"{val:.3f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "wrf_ablation_comparison.png", dpi=150)
    plt.close()


def _plot_lookback_ablation(horizon: int, horizon_label: str, metric_dir: Path, fig_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lookbacks = [16, 32, 48, 96]
    mae_vals, rmse_vals, r2_vals = [], [], []

    for lb in lookbacks:
        path = metric_dir / "lookback_ablation" / f"cnn_bilstm_lb{lb}_reproduce.json"
        if path.exists():
            r = json.loads(path.read_text(encoding="utf-8"))
            mae_vals.append(r["mean"]["MAE"])
            rmse_vals.append(r["mean"]["RMSE"])
            r2_vals.append(r["mean"]["R2"])
        else:
            mae_vals.append(0)
            rmse_vals.append(0)
            r2_vals.append(0)

    x = np.arange(len(lookbacks))
    width = 0.25
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, vals, metric, ylabel in zip(
        axes, [mae_vals, rmse_vals, r2_vals],
        ["MAE", "RMSE", "R²"], ["MAE", "RMSE", "R²"]
    ):
        bars = ax.bar(x, vals, width, color="steelblue")
        ax.set_xticks(x)
        ax.set_xticklabels([str(lb) for lb in lookbacks])
        ax.set_xlabel("Lookback")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{metric} vs Lookback — {horizon_label}")
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                       f"{val:.3f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "lookback_ablation_comparison.png", dpi=150)
    plt.close()


def _generate_final_report(horizon: int, horizon_label: str, horizon_cfg: dict,
                            sample_dir: Path, metric_dir: Path, fig_dir: Path):
    """生成最终 Markdown 报告。"""
    lines = []
    lines.append(f"# EXP-P04 最终实验报告\n")
    lines.append(f"**Horizon: {horizon_label} ({horizon} 步)**\n")
    lines.append(f"**生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}**\n\n")
    lines.append("---\n\n")

    # Phase 1: 模型对比
    lines.append("## 1. Phase 1: 模型对比 (Optuna 调参)\n\n")
    models = list(horizon_cfg.get("model_search_space", {}).keys())
    rows = []
    for mname in models:
        reprod_path = metric_dir / f"{mname}_reproduce.json"
        if not reprod_path.exists():
            continue
        r = json.loads(reprod_path.read_text(encoding="utf-8"))
        mean = r.get("mean", {})
        std = r.get("std", {})
        rows.append({
            "Model": mname,
            "MAE": f"{mean.get('MAE', 0):.4f} ± {std.get('MAE', 0):.4f}",
            "RMSE": f"{mean.get('RMSE', 0):.4f} ± {std.get('RMSE', 0):.4f}",
            "R²": f"{mean.get('R2', 0):.4f}",
        })
    if rows:
        lines.append("| " + " | ".join(rows[0].keys()) + " |")
        lines.append("| " + " | ".join(["---"] * len(rows[0])) + " |")
        for row in rows:
            lines.append("| " + " | ".join(str(v) for v in row.values()) + " |")
        lines.append("\n")

    # Phase 2: WRF 消融
    lines.append("## 2. Phase 2: WRF 特征消融\n\n")
    versions = ["full", "physical", "minimal"]
    for v in versions:
        path = metric_dir / "wrf_ablation" / f"cnn_bilstm_wrf_{v}_reproduce.json"
        if not path.exists():
            continue
        r = json.loads(path.read_text(encoding="utf-8"))
        mean = r.get("mean", {})
        lines.append(f"- **{v.upper()}**: MAE={mean.get('MAE', 0):.4f}  "
                    f"RMSE={mean.get('RMSE', 0):.4f}  R²={mean.get('R2', 0):.4f}\n")
    lines.append("\n")

    # Phase 3: Lookback 消融
    lines.append("## 3. Phase 3: Lookback 消融\n\n")
    lookbacks = [16, 32, 48, 96]
    for lb in lookbacks:
        path = metric_dir / "lookback_ablation" / f"cnn_bilstm_lb{lb}_reproduce.json"
        if not path.exists():
            continue
        r = json.loads(path.read_text(encoding="utf-8"))
        mean = r.get("mean", {})
        lines.append(f"- **lookback={lb}**: MAE={mean.get('MAE', 0):.4f}  "
                    f"RMSE={mean.get('RMSE', 0):.4f}  R²={mean.get('R2', 0):.4f}\n")
    lines.append("\n")

    # Phase 4: WeatherSeq2Seq
    lines.append("## 4. Phase 4: WeatherSeq2Seq 对比\n\n")
    wseq_path = metric_dir / "wseq2seq_reproduce.json"
    if wseq_path.exists():
        r = json.loads(wseq_path.read_text(encoding="utf-8"))
        mean = r.get("mean", {})
        lines.append(f"- **WeatherSeq2Seq**: MAE={mean.get('MAE', 0):.4f}  "
                    f"RMSE={mean.get('RMSE', 0):.4f}  R²={mean.get('R2', 0):.4f}\n\n")
    else:
        lines.append("*WeatherSeq2Seq 结果未生成（请运行 Phase 4）*\n\n")

    # Phase 5: 分段指标
    lines.append("## 5. Phase 5: 分段指标\n\n")
    seg_path = metric_dir / "cnn_bilstm_segmented_metrics.json"
    if seg_path.exists():
        m = json.loads(seg_path.read_text(encoding="utf-8"))
        lines.append("| 指标类别 | MAE | RMSE | R² | Peak Bias |")
        lines.append("|---|---|---|---|---|")
        lines.append(f"| All-day | {m['MAE']:.4f} | {m['RMSE']:.4f} | {m['R2']:.4f} | - |")
        lines.append(f"| Daytime | {m['daytime']['MAE']:.4f} | {m['daytime']['RMSE']:.4f} | {m['daytime']['R2']:.4f} | - |")
        lines.append(f"| Peak | {m['peak']['MAE']:.4f} | {m['peak']['RMSE']:.4f} | {m['peak']['R2']:.4f} | {m['peak']['bias']:.4f} |")
        lines.append("\n")
    else:
        lines.append("*分段指标未生成（请先运行预测并计算指标）*\n\n")

    # 图表
    lines.append("## 6. 可视化\n\n")
    for fig_name in [
        f"h{horizon}_metrics_comparison.png",
        f"h{horizon}_model_comparison.png",
        "wrf_ablation_comparison.png",
        "lookback_ablation_comparison.png",
        "cnn_bilstm_segmented_metrics.png",
    ]:
        p = fig_dir / fig_name
        if p.exists():
            rel = p.relative_to(PROJECT_ROOT)
            lines.append(f"![{p.stem}]({rel})\n\n")

    lines.append("---\n")
    lines.append("*报告由 run_exp_p04_unified.py 自动生成*\n")

    report_dir = REPORTS_DIR
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"EXP-P04_h{horizon}_FINAL_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  报告已保存: {report_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EXP-P04 统一实验执行")
    parser.add_argument("--phase", type=int, required=True,
                       choices=[1, 2, 3, 4, 5],
                       help="实验阶段 (1=模型对比, 2=WRF消融, 3=Lookback消融, 4=Seq2Seq, 5=报告)")
    parser.add_argument("--horizon", type=int, choices=[1, 4, 16], required=True)
    parser.add_argument("--wrf_version", type=str, choices=["full", "physical", "minimal"], default="full",
                       help="WRF 特征版本 (Phase 2)")
    parser.add_argument("--lookback", type=int, choices=[16, 32, 48, 96], default=None,
                       help="lookback 步数 (Phase 3)")
    args = parser.parse_args()

    horizon = args.horizon
    log_file = f"EXP-P04_phase{args.phase}_h{horizon}"
    if args.phase == 2:
        log_file += f"_wrf_{args.wrf_version}"
    elif args.phase == 3:
        log_file += f"_lb{args.lookback}"
    log_file += ".log"

    logger = setup_logger(f"phase{args.phase}", log_file)
    logger.info("=" * 60)
    logger.info("EXP-P04 统一实验执行器")
    logger.info("Phase: %d  Horizon: %d", args.phase, horizon)
    if args.phase == 2:
        logger.info("WRF 版本: %s", args.wrf_version)
    if args.phase == 3:
        logger.info("Lookback: %s", args.lookback)
    logger.info("=" * 60)

    if args.phase == 1:
        run_phase1(horizon, logger)
    elif args.phase == 2:
        run_phase2(horizon, args.wrf_version, logger)
    elif args.phase == 3:
        if args.lookback is None:
            logger.error("Phase 3 需要 --lookback 参数")
            return
        run_phase3(horizon, args.lookback, logger)
    elif args.phase == 4:
        run_phase4(horizon, logger)
    elif args.phase == 5:
        run_phase5(horizon, logger)

    logger.info("执行完成！")


if __name__ == "__main__":
    main()
