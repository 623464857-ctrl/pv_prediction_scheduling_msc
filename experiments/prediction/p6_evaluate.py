"""
预测 P6: 在验证集/测试集上预测, 计算指标, 画预测曲线

输入: result/prediction/samples/h{H}_lb{L}/, result/prediction/models/h{H}_{mode}_seed*.pt
输出: result/prediction/predictions/h{H}_{mode}_{val,test}.csv   列: timestamp(预测目标时刻), step, y_true, y_pred, y_persist
      result/prediction/predictions/h{H}_{mode}_metrics.json
      result/prediction/figures/h{H}_{mode}_test_curve.png        测试集全程 + 4 天放大

多个 seed 时取各模型预测的平均值 (集成)。指标单位为 power_pu (= kW / CAPACITY_KW)。
基准 "持续性预测" (y_persist): 直接把当前时刻功率当作未来的预测值。

python -m experiments.prediction.p6_evaluate                              # 默认 H=1, mode=hybrid
python -m experiments.prediction.p6_evaluate --horizon 4 --mode optuna     # H=4, optuna 模式
python -m experiments.prediction.p6_evaluate --seeds 42 43 44
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.common import PRED_FIGURES_DIR, PREDICTIONS_DIR, compute_metrics, sample_dir, setup_logger
from experiments.prediction.models import build_model
from experiments.prediction.p5_train import model_path, load_params_with_mode
from experiments.prediction.torch_utils import get_device, predict

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def predict_split(horizon: int, mode: str, seeds: list[int], split: str) -> tuple[pd.DataFrame, list[dict]]:
    d = sample_dir(horizon)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    X, y = np.load(d / f"X_{split}.npy"), np.load(d / f"y_{split}.npy")
    params, device = load_params_with_mode(horizon, mode), get_device()

    preds = []
    for seed in seeds:
        model = build_model(meta["n_features"], meta["lookback"], horizon, params)
        model.load_state_dict(torch.load(model_path(horizon, mode, seed), map_location=device, weights_only=True))
        preds.append(predict(model, X, device))
    y_pred = np.mean(preds, axis=0)
    per_seed = [{"seed": s, **compute_metrics(y, p)} for s, p in zip(seeds, preds)]

    # 持续性基准: 输入窗口最后一步的 power_pu_lag_0 (特征 0), 先反标准化
    sp = json.loads((d / "scaler_params.json").read_text(encoding="utf-8"))
    last_power = X[:, -1, 0] * sp["x_std"][0] + sp["x_mean"][0]

    # 展开成长表: 每个样本的第 k 步, 目标时刻 = 第 1 个预测点 + (k-1)*15min
    first_ts = pd.to_datetime(pd.read_csv(d / f"{split}_timestamps.csv")["timestamp"]).to_numpy()
    steps = np.tile(np.arange(1, horizon + 1), len(first_ts))
    return pd.DataFrame({
        "timestamp": np.repeat(first_ts, horizon) + (steps - 1) * np.timedelta64(15, "m"),
        "step": steps,
        "y_true": y.ravel(),
        "y_pred": y_pred.ravel(),
        "y_persist": np.repeat(last_power, horizon),
    }), per_seed


def plot_test(df: pd.DataFrame, horizon: int, mode: str, metrics: dict, out: Path) -> None:
    df = df[df["step"] == horizon]  # 每个目标时刻只画提前 H 步的那次预测
    fig, axes = plt.subplots(2, 1, figsize=(16, 8))
    zoom = df[df["timestamp"] < df["timestamp"].iloc[0] + pd.Timedelta(days=4)]
    for ax, part, title in ((axes[0], df, "测试集全程"), (axes[1], zoom, "前 4 天放大")):
        ax.plot(part["timestamp"], part["y_true"], color="black", lw=1.2, label="真实值")
        ax.plot(part["timestamp"], part["y_pred"], color="#2E86AB", lw=1.0, ls="--", label="CNN-BiLSTM 预测")
        ax.set_ylabel("功率 (p.u.)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")
    m = metrics["model"]
    fig.suptitle(f"亳州光伏 H={horizon} (提前 {15 * horizon} 分钟) | RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  R²={m['R2']:.4f}")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="评估 CNN-BiLSTM")
    parser.add_argument("--horizon", type=int, default=1, choices=[1, 4, 16])
    parser.add_argument("--mode", type=str, default="hybrid",
                        choices=["optuna", "afsa", "hybrid"],
                        help="超参数来源模式: optuna/afsa/hybrid")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    args = parser.parse_args()
    h = args.horizon
    m = args.mode
    logger = setup_logger(f"p6_evaluate_{m}", f"p6_evaluate_h{h}_{m}.log")

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    result = {"horizon": h, "mode": m, "seeds": args.seeds}
    for split in ("val", "test"):
        df, per_seed = predict_split(h, m, args.seeds, split)
        df.to_csv(PREDICTIONS_DIR / f"h{h}_{m}_{split}.csv", index=False)
        result[split] = {"model": compute_metrics(df["y_true"], df["y_pred"]),
                         "persistence": compute_metrics(df["y_true"], df["y_persist"])}
        if len(args.seeds) > 1:
            result[split]["per_seed"] = per_seed
            rmse = [r["RMSE"] for r in per_seed]
            for r in per_seed:
                logger.info("%-4s seed=%-5d   RMSE=%.4f MAE=%.4f R2=%.4f", split, r["seed"], r["RMSE"], r["MAE"], r["R2"])
            logger.info("%-4s 单模型均值    RMSE=%.4f ± %.4f", split, np.mean(rmse), np.std(rmse))
        for name in ("model", "persistence"):
            met = result[split][name]
            label = f"集成({len(args.seeds)}个)" if name == "model" and len(args.seeds) > 1 else name
            logger.info("%-4s %-11s RMSE=%.4f MAE=%.4f R2=%.4f", split, label, met["RMSE"], met["MAE"], met["R2"])
    test = result["test"]
    skill = 1 - test["model"]["RMSE"] / test["persistence"]["RMSE"]
    logger.info("测试集: 模型 RMSE 比持续性基准低 %.1f%%", skill * 100)

    metrics_file = PREDICTIONS_DIR / f"h{h}_{m}_metrics.json"
    fig_file = PREDICTIONS_DIR.parent / "figures" / f"h{h}_{m}_test_curve.png"
    metrics_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_test(df, h, m, test, fig_file)
    logger.info("预测结果: %s | 图: %s", PREDICTIONS_DIR / f"h{h}_{m}_test.csv", fig_file)


if __name__ == "__main__":
    main()
