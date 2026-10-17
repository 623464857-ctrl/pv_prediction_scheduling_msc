"""
预测 P4: 超参数搜索 —— 支持三种模式

模式说明:
  optuna   : 单 Optuna TPE 搜索 (粗搜, 覆盖面广)
  afsa     : 单 AFSA 鱼群算法搜索 (从随机种群开始精搜)
  hybrid   : Optuna-TPE 粗搜 + AFSA 精搜 (默认, 效果最好)

评价标准: 验证集 MSE (只看验证集, 测试集完全不参与, 避免"用测试集调参")。

输入: result/prediction/samples/h{H}_lb{L}/
输出: result/prediction/params/best_params_h{H}_{mode}.json
      result/prediction/params/search_history_h{H}_{mode}.csv
      result/prediction/figures/h{H}_search_curve_{mode}.png

python -m experiments.prediction.p4_search_hparams                           # 默认 H=1, optuna 模式
python -m experiments.prediction.p4_search_hparams --horizon 1 --mode optuna --trials 30
python -m experiments.prediction.p4_search_hparams --horizon 4 --mode afsa   --evals 20
python -m experiments.prediction.p4_search_hparams --horizon 16 --mode hybrid
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.common import (DEFAULT_PARAMS, PARAMS_DIR, PRED_FIGURES_DIR, params_path,
                                sample_dir, set_seed, setup_logger)
from experiments.prediction.afsa import AFSA
from experiments.prediction.models import build_model
from experiments.prediction.torch_utils import get_device, make_loader, train_with_early_stop

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# 搜索空间 (离散候选值)
SEARCH_SPACE = {
    "lr":           [0.0005, 0.001, 0.002, 0.005],
    "batch_size":   [32, 64, 128],
    "hidden_size":  [32, 64, 128],
    "num_layers":   [1, 2, 3],
    "dropout":      [0.1, 0.2, 0.3],
    "cnn_channels": ["16,32", "32,64", "64,128"],
    "kernel_size":  [3, 5, 7],
}
TRIAL_EPOCHS = 12    # 每次试验最多训练 12 个 epoch (正式训练是 20)
TRIAL_PATIENCE = 3
SEED = 42

# 精简搜索次数
DEFAULT_OPTUNA_TRIALS = 15  # 精简: 20 → 15
DEFAULT_AFSA_EVALS = 10     # 精简: 15 → 10


def to_model_params(p: dict) -> dict:
    return {**p, "cnn_channels": [int(c) for c in str(p["cnn_channels"]).split(",")]}


def to_search_params(p: dict) -> dict:
    return {**p, "cnn_channels": ",".join(str(c) for c in p["cnn_channels"])}


class Objective:
    """训练一次模型并返回验证集最小 MSE; 结果按超参数缓存, 同一组合不重复训练。"""

    def __init__(self, horizon: int, logger, pbar):
        d = sample_dir(horizon)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        self.X_train, self.y_train = np.load(d / "X_train.npy"), np.load(d / "y_train.npy")
        self.X_val, self.y_val = np.load(d / "X_val.npy"), np.load(d / "y_val.npy")
        self.n_features, self.lookback, self.horizon = meta["n_features"], meta["lookback"], horizon
        self.device, self.logger, self.pbar = get_device(), logger, pbar
        self.cache: dict[str, float] = {}
        self.history: list[dict] = []
        self.stage = ""

    def n_new(self, stage: str) -> int:
        return sum(1 for h in self.history if h["stage"] == stage)

    def __call__(self, p: dict) -> float:
        key = json.dumps(p, sort_keys=True)
        if key in self.cache:
            return self.cache[key]

        set_seed(SEED)
        mp = to_model_params(p)
        model = build_model(self.n_features, self.lookback, self.horizon, mp)
        bs = int(mp["batch_size"])
        t0 = time.time()
        _, hist = train_with_early_stop(
            model,
            make_loader(self.X_train, self.y_train, bs, shuffle=True),
            make_loader(self.X_val, self.y_val, bs, shuffle=False),
            lr=float(mp["lr"]), max_epochs=TRIAL_EPOCHS, patience=TRIAL_PATIENCE, device=self.device,
        )
        val = min(h["val_loss"] for h in hist)
        self.cache[key] = val

        best = min([h["val_loss"] for h in self.history] + [val])
        self.history.append({"eval": len(self.history) + 1, "stage": self.stage, **p,
                             "val_loss": val, "epochs": len(hist), "sec": round(time.time() - t0, 1)})
        self.pbar.update(1)
        self.pbar.set_postfix(stage=self.stage, best=f"{best:.5f}")
        tqdm.write(f"[{len(self.history):>2}] {self.stage:<6} val_loss={val:.5f} (best {best:.5f}) "
                   f"{time.time() - t0:4.0f}s  {p}")
        return val


def run_optuna(objective: Objective, n_trials: int) -> list[dict]:
    """纯 Optuna 搜索, 返回历史记录中 Optuna 阶段的最好几组。"""
    objective.stage = "optuna"
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    default = to_search_params(DEFAULT_PARAMS)
    study.enqueue_trial(default)
    study.optimize(
        lambda t: objective({k: t.suggest_categorical(k, v) for k, v in SEARCH_SPACE.items()}),
        n_trials=n_trials
    )
    return sorted(objective.history, key=lambda r: r["val_loss"])[:5]


def run_afsa_from_init(objective: Objective, init: list[dict], n_evals: int, pbar: tqdm):
    """AFSA 精搜, 以 init 中的超参数为初始鱼群。"""
    objective.stage = "afsa"
    afsa = AFSA(SEARCH_SPACE, objective, n_fish=len(init), seed=SEED)
    afsa.run(init, n_iter=100, stop=lambda: objective.n_new("afsa") >= n_evals)


def run_afsa_standalone(objective: Objective, n_fish: int, n_iter: int, pbar: tqdm):
    """纯 AFSA 搜索, 从随机初始种群开始。"""
    objective.stage = "afsa"
    # 随机生成初始鱼群
    rng = np.random.default_rng(SEED)
    init = []
    for _ in range(n_fish):
        fish = {}
        for k, v in SEARCH_SPACE.items():
            fish[k] = rng.choice(v)
        init.append(fish)
    # 同时加入默认参数作为第一条鱼
    init[0] = to_search_params(DEFAULT_PARAMS)
    
    afsa = AFSA(SEARCH_SPACE, objective, n_fish=len(init), seed=SEED)
    afsa.run(init, n_iter=n_iter, stop=lambda: objective.n_new("afsa") >= n_fish * n_iter)


def plot_curve(hist: pd.DataFrame, out: Path, mode: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    
    # 根据模式选择颜色
    if mode == "optuna":
        color = "#2E86AB"
        label = "Optuna 单次试验"
    elif mode == "afsa":
        color = "#F18F01"
        label = "AFSA 单次试验"
    else:  # hybrid
        colors = {"optuna": "#2E86AB", "afsa": "#F18F01"}
        for stage, col in colors.items():
            part = hist[hist["stage"] == stage]
            if not part.empty:
                ax.scatter(part["eval"], part["val_loss"], s=22, color=col, label=f"{stage} 单次试验")
        color = "black"
        label = "目前最优"
    
    if mode != "hybrid":
        ax.scatter(hist["eval"], hist["val_loss"], s=22, color=color, label=label)
    
    ax.plot(hist["eval"], hist["val_loss"].cummin(), color=color, lw=1.5, label="目前最优")
    ax.set_xlabel("试验次数")
    ax.set_ylabel("验证集 MSE")
    
    mode_names = {"optuna": "Optuna-TPE", "afsa": "AFSA", "hybrid": "Optuna-TPE → AFSA"}
    ax.set_title(f"超参数搜索收敛曲线 ({mode_names[mode]})")
    ax.set_ylim(top=min(hist["val_loss"].quantile(0.9) * 1.2, hist["val_loss"].max()))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def save_results(objective: Objective, horizon: int, mode: str, total_trials: int):
    """保存搜索结果到文件。"""
    hist = pd.DataFrame(objective.history)
    best = hist.loc[hist["val_loss"].idxmin()]
    best_params = to_model_params({k: best[k] for k in SEARCH_SPACE})
    best_params = {k: (v.item() if hasattr(v, "item") else v) for k, v in best_params.items()}
    default_loss = objective.cache[json.dumps(to_search_params(DEFAULT_PARAMS), sort_keys=True)]

    PARAMS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 模式特定的文件名
    hist_file = PARAMS_DIR / f"search_history_h{horizon}_{mode}.csv"
    params_file = PARAMS_DIR / f"best_params_h{horizon}_{mode}.json"
    fig_file = PRED_FIGURES_DIR / f"h{horizon}_search_curve_{mode}.png"
    
    hist.to_csv(hist_file, index=False)
    
    result = {
        "horizon": horizon,
        "mode": mode,
        "params": best_params,
        "val_loss": float(best["val_loss"]),
        "found_by": best["stage"],
        "default_params": DEFAULT_PARAMS,
        "default_val_loss": default_loss,
        "n_evals": total_trials,
        "trial_epochs": TRIAL_EPOCHS,
    }
    params_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    
    plot_curve(hist, fig_file, mode)
    
    return best, best_params, default_loss


def main():
    parser = argparse.ArgumentParser(description="超参数搜索 (Optuna / AFSA / Hybrid)")
    parser.add_argument("--horizon", type=int, default=1, choices=[1, 4, 16])
    parser.add_argument("--mode", type=str, default="optuna",
                        choices=["optuna", "afsa", "hybrid"],
                        help="搜索模式: optuna=单Optuna, afsa=单AFSA, hybrid=Optuna+AFSA")
    parser.add_argument("--trials", type=int, default=DEFAULT_OPTUNA_TRIALS,
                        help="Optuna 试验次数 (mode=optuna 或 hybrid)")
    parser.add_argument("--evals", type=int, default=DEFAULT_AFSA_EVALS,
                        help="AFSA 评估次数 (mode=afsa 或 hybrid)")
    parser.add_argument("--afsa-fish", type=int, default=5,
                        help="AFSA 鱼群数量 (mode=afsa)")
    parser.add_argument("--afsa-iter", type=int, default=20,
                        help="AFSA 最大迭代次数 (mode=afsa)")
    args = parser.parse_args()
    
    h = args.horizon
    mode = args.mode
    logger = setup_logger(f"p4_search_{mode}", f"p4_search_h{h}_{mode}.log")
    
    # 根据模式设置进度条总数
    if mode == "optuna":
        total = args.trials
        afsa_evals = 0
    elif mode == "afsa":
        total = args.afsa_fish * args.afsa_iter
        args.evals = total  # AFSA standalone 使用全部预算
    else:  # hybrid
        total = args.trials + args.evals
        afsa_evals = args.evals
    
    logger.info("=" * 60)
    logger.info("H=%d 超参数搜索 | 模式: %s", h, mode.upper())
    logger.info("=" * 60)
    
    pbar = tqdm(total=total, desc=f"搜索进度 ({mode})", unit="次", dynamic_ncols=True)
    objective = Objective(h, logger, pbar)

    if mode == "optuna":
        # ---- 模式 1: 单 Optuna ------------------------------------------------
        logger.info("运行单 Optuna TPE 搜索, %d 次试验", args.trials)
        run_optuna(objective, args.trials)
        total_trials = objective.n_new("optuna")
        
    elif mode == "afsa":
        # ---- 模式 2: 单 AFSA (随机初始种群) ------------------------------------
        logger.info("运行单 AFSA 搜索, %d 条鱼 x %d 迭代 = %d 次评估", 
                    args.afsa_fish, args.afsa_iter, total)
        run_afsa_standalone(objective, args.afsa_fish, args.afsa_iter, pbar)
        total_trials = objective.n_new("afsa")
        
    else:  # hybrid
        # ---- 模式 3: Optuna + AFSA 混合 ----------------------------------------
        logger.info("运行 Optuna + AFSA 混合搜索: Optuna %d 次 + AFSA 最多 %d 次", 
                    args.trials, args.evals)
        
        # 阶段 1: Optuna
        top = run_optuna(objective, args.trials)
        logger.info("Optuna 阶段完成, 最优 val_loss=%.5f", top[0]["val_loss"])
        
        # 阶段 2: AFSA
        run_afsa_from_init(objective, top, afsa_evals, pbar)
        total_trials = objective.n_new("optuna") + objective.n_new("afsa")

    pbar.close()

    # ---- 保存结果 -------------------------------------------------------------
    best, best_params, default_loss = save_results(objective, h, mode, total_trials)
    
    logger.info("-" * 60)
    logger.info("搜索完成! 模式: %s, 总评估次数: %d", mode.upper(), total_trials)
    logger.info("默认超参数 val_loss=%.5f", default_loss)
    logger.info("最优超参数 val_loss=%.5f (由 %s 找到, 改善 %.1f%%)",
                best["val_loss"], best["stage"],
                (1 - best["val_loss"] / default_loss) * 100)
    logger.info("最优参数: %s", best_params)
    logger.info("结果已保存:")
    logger.info("  - 历史记录: %s", PARAMS_DIR / f"search_history_h{h}_{mode}.csv")
    logger.info("  - 最优参数: %s", PARAMS_DIR / f"best_params_h{h}_{mode}.json")
    logger.info("  - 收敛曲线: %s", PRED_FIGURES_DIR / f"h{h}_search_curve_{mode}.png")


if __name__ == "__main__":
    main()
