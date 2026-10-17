"""
预测 P5: 用最优超参数训练 CNN-BiLSTM

输入: result/prediction/samples/h{H}_lb{L}/  (P3 输出)
      result/prediction/params/best_params_h{H}_{mode}.json  (P4 输出; 不存在时用 common.DEFAULT_PARAMS)
输出: result/prediction/models/h{H}_{mode}_seed{seed}.pt
      result/prediction/models/h{H}_{mode}_seed{seed}_history.json  (每个 epoch 的损失)

python -m experiments.prediction.p5_train                    # 默认 H=1, mode=hybrid, seed=42
python -m experiments.prediction.p5_train --horizon 1 --mode optuna --seeds 42 43 44
python -m experiments.prediction.p5_train --horizon 4 --mode afsa
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.common import MODELS_DIR, PARAMS_DIR, DEFAULT_PARAMS, sample_dir, set_seed, setup_logger
from experiments.prediction.models import build_model
from experiments.prediction.torch_utils import get_device, make_loader, train_with_early_stop

MAX_EPOCHS = 20
PATIENCE = 5


def model_path(horizon: int, mode: str, seed: int) -> Path:
    return MODELS_DIR / f"h{horizon}_{mode}_seed{seed}.pt"


def search_params_path(horizon: int, mode: str) -> Path:
    """根据模式和预测步长返回超参数文件路径。"""
    return PARAMS_DIR / f"best_params_h{horizon}_{mode}.json"


def load_params_with_mode(horizon: int, mode: str) -> dict:
    """读取超参数: 优先用指定模式的搜索结果, 否则用 DEFAULT_PARAMS。"""
    path = search_params_path(horizon, mode)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["params"]
    return dict(DEFAULT_PARAMS)


def train(horizon: int, mode: str, seed: int, logger) -> dict:
    d = sample_dir(horizon)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    X_train, y_train = np.load(d / "X_train.npy"), np.load(d / "y_train.npy")
    X_val, y_val = np.load(d / "X_val.npy"), np.load(d / "y_val.npy")
    params = load_params_with_mode(horizon, mode)
    params_file = search_params_path(horizon, mode)
    
    logger.info("H=%d mode=%s seed=%d | train=%d val=%d | 超参数来源: %s | %s",
                horizon, mode, seed, len(X_train), len(X_val),
                params_file.name if params_file.exists() else "DEFAULT_PARAMS", params)

    set_seed(seed)
    model = build_model(meta["n_features"], meta["lookback"], horizon, params)
    bs = int(params["batch_size"])
    t0 = time.time()
    model, history = train_with_early_stop(
        model,
        make_loader(X_train, y_train, bs, shuffle=True),
        make_loader(X_val, y_val, bs, shuffle=False),
        lr=float(params["lr"]), max_epochs=MAX_EPOCHS, patience=PATIENCE,
        device=get_device(), progress=True,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = model_path(horizon, mode, seed)
    torch.save(model.state_dict(), path)
    path.with_name(path.stem + "_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    best = min(h["val_loss"] for h in history)
    logger.info("H=%d mode=%s seed=%d 完成: 最优 val_loss=%.5f, epochs=%d, 耗时 %.0fs -> %s",
                horizon, mode, seed, best, len(history), time.time() - t0, path.name)
    return {"mode": mode, "seed": seed, "best_val_loss": best}


def main():
    parser = argparse.ArgumentParser(description="训练 CNN-BiLSTM")
    parser.add_argument("--horizon", type=int, default=1, choices=[1, 4, 16])
    parser.add_argument("--mode", type=str, default="hybrid",
                        choices=["optuna", "afsa", "hybrid"],
                        help="超参数来源模式: optuna/afsa/hybrid")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    args = parser.parse_args()
    logger = setup_logger(f"p5_train_{args.mode}", f"p5_train_h{args.horizon}_{args.mode}.log")
    for seed in args.seeds:
        train(args.horizon, args.mode, seed, logger)


if __name__ == "__main__":
    main()
