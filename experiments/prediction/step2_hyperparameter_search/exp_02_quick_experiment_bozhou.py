"""
亳州数据集快速参数搜索脚本 (简化版)
只运行S2策略进行快速测试
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.prediction.step2_hyperparameter_search.exp_02_common import (
    METRICS_DIR,
    MODELS_DIR,
    compute_all_metrics,
    setup_logger,
    STEP4_ROOT,
)
from experiments.prediction.step2_hyperparameter_search.exp_02_hybrid_optimizer import run_strategy_s2
from experiments.prediction.step3_deep_learning.exp_03_models import build_model
from experiments.prediction.step3_deep_learning.exp_03_torch_utils import (
    get_device,
    make_loader,
    predict,
    train_with_early_stop,
)

# 配置
BOZHOU_CFG = {
    1: {"lookback": 16, "n_epochs": 15, "patience": 3},
    4: {"lookback": 48, "n_epochs": 15, "patience": 3},
    16: {"lookback": 96, "n_epochs": 15, "patience": 3},
}


def load_bozhou_data(horizon: int) -> dict:
    """加载亳州样本数据"""
    lookback = BOZHOU_CFG[horizon]["lookback"]
    lb_suffix = f"_lb{lookback}"
    hdir = STEP4_ROOT / "samples" / f"bozhou_h{horizon}{lb_suffix}"
    
    data = {
        "X_train": np.load(hdir / "X_train_seq.npy"),
        "y_train": np.load(hdir / "y_train.npy"),
        "X_val": np.load(hdir / "X_val_seq.npy"),
        "y_val": np.load(hdir / "y_val.npy"),
        "X_test": np.load(hdir / "X_test_seq.npy"),
        "y_test": np.load(hdir / "y_test.npy"),
    }
    
    with open(hdir / "meta.json", "r", encoding="utf-8") as f:
        data["meta"] = json.load(f)
    
    return data


def quick_search(horizon: int, n_iter: int = 5, seed: int = 42) -> dict:
    """快速参数搜索"""
    print(f"\n{'='*60}")
    print(f"亳州数据集快速搜索 horizon={horizon}")
    print(f"{'='*60}")
    
    data = load_bozhou_data(horizon)
    _, _, n_features = data["X_train"].shape
    seq_len = data["meta"]["lookback"]
    
    device = get_device()
    print(f"设备: {device}")
    print(f"训练样本: {len(data['X_train']):,}")
    print(f"特征数: {n_features}, 序列长度: {seq_len}")
    
    # 快速搜索空间
    search_space = {
        "lr": [0.001, 0.002],
        "batch_size": [64, 128],
        "hidden_size": [32, 64],
        "num_layers": [1, 2],
        "dropout": [0.1, 0.2],
        "cnn_channels_0": [32, 64],
        "cnn_channels_1": [64, 128],
        "kernel_size": [3, 5],
    }
    
    n_epochs = BOZHOU_CFG[horizon]["n_epochs"]
    patience = BOZHOU_CFG[horizon]["patience"]
    
    # 使用后1/3数据快速训练
    n_total = len(data["X_train"])
    tr_end = int(n_total * 2 / 3)
    X_quick = data["X_train"][tr_end:]
    y_quick = data["y_train"][tr_end:]
    
    print(f"\n快速训练: {len(X_quick):,} 样本")
    
    # 运行S2 (AFSA) 策略
    print(f"\n运行AFSA搜索 (n_iter={n_iter})...")
    t0 = time.time()
    result = run_strategy_s2(
        search_space, n_iter, seed,
        X_quick, y_quick, data["X_val"], data["y_val"],
        seq_len, n_features, horizon, n_epochs, patience, device
    )
    elapsed = time.time() - t0
    
    print(f"\n搜索完成! RMSE={result['rmse']:.6f}, 耗时={elapsed:.1f}秒")
    print(f"最佳参数: {result['params']}")
    
    return result


def train_final_model(horizon: int, params: dict) -> dict:
    """使用最佳参数训练最终模型"""
    print(f"\n{'='*60}")
    print(f"训练最终模型 horizon={horizon}")
    print(f"{'='*60}")
    
    data = load_bozhou_data(horizon)
    _, _, n_features = data["X_train"].shape
    seq_len = data["meta"]["lookback"]
    
    device = get_device()
    n_epochs = BOZHOU_CFG[horizon]["n_epochs"]
    patience = BOZHOU_CFG[horizon]["patience"]
    
    # 清理参数
    clean_params = {}
    cnn_channels = []
    for k, v in params.items():
        if k in ("batch_size",):
            clean_params[k] = int(v)
        elif k == "lr":
            clean_params[k] = float(v)
        elif k.startswith("cnn_channels_"):
            idx = int(k.split("_")[-1])
            while len(cnn_channels) <= idx:
                cnn_channels.append(None)
            cnn_channels.append(int(v))
        else:
            try:
                clean_params[k] = int(v)
            except:
                clean_params[k] = v
    clean_params["cnn_channels"] = cnn_channels
    
    print(f"模型参数: {clean_params}")
    
    # 构建模型
    model = build_model(
        "cnn_bilstm",
        n_features=n_features,
        seq_len=seq_len,
        horizon=horizon,
        **clean_params
    ).to(device)
    
    # 训练
    batch_size = int(params.get("batch_size", 64))
    lr = float(params.get("lr", 0.001))
    
    train_loader = make_loader(data["X_train"], data["y_train"], batch_size=batch_size, shuffle=True)
    val_loader = make_loader(data["X_val"], data["y_val"], batch_size=batch_size, shuffle=False)
    
    print(f"\n训练模型 (epochs={n_epochs}, patience={patience})...")
    t0 = time.time()
    _, history = train_with_early_stop(
        model, train_loader, val_loader,
        lr=lr, max_epochs=n_epochs, patience=patience, device=device
    )
    train_time = time.time() - t0
    
    # 评估
    y_pred_scaled = predict(model, data["X_test"], device)
    y_true = data["y_test"]
    
    # 反归一化
    scaler_params = json.load(open(STEP4_ROOT / "samples" / f"bozhou_h{horizon}_lb{BOZHOU_CFG[horizon]['lookback']}" / "scaler_params.json"))
    y_mean = np.array(scaler_params["y_mean"])
    y_scale = np.array(scaler_params["y_scale"])
    
    y_pred = y_pred_scaled * y_scale + y_mean
    y_true_orig = y_true * y_scale + y_mean
    
    metrics = compute_all_metrics(y_true_orig.ravel(), y_pred.ravel())
    
    print(f"\n训练完成! 耗时={train_time:.1f}秒")
    print(f"\n测试集指标:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="亳州数据集快速参数搜索")
    parser.add_argument("--horizon", type=int, choices=[1, 4, 16], required=True,
                        help="预测步长")
    parser.add_argument("--n_iter", type=int, default=5,
                        help="AFSA迭代次数 (默认: 5)")
    parser.add_argument("--skip_train", action="store_true",
                        help="跳过最终训练")
    args = parser.parse_args()
    
    t0 = time.time()
    
    # 快速搜索
    result = quick_search(args.horizon, n_iter=args.n_iter)
    
    # 训练最终模型
    if not args.skip_train:
        metrics = train_final_model(args.horizon, result["params"])
    else:
        metrics = {}
    
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"全部完成! 总耗时: {elapsed:.1f}秒")
    print(f"{'='*60}")
    
    return result, metrics


if __name__ == "__main__":
    main()
