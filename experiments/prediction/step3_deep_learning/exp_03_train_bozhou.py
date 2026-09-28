"""
亳州数据集模型训练脚本 (EXP-P05)
- 使用Optuna调优的最佳超参数训练CNN-BiLSTM模型
- 支持多预测步长 (H=1, H=4, H=16)

python -m experiments.prediction.step3_deep_learning.exp_03_train_bozhou
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
    BOZHOU_MODELS_DIR,
    BOZHOU_METRICS_DIR,
    load_bozhou_sample_dir,
    load_bozhou_y_scaler,
    load_best_params,
    setup_logger,
)
from experiments.prediction.step3_deep_learning.exp_03_models import build_model
from experiments.prediction.step3_deep_learning.exp_03_torch_utils import (
    get_device,
    make_loader,
    train_with_early_stop,
)


# =============================================================================
# 训练函数
# =============================================================================

def train_model(horizon: int, seed: int = 42) -> dict:
    """训练单个预测步长的模型"""
    print(f"\n{'='*60}")
    print(f"训练模型: horizon={horizon}, seed={seed}")
    print(f"{'='*60}")
    
    # 加载样本数据
    hdir = load_bozhou_sample_dir(horizon)
    X_train = np.load(hdir / "X_train_seq.npy")
    y_train = np.load(hdir / "y_train.npy")
    X_val = np.load(hdir / "X_val_seq.npy")
    y_val = np.load(hdir / "y_val.npy")
    
    # 加载元数据
    with open(hdir / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    seq_len = meta["lookback"]
    n_features = meta["n_features"]
    
    print(f"  数据: train={len(X_train)}, val={len(X_val)}")
    print(f"  seq_len={seq_len}, n_features={n_features}, horizon={horizon}")
    
    # 加载最佳参数
    best_params = load_best_params(horizon)
    print(f"  最佳参数: lr={best_params.get('lr')}, batch_size={best_params.get('batch_size')}")
    
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 构建模型
    device = get_device()
    print(f"  设备: {device}")
    
    model = build_model(
        "cnn_bilstm",
        n_features=n_features,
        seq_len=seq_len,
        horizon=horizon,
        **{k: v for k, v in best_params.items() if k not in ["lr", "batch_size"]}
    ).to(device)
    
    # 创建数据加载器
    batch_size = int(best_params.get("batch_size", 64))
    train_loader = make_loader(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, batch_size=batch_size, shuffle=False)
    
    # 训练模型
    print(f"  训练中 (max_epochs=20, patience=5)...")
    t0 = time.time()
    best_model, history = train_with_early_stop(
        model, train_loader, val_loader,
        lr=best_params.get("lr", 0.001),
        max_epochs=20, patience=5, device=device,
    )
    elapsed = time.time() - t0
    
    # 保存模型
    model_dir = BOZHOU_MODELS_DIR / f"bozhou_h{horizon}"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"bozhou_cnn_bilstm_seed{seed}.pt"
    torch.save(best_model.state_dict(), model_path)
    
    print(f"  模型已保存: {model_path}")
    print(f"  训练耗时: {elapsed:.1f}秒")
    
    return {
        "horizon": horizon,
        "seed": seed,
        "model_path": model_path,
        "history": history,
        "best_params": best_params,
        "elapsed": elapsed,
    }


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="亳州数据集模型训练")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 4, 16],
                        help="预测步长列表 (默认: 1 4 16)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认: 42)")
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logger(
        "train_bozhou",
        f"EXP-P05_bozhou_train_h{'_'.join(map(str, args.horizons))}.log"
    )
    
    logger.info("=" * 60)
    logger.info("亳州数据集模型训练")
    logger.info("Horizons: %s", args.horizons)
    logger.info("Seed: %d", args.seed)
    
    results = {}
    
    for horizon in args.horizons:
        t0 = time.time()
        try:
            res = train_model(horizon, seed=args.seed)
            results[horizon] = res
            elapsed = time.time() - t0
            logger.info("horizon=%d 完成 (%.1fs)", horizon, elapsed)
        except Exception as e:
            logger.error("horizon=%d 失败: %s", horizon, str(e))
            import traceback
            logger.error(traceback.format_exc())
    
    # 打印汇总
    print("\n" + "=" * 60)
    print("训练结果汇总:")
    print("=" * 60)
    for horizon, res in sorted(results.items()):
        print(f"  H={horizon:2d}: seed={res['seed']}, elapsed={res['elapsed']:.1f}s")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()
