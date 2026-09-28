"""
亳州数据集预测与评估脚本 (EXP-P05)
- 加载训练好的模型进行预测
- 计算评估指标 (RMSE, MAE, R², MAPE)
- 绘制预测曲线图

python -m experiments.prediction.step4_evaluation.exp_04_evaluate_bozhou
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.prediction.step2_hyperparameter_search.exp_02_common import (
    BOZHOU_FIGURES_DIR,
    BOZHOU_METRICS_DIR,
    BOZHOU_MODELS_DIR,
    BOZHOU_PRED_DIR,
    STEP4_ROOT,
    compute_all_metrics,
    load_bozhou_sample_dir,
    load_bozhou_y_scaler,
    setup_logger,
)
from experiments.prediction.step3_deep_learning.exp_03_models import build_model
from experiments.prediction.step3_deep_learning.exp_03_torch_utils import (
    get_device,
    predict,
)


# =============================================================================
# 预测函数
# =============================================================================

def predict_horizon(horizon: int, seed: int = 42) -> dict:
    """对单个预测步长进行预测"""
    print(f"\n{'='*60}")
    print(f"预测: horizon={horizon}")
    print(f"{'='*60}")
    
    # 加载样本数据
    hdir = load_bozhou_sample_dir(horizon)
    X_test = np.load(hdir / "X_test_seq.npy")
    y_test = np.load(hdir / "y_test.npy")
    
    # 加载元数据
    with open(hdir / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    seq_len = meta["lookback"]
    n_features = meta["n_features"]
    
    # 加载模型
    device = get_device()
    print(f"  设备: {device}")
    
    model_dir = BOZHOU_MODELS_DIR / f"bozhou_h{horizon}"
    model_path = model_dir / f"bozhou_cnn_bilstm_seed{seed}.pt"
    
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 加载 y scaler
    y_scaler = load_bozhou_y_scaler(horizon)
    
    # 构建并加载模型
    model = build_model(
        "cnn_bilstm",
        n_features=n_features,
        seq_len=seq_len,
        horizon=horizon,
        hidden_size=64, num_layers=2, dropout=0.2,
        cnn_channels=[32, 64], kernel_size=3,
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    # 生成预测
    print(f"  生成预测中...")
    y_pred_scaled = predict(model, X_test, device)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_true = y_scaler.inverse_transform(y_test)
    
    # 计算指标
    metrics = compute_all_metrics(y_true.ravel(), y_pred.ravel())
    print(f"  测试集指标: RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}, R²={metrics['R2']:.4f}")
    
    # 保存预测结果
    pred_dir = BOZHOU_PRED_DIR / f"bozhou_h{horizon}"
    pred_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理时间戳
    test_timestamps = pd.read_csv(hdir / "test_timestamps.csv")
    test_timestamps["timestamp"] = pd.to_datetime(test_timestamps["timestamp"])
    
    # horizon > 1 时展平时间戳
    if horizon > 1:
        n_samples = len(y_true)
        n_repeat = horizon
        ts_flat = np.repeat(test_timestamps["timestamp"].values, n_repeat)[:n_samples]
    else:
        ts_flat = test_timestamps["timestamp"].values
    
    # 创建预测 DataFrame
    pred_df = pd.DataFrame({
        "timestamp": ts_flat,
        "y_true": y_true.ravel(),
        "y_pred": y_pred.ravel(),
    })
    pred_df.to_csv(pred_dir / "predictions.csv", index=False)
    print(f"  预测结果已保存: {pred_dir / 'predictions.csv'}")
    
    # 返回结果
    return {
        "horizon": horizon,
        "y_true": y_true,
        "y_pred": y_pred,
        "timestamps": test_timestamps,
        "metrics": metrics,
        "seq_len": seq_len,
    }


# =============================================================================
# 绘图函数
# =============================================================================

def plot_bozhou_predictions(results: dict, output_path: Path = None):
    """绘制三图上下排列的亳州预测曲线
    
    要求：
    - 预测曲线：彩色虚线
    - 真实曲线：黑色实线
    - 3个H共用横轴，上下三图排列
    """
    
    fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
    plt.subplots_adjust(hspace=0.15)
    
    horizon_labels = {1: "H=1 (15分钟预测)", 4: "H=4 (1小时预测)", 16: "H=16 (4小时预测)"}
    colors = {1: "#2E86AB", 4: "#A23B72", 16: "#F18F01"}  # 蓝、紫、橙
    
    for idx, (horizon, res) in enumerate(sorted(results.items())):
        ax = axes[idx]
        
        y_true = res["y_true"]
        y_pred = res["y_pred"]
        timestamps = res["timestamps"]
        metrics = res["metrics"]
        
        # 处理时间戳 - horizon > 1 时需要重复
        if horizon > 1:
            n_samples = len(y_true)
            n_repeat = horizon
            ts_vals = np.repeat(timestamps["timestamp"].values, n_repeat)
            ts_vals = ts_vals[:n_samples]
        else:
            ts_vals = timestamps["timestamp"].values
        
        # 转换时间戳为 datetime 格式
        ts_dates = pd.to_datetime(ts_vals)
        
        # 展平数组
        y_true_flat = y_true.ravel()
        y_pred_flat = y_pred.ravel()
        
        # 绘制真实值（黑色实线）和预测值（彩色虚线）
        ax.plot(ts_dates, y_true_flat, 
                label="真实值 (Actual)", 
                color="black", 
                linewidth=1.5, 
                linestyle='-',     # 实线
                alpha=0.95)
        ax.plot(ts_dates, y_pred_flat, 
                label="预测值 (Predicted)", 
                color=colors[horizon], 
                linewidth=1.2, 
                linestyle='--',    # 虚线
                alpha=0.85)
        
        # 设置标签和标题
        ax.set_ylabel("功率 (kW)", fontsize=12, fontweight='bold')
        ax.set_title(f"亳州光伏预测 - {horizon_labels[horizon]}  |  "
                    f"RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}, R²={metrics['R2']:.4f}",
                    fontsize=13, fontweight="bold", pad=10)
        
        # 图例
        ax.legend(loc="upper right", fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 美化x轴
        ax.tick_params(axis='x', rotation=30, labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        
        # 设置y轴范围稍微扩展
        y_min = min(y_true_flat.min(), y_pred_flat.min())
        y_max = max(y_true_flat.max(), y_pred_flat.max())
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)
    
    # 最后一个图添加x轴标签
    axes[-1].set_xlabel("时间 (Time)", fontsize=12, fontweight='bold')
    
    # 整体标题
    fig.suptitle("亳州光伏电站功率预测 - 多预测步长对比\nBozhou PV Station Power Prediction - Multi-Horizon Comparison", 
                 fontsize=15, fontweight='bold', y=0.98)
    
    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # 保存图片
    if output_path is None:
        output_path = BOZHOU_FIGURES_DIR / "bozhou_all_horizons_prediction_curve.png"
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n图片已保存: {output_path}")
    
    # 同时保存PDF版本
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"PDF已保存: {pdf_path}")
    
    plt.close()


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="亳州数据集预测与评估")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 4, 16],
                        help="预测步长 (默认: 1 4 16)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认: 42)")
    args = parser.parse_args()
    
    # 设置日志
    log_file = "EXP-P05_bozhou_evaluate.log"
    logger = setup_logger("evaluate_bozhou", log_file)
    
    logger.info("=" * 60)
    logger.info("亳州数据集预测与评估")
    logger.info("Horizons: %s", args.horizons)
    logger.info("Seed: %d", args.seed)
    
    results = {}
    
    # 预测
    for horizon in args.horizons:
        try:
            res = predict_horizon(horizon, seed=args.seed)
            results[horizon] = res
            logger.info("horizon=%d 完成: RMSE=%.4f, MAE=%.4f, R2=%.4f",
                       horizon, res["metrics"]["RMSE"], res["metrics"]["MAE"],
                       res["metrics"]["R2"])
        except Exception as e:
            logger.error("horizon=%d 失败: %s", horizon, str(e))
            import traceback
            logger.error(traceback.format_exc())
    
    # 绘图
    if results:
        logger.info("开始绘图...")
        plot_bozhou_predictions(results)
        logger.info("绘图完成")
    
    # 打印汇总
    print("\n" + "=" * 60)
    print("预测结果汇总:")
    print("=" * 60)
    for horizon, res in sorted(results.items()):
        m = res["metrics"]
        print(f"  H={horizon:2d}: RMSE={m['RMSE']:.4f}, MAE={m['MAE']:.4f}, R²={m['R2']:.4f}")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()
