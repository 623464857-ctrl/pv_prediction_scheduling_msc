"""
亳州光伏预测结果绘图脚本

从已保存的预测结果 CSV 文件绘制多预测步长对比图。
使用方法:
    python plot_bozhou_predictions.py
    
可选参数:
    --data-dir   指定预测结果目录 (默认: data/prediction/step2_hyperparameter_search/predictions)
    --output     指定输出图片路径
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 路径配置
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREDICTIONS_DIR = PROJECT_ROOT / "data" / "prediction" / "step2_hyperparameter_search" / "predictions"
OUTPUT_DIR = PROJECT_ROOT / "result" / "figures"


def load_predictions(horizon: int, predictions_dir: Path) -> tuple:
    """加载指定 horizon 的预测结果"""
    pred_file = predictions_dir / f"bozhou_h{horizon}" / "predictions.csv"
    
    if not pred_file.exists():
        raise FileNotFoundError(f"预测文件不存在: {pred_file}")
    
    df = pd.read_csv(pred_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    return df


def plot_predictions(horizons: list, predictions_dir: Path, output_path: Path = None):
    """绘制三图上下排列的亳州预测曲线
    
    Args:
        horizons: 预测步长列表 [1, 4, 16]
        predictions_dir: 预测结果目录
        output_path: 输出图片路径
    """
    
    # 设置图表样式
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
    plt.subplots_adjust(hspace=0.15)
    
    horizon_labels = {
        1: "H=1 (15分钟预测)", 
        4: "H=4 (1小时预测)", 
        16: "H=16 (4小时预测)"
    }
    colors = {
        1: "#2E86AB",   # 蓝色
        4: "#A23B72",   # 紫色
        16: "#F18F01"   # 橙色
    }
    
    results = {}
    
    # 加载所有 horizon 的数据
    for horizon in horizons:
        try:
            df = load_predictions(horizon, predictions_dir)
            results[horizon] = df
        except FileNotFoundError as e:
            print(f"警告: {e}")
            continue
    
    if not results:
        print("错误: 没有找到任何预测结果文件")
        return
    
    # 绘制每个 horizon 的图
    for idx, horizon in enumerate(sorted(results.keys())):
        ax = axes[idx]
        df = results[horizon]
        
        y_true = df["y_true"].values
        y_pred = df["y_pred"].values
        timestamps = df["timestamp"]
        
        # 计算指标
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))
        r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
        
        # 绘制真实值（黑色实线）和预测值（彩色虚线）
        ax.plot(timestamps, y_true, 
                label="真实值 (Actual)", 
                color="black", 
                linewidth=1.5, 
                linestyle='-',     # 实线
                alpha=0.95)
        ax.plot(timestamps, y_pred, 
                label="预测值 (Predicted)", 
                color=colors.get(horizon, "blue"), 
                linewidth=1.2, 
                linestyle='--',    # 虚线
                alpha=0.85)
        
        # 设置标签和标题
        ax.set_ylabel("功率 (kW)", fontsize=12, fontweight='bold')
        ax.set_title(f"亳州光伏预测 - {horizon_labels.get(horizon, f'H={horizon}')}  |  "
                    f"RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}",
                    fontsize=13, fontweight="bold", pad=10)
        
        # 图例
        ax.legend(loc="upper right", fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 美化x轴
        ax.tick_params(axis='x', rotation=30, labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        
        # 设置y轴范围稍微扩展
        y_min = min(y_true.min(), y_pred.min())
        y_max = max(y_true.max(), y_pred.max())
        y_range = y_max - y_min
        if y_range > 0:
            ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)
    
    # 最后一个图添加x轴标签
    axes[-1].set_xlabel("时间 (Time)", fontsize=12, fontweight='bold')
    
    # 整体标题
    fig.suptitle("亳州光伏电站功率预测 - 多预测步长对比\nBozhou PV Station Power Prediction - Multi-Horizon Comparison", 
                 fontsize=15, fontweight='bold', y=0.98)
    
    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # 确保输出目录存在
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 保存图片
    if output_path is None:
        output_path = OUTPUT_DIR / "bozhou_all_horizons_prediction_curve.png"
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"图片已保存: {output_path}")
    
    # 同时保存PDF版本
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"PDF已保存: {pdf_path}")
    
    plt.close()
    
    return output_path


def plot_4days_detail(horizons: list, predictions_dir: Path, output_path: Path = None):
    """绘制4天时间范围的详细预测曲线"""
    
    fig, axes = plt.subplots(3, 1, figsize=(18, 14), sharex=True)
    plt.subplots_adjust(hspace=0.12)
    
    horizon_labels = {
        1: "H=1 (15分钟预测)", 
        4: "H=4 (1小时预测)", 
        16: "H=16 (4小时预测)"
    }
    colors = {
        1: "#2E86AB",
        4: "#A23B72", 
        16: "#F18F01"
    }
    
    # 只绘制最后4天的数据
    n_days = 4
    hours_per_day = 24
    # 假设15分钟采样间隔
    samples_per_day = hours_per_day * 4
    n_samples = n_days * samples_per_day
    
    for idx, horizon in enumerate(sorted(horizons)):
        try:
            df = load_predictions(horizon, predictions_dir)
        except FileNotFoundError:
            continue
        
        ax = axes[idx]
        
        # 取最后4天数据
        y_true = df["y_true"].values[-n_samples:]
        y_pred = df["y_pred"].values[-n_samples:]
        timestamps = df["timestamp"].values[-n_samples:]
        
        # 计算指标
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))
        r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
        
        # 绘制
        ax.plot(timestamps, y_true, 
                label="真实值 (Actual)", 
                color="black", 
                linewidth=2, 
                linestyle='-',
                alpha=0.95)
        ax.plot(timestamps, y_pred, 
                label="预测值 (Predicted)", 
                color=colors.get(horizon, "blue"), 
                linewidth=1.8, 
                linestyle='--',
                alpha=0.85)
        
        # 填充预测误差区域
        ax.fill_between(timestamps, y_true, y_pred, alpha=0.2, color='gray')
        
        ax.set_ylabel("功率 (kW)", fontsize=12, fontweight='bold')
        ax.set_title(f"亳州光伏预测 - {horizon_labels.get(horizon, f'H={horizon}')}  |  "
                    f"RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}",
                    fontsize=13, fontweight="bold", pad=10)
        
        ax.legend(loc="upper right", fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(axis='x', rotation=30, labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
    
    axes[-1].set_xlabel("时间 (Time)", fontsize=12, fontweight='bold')
    
    fig.suptitle("亳州光伏电站功率预测 - 最近4天详细视图\nBozhou PV Station Power Prediction - 4-Day Detail View", 
                 fontsize=15, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if output_path is None:
        output_path = OUTPUT_DIR / "bozhou_all_horizons_prediction_curve_4days.png"
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"4天详细图已保存: {output_path}")
    
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"PDF已保存: {pdf_path}")
    
    plt.close()
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="亳州光伏预测结果绘图")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="预测结果目录路径")
    parser.add_argument("--output", type=str, default=None,
                        help="输出图片路径")
    parser.add_argument("--4days", action="store_true",
                        help="额外绘制4天详细视图")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 4, 16],
                        help="预测步长列表 (默认: 1 4 16)")
    
    args = parser.parse_args()
    
    # 设置路径
    if args.data_dir:
        predictions_dir = Path(args.data_dir)
    else:
        predictions_dir = DEFAULT_PREDICTIONS_DIR
    
    output_path = Path(args.output) if args.output else None
    
    print("=" * 60)
    print("亳州光伏预测结果绘图")
    print(f"数据目录: {predictions_dir}")
    print(f"预测步长: {args.horizons}")
    print("=" * 60)
    
    # 绘制主图
    plot_predictions(args.horizons, predictions_dir, output_path)
    
    # 绘制4天详细视图
    if args.4days:
        plot_4days_detail(args.horizons, predictions_dir)
    
    print("\n绘图完成!")


if __name__ == "__main__":
    main()
