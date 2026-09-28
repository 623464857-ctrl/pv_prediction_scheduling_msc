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


def load_predictions(horizon: int, predictions_dir: Path, deduplicate: bool = False) -> pd.DataFrame:
    """加载指定 horizon 的预测结果
    
    Args:
        horizon: 预测步长
        predictions_dir: 预测结果目录
        deduplicate: 是否去重（去重用于绘图，保留原始数据用于计算指标）
    """
    pred_file = predictions_dir / f"bozhou_h{horizon}" / "predictions.csv"
    
    if not pred_file.exists():
        raise FileNotFoundError(f"预测文件不存在: {pred_file}")
    
    df = pd.read_csv(pred_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # 对于 H>1 的数据，同一时间戳可能有多个预测（多起始点）
    if horizon > 1 and deduplicate:
        df = df.groupby("timestamp").first().reset_index()
    
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
    
    results_raw = {}  # 原始数据用于计算指标
    results_plot = {}  # 去重后数据用于绘图
    
    # 加载所有 horizon 的数据
    for horizon in horizons:
        try:
            # 加载原始数据用于计算指标
            df_raw = load_predictions(horizon, predictions_dir, deduplicate=False)
            results_raw[horizon] = df_raw
            # 加载去重数据用于绘图
            df_plot = load_predictions(horizon, predictions_dir, deduplicate=True)
            results_plot[horizon] = df_plot
        except FileNotFoundError as e:
            print(f"警告: {e}")
            continue
    
    if not results_raw:
        print("错误: 没有找到任何预测结果文件")
        return
    
    # 获取共同真实值（使用H=1的真实值作为基准）
    if 1 in results_plot:
        reference_df = results_plot[1][['timestamp', 'y_true']].copy()
        reference_df = reference_df.rename(columns={'y_true': 'y_true_ref'})
    else:
        first_key = list(results_plot.keys())[0]
        reference_df = results_plot[first_key][['timestamp', 'y_true']].copy()
        reference_df = reference_df.rename(columns={'y_true': 'y_true_ref'})
    
    # 绘制每个 horizon 的图
    for idx, horizon in enumerate(sorted(results_raw.keys())):
        ax = axes[idx]
        
        # 使用原始数据计算指标
        df_raw = results_raw[horizon]
        y_true_raw = df_raw["y_true"].values
        y_pred_raw = df_raw["y_pred"].values
        rmse = np.sqrt(np.mean((y_true_raw - y_pred_raw) ** 2))
        mae = np.mean(np.abs(y_true_raw - y_pred_raw))
        r2 = 1 - np.sum((y_true_raw - y_pred_raw) ** 2) / np.sum((y_true_raw - np.mean(y_true_raw)) ** 2)
        
        # 使用去重数据绘图
        df_plot = results_plot[horizon]
        df_plot_merge = reference_df.merge(df_plot[['timestamp', 'y_pred']], on='timestamp', how='inner')
        
        y_true = df_plot_merge["y_true_ref"].values
        y_pred = df_plot_merge["y_pred"].values
        timestamps = df_plot_merge["timestamp"]
        
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
    
    plt.close()
    
    return output_path


def plot_4days_detail(horizons: list, predictions_dir: Path, output_path: Path = None, start_date: str = None):
    """绘制4天时间范围的详细预测曲线
    
    Args:
        horizons: 预测步长列表
        predictions_dir: 预测结果目录
        output_path: 输出图片路径
        start_date: 起始日期字符串，格式为 'YYYY-MM-DD'，默认为数据末尾4天
    """
    
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
    
    results_raw = {}  # 原始数据用于计算指标
    results_plot = {}  # 去重后数据用于绘图
    
    # 加载所有 horizon 的数据
    for horizon in horizons:
        try:
            # 加载原始数据用于计算指标
            df_raw = load_predictions(horizon, predictions_dir, deduplicate=False)
            results_raw[horizon] = df_raw
            # 加载去重数据用于绘图
            df_plot = load_predictions(horizon, predictions_dir, deduplicate=True)
            results_plot[horizon] = df_plot
        except FileNotFoundError:
            continue
    
    if not results_raw:
        return
    
    # 获取共同真实值（使用H=1的真实值作为基准）
    if 1 in results_plot:
        reference_df = results_plot[1][['timestamp', 'y_true']].copy()
        reference_df = reference_df.rename(columns={'y_true': 'y_true_ref'})
    else:
        first_key = list(results_plot.keys())[0]
        reference_df = results_plot[first_key][['timestamp', 'y_true']].copy()
        reference_df = reference_df.rename(columns={'y_true': 'y_true_ref'})
    
    # 计算时间范围
    n_days = 4
    hours_per_day = 24
    samples_per_day = hours_per_day * 4
    n_samples = n_days * samples_per_day
    
    # 获取数据的时间范围
    all_timestamps = reference_df['timestamp'].values
    if start_date:
        start_ts = pd.to_datetime(start_date)
        # 找到最接近起始日期的时间戳
        start_idx = np.argmin(np.abs(pd.to_datetime(all_timestamps) - start_ts))
    else:
        # 默认为末尾4天
        start_idx = len(all_timestamps) - n_samples
    
    # 确保索引不超出范围
    start_idx = max(0, min(start_idx, len(all_timestamps) - n_samples))
    
    # 确定时间范围标题
    start_datetime = pd.to_datetime(all_timestamps[start_idx])
    end_datetime = pd.to_datetime(all_timestamps[start_idx + n_samples - 1])
    date_range_title = f"{start_datetime.strftime('%Y-%m-%d %H:%M')} 至 {end_datetime.strftime('%Y-%m-%d %H:%M')}"
    
    for idx, horizon in enumerate(sorted(results_raw.keys())):
        ax = axes[idx]
        
        # 使用原始数据计算指标
        df_raw = results_raw[horizon]
        y_true_raw = df_raw["y_true"].values
        y_pred_raw = df_raw["y_pred"].values
        rmse = np.sqrt(np.mean((y_true_raw - y_pred_raw) ** 2))
        mae = np.mean(np.abs(y_true_raw - y_pred_raw))
        r2 = 1 - np.sum((y_true_raw - y_pred_raw) ** 2) / np.sum((y_true_raw - np.mean(y_true_raw)) ** 2)
        
        # 使用去重数据绘图
        df_plot = results_plot[horizon]
        df_plot_merge = reference_df.merge(df_plot[['timestamp', 'y_pred']], on='timestamp', how='inner')
        
        # 根据起始位置截取数据
        y_true = df_plot_merge["y_true_ref"].values[start_idx:start_idx + n_samples]
        y_pred = df_plot_merge["y_pred"].values[start_idx:start_idx + n_samples]
        timestamps = df_plot_merge["timestamp"].values[start_idx:start_idx + n_samples]
        
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
    
    fig.suptitle(f"亳州光伏电站功率预测 - 4天详细视图 ({date_range_title})\nBozhou PV Station Power Prediction - 4-Day Detail View", 
                 fontsize=15, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if output_path is None:
        # 根据起始日期生成文件名
        filename = f"bozhou_all_horizons_prediction_curve_4days_{start_datetime.strftime('%m%d')}.png"
        output_path = OUTPUT_DIR / filename
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"4天详细图已保存: {output_path}")
    
    plt.close()
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="亳州光伏预测结果绘图")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="预测结果目录路径")
    parser.add_argument("--output", type=str, default=None,
                        help="输出图片路径")
    parser.add_argument("--detail-4days", action="store_true",
                        help="绘制4天详细视图")
    parser.add_argument("--detail-start", type=str, default=None,
                        help="4天视图的起始日期，格式: YYYY-MM-DD")
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
    if args.detail_4days:
        plot_4days_detail(args.horizons, predictions_dir, start_date=args.detail_start)
    
    print("\n绘图完成!")


if __name__ == "__main__":
    main()
