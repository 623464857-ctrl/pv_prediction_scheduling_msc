"""
明月湖预测曲线可视化脚本
生成三个H各自的预测曲线长图和合并图
使用原始保存的预测数据进行指标展示
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from pathlib import Path
import json

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 路径配置
PRED_DIR = Path(__file__).parent.parent.parent.parent / "data" / "prediction" / "step2_hyperparameter_search" / "predictions"
RAW_DIR = Path(__file__).parent.parent.parent.parent / "data" / "raw"
METRICS_DIR = Path(__file__).parent.parent.parent.parent / "data" / "prediction" / "step2_hyperparameter_search" / "metrics"
OUTPUT_DIR = Path(__file__).parent.parent.parent.parent / "data" / "prediction" / "step2_hyperparameter_search" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 明月湖装机容量 (kW)
CAPACITY_KW = 281.6


def is_night_hour(ts: pd.Timestamp) -> bool:
    """判断是否为夜间时刻（功率应为0）"""
    hour = ts.hour
    return hour < 6 or hour >= 20


def load_original_data() -> pd.DataFrame:
    """加载原始完整数据（含夜间0功率），归一化到p.u."""
    file_path = RAW_DIR / "明月湖光伏发电.csv"
    df = pd.read_csv(file_path)
    df.columns = ['timestamp', 'power_kw']  # 重命名中文列
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 归一化到p.u.
    df['power_pu'] = df['power_kw'] / CAPACITY_KW
    
    return df[['timestamp', 'power_pu']]


def load_prediction_data(horizon: int) -> pd.DataFrame:
    """加载指定horizon的预测数据（实际功率 p.u.）"""
    file_path = PRED_DIR / f"mingyuehu_h{horizon}" / "cnn_bilstm_seed42_test.csv"
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 对同一时间戳的多个样本取平均
    agg_df = df.groupby('timestamp').agg({
        'y_true': 'mean',
        'y_pred': 'mean'
    }).reset_index()
    
    return agg_df[['timestamp', 'y_true', 'y_pred']]


def load_metrics(horizon: int) -> dict:
    """加载指定horizon的原始保存指标"""
    metrics_file = METRICS_DIR / f"mingyuehu_h{horizon}" / "mingyuehu_cnn_bilstm_reproduce.json"
    if metrics_file.exists():
        with open(metrics_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 使用seed=42的指标
        for seed_data in data['per_seed']:
            if seed_data['seed'] == 42:
                return seed_data
    return {}


def merge_with_original(pred_df: pd.DataFrame, orig_df: pd.DataFrame) -> pd.DataFrame:
    """合并原始数据和预测数据，保持完整时间序列（含夜间）"""
    min_ts = pred_df['timestamp'].min().floor('D')
    max_ts = pred_df['timestamp'].max().ceil('D')
    
    # 筛选原始数据中对应的时间范围
    orig_filtered = orig_df[(orig_df['timestamp'] >= min_ts) & (orig_df['timestamp'] <= max_ts)].copy()
    orig_filtered = orig_filtered.sort_values('timestamp').reset_index(drop=True)
    
    # 合并预测数据
    merged = orig_filtered.merge(
        pred_df[['timestamp', 'y_pred']], 
        on='timestamp', 
        how='left'
    )
    merged = merged.rename(columns={'power_pu': 'y_true'})
    
    # 强制夜间功率为0
    night_mask = merged['timestamp'].apply(is_night_hour)
    merged.loc[night_mask, 'y_pred'] = 0.0
    merged['y_pred'] = merged['y_pred'].fillna(0.0)
    
    return merged[['timestamp', 'y_true', 'y_pred']]


def plot_prediction_curves(horizon: int, merged_df: pd.DataFrame, metrics: dict, out_path: Path) -> None:
    """绘制单个horizon的预测曲线"""
    df = merged_df.copy()
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    time_range = df['timestamp'].max() - df['timestamp'].min()
    total_hours = time_range.total_seconds() / 3600
    
    if total_hours <= 24:
        figsize = (16, 6)
    elif total_hours <= 48:
        figsize = (20, 6)
    else:
        figsize = (24, 6)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制真实曲线
    ax.plot(df['timestamp'], df['y_true'], 
            color='black', linewidth=1.5, 
            label='真实值 (Actual)', 
            linestyle='-',
            zorder=3)
    
    # 绘制预测曲线
    df_with_pred = df.dropna(subset=['y_pred'])
    if len(df_with_pred) > 0:
        ax.plot(df_with_pred['timestamp'], df_with_pred['y_pred'], 
                color='#1f77b4', linewidth=1.2, 
                label='预测值 (Predicted)', 
                linestyle='--',
                alpha=0.8,
                zorder=2)
    
    # 设置x轴格式
    if total_hours <= 24:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, int(total_hours/12))))
    elif total_hours <= 48:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator())
    
    plt.xticks(rotation=45, ha='right')
    ax.set_xlabel('时间 (Time)', fontsize=12)
    ax.set_ylabel('功率 (p.u.)', fontsize=12)
    
    horizon_desc = {1: "15min", 4: "1小时", 16: "4小时"}
    ax.set_title(f'明月湖光伏功率预测曲线 (H={horizon}, {horizon_desc.get(horizon, str(horizon))}预测)\n'
                 f'CNN-BiLSTM模型 | 测试集 | 单位: p.u. | 时间范围: {df["timestamp"].min().strftime("%Y-%m-%d %H:%M")} ~ {df["timestamp"].max().strftime("%Y-%m-%d %H:%M")}',
                 fontsize=14, fontweight='bold')
    
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_ylim(-0.02, max(1.1, df['y_true'].max() * 1.1))
    
    # 标注夜间区域
    for i in range(len(df) - 1):
        if is_night_hour(df['timestamp'].iloc[i]):
            ax.axvspan(df['timestamp'].iloc[i], df['timestamp'].iloc[i + 1],
                       alpha=0.08, color='gray', zorder=1)
    
    # 使用原始保存的指标
    if metrics:
        mae = metrics.get('MAE', 0)
        rmse = metrics.get('RMSE', 0)
        r2 = metrics.get('R2', 0)
        stats_text = f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR2: {r2:.4f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"已保存: {out_path}")


def plot_combined_curves(all_merged_dfs: dict, all_metrics: dict, out_path: Path) -> None:
    """绘制三个H的预测曲线在同一张图中（上下排列，x轴对齐）"""
    horizons = [1, 4, 16]
    horizon_desc = {1: "H=1 (15min)", 4: "H=4 (1小时)", 16: "H=16 (4小时)"}
    
    # 统一时间范围：使用所有H中最宽的范围
    min_ts = min(df['timestamp'].min() for df in all_merged_dfs.values())
    max_ts = max(df['timestamp'].max() for df in all_merged_dfs.values())
    
    fig, axes = plt.subplots(3, 1, figsize=(20, 12), sharex=True)
    fig.suptitle(f'明月湖光伏功率预测曲线 (CNN-BiLSTM模型 | 测试集)\n'
                 f'时间范围: {min_ts.strftime("%Y-%m-%d %H:%M")} ~ {max_ts.strftime("%Y-%m-%d %H:%M")}',
                 fontsize=14, fontweight='bold', y=0.98)
    
    colors = {1: '#1f77b4', 4: '#ff7f0e', 16: '#2ca02c'}
    
    for idx, h in enumerate(horizons):
        ax = axes[idx]
        df = all_merged_dfs[h].copy()
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # 筛选到统一时间范围
        df = df[(df['timestamp'] >= min_ts) & (df['timestamp'] <= max_ts)]
        
        # 绘制真实曲线
        ax.plot(df['timestamp'], df['y_true'], 
                color='black', linewidth=1.5, 
                label='真实值 (Actual)', 
                linestyle='-',
                zorder=3)
        
        # 绘制预测曲线
        df_with_pred = df.dropna(subset=['y_pred'])
        if len(df_with_pred) > 0:
            ax.plot(df_with_pred['timestamp'], df_with_pred['y_pred'], 
                    color=colors[h], linewidth=1.2, 
                    label=f'预测值 (H={h})', 
                    linestyle='--',
                    alpha=0.8,
                    zorder=2)
        
        ax.set_ylabel('功率 (p.u.)', fontsize=11)
        ax.set_title(horizon_desc[h], fontsize=12, fontweight='bold', loc='left')
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_ylim(-0.02, 1.1)
        
        # 标注夜间区域
        for i in range(len(df) - 1):
            if is_night_hour(df['timestamp'].iloc[i]):
                ax.axvspan(df['timestamp'].iloc[i], df['timestamp'].iloc[i + 1],
                           alpha=0.08, color='gray', zorder=1)
        
        # 使用原始保存的指标
        metrics = all_metrics.get(h, {})
        if metrics:
            mae = metrics.get('MAE', 0)
            rmse = metrics.get('RMSE', 0)
            r2 = metrics.get('R2', 0)
            stats_text = f'MAE: {mae:.4f}  RMSE: {rmse:.4f}  R2: {r2:.4f}'
            ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 设置x轴格式
    axes[-1].set_xlabel('时间 (Time)', fontsize=12)
    
    time_range = max_ts - min_ts
    total_hours = time_range.total_seconds() / 3600
    
    if total_hours <= 24:
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=max(1, int(total_hours/12))))
    elif total_hours <= 48:
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=4))
    else:
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        axes[-1].xaxis.set_major_locator(mdates.DayLocator())
    
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, hspace=0.15)
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"已保存: {out_path}")


def main():
    # 加载原始数据
    orig_df = load_original_data()
    print(f"原始数据范围: {orig_df['timestamp'].min()} ~ {orig_df['timestamp'].max()}")
    
    horizons = [1, 4, 16]
    all_merged_dfs = {}
    all_metrics = {}
    
    for h in horizons:
        print(f"\n处理 horizon={h}...")
        
        # 加载预测数据
        pred_df = load_prediction_data(h)
        print(f"  预测数据: {len(pred_df)} 条")
        print(f"  预测时间范围: {pred_df['timestamp'].min()} ~ {pred_df['timestamp'].max()}")
        
        # 合并数据
        merged_df = merge_with_original(pred_df, orig_df)
        all_merged_dfs[h] = merged_df
        print(f"  合并后数据: {len(merged_df)} 条")
        
        # 加载原始保存的指标
        metrics = load_metrics(h)
        all_metrics[h] = metrics
        if metrics:
            print(f"  原始指标: MAE={metrics.get('MAE', 0):.4f}, RMSE={metrics.get('RMSE', 0):.4f}, R2={metrics.get('R2', 0):.4f}")
        
        # 生成单独图片
        out_path = OUTPUT_DIR / f"mingyuehu_h{h}_prediction_curve.png"
        plot_prediction_curves(h, merged_df, metrics, out_path)
    
    # 生成合并图片
    combined_out_path = OUTPUT_DIR / "mingyuehu_all_horizons_prediction_curve.png"
    plot_combined_curves(all_merged_dfs, all_metrics, combined_out_path)
    
    print(f"\n所有图片已保存到: {OUTPUT_DIR}")
    
    # 打印对比摘要
    print("\n" + "=" * 50)
    print("Metric Summary (Original saved data, seed=42):")
    print("=" * 50)
    print(f"{'Horizon':<12} {'MAE':<10} {'RMSE':<10} {'R2':<10}")
    print("-" * 50)
    for h in horizons:
        m = all_metrics.get(h, {})
        print(f"H={h:<10} {m.get('MAE', 0):.4f}     {m.get('RMSE', 0):.4f}     {m.get('R2', 0):.4f}")


if __name__ == "__main__":
    main()
