"""
生成选定特征与 power_kw 的直方图和散点图
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
DATA_PATH = Path("data/prediction/step1_preprocessing/processed/stations/Mingyuehu_1_preprocessed.csv")
df = pd.read_csv(DATA_PATH)

print(f"数据形状: {df.shape}")
print(f"power_kw 范围: {df['power_kw'].min():.2f} ~ {df['power_kw'].max():.2f} kW")

# 选定的特征及名称
features = {
    'temperature_c': 'Temperature (C)',
    'relative_humidity_pct': 'Relative Humidity (%)',
    'wind_gust_ms': 'Wind Gust (m/s)',
    'solar_altitude_deg': 'Solar Altitude (deg)',
    'ghi_wm2': 'GHI (W/m2)',
}

feature_keys = list(features.keys())
feature_labels = list(features.values())

# 创建 5x2 的图表布局 (5个特征 x 2种图: 直方图 + 散点图)
fig, axes = plt.subplots(len(feature_keys), 2, figsize=(14, 20))

for idx, (feat, label) in enumerate(zip(feature_keys, feature_labels)):
    # 左列: 直方图
    ax_hist = axes[idx, 0]
    ax_scatter = axes[idx, 1]
    
    # 过滤有效数据
    valid_mask = df[feat].notna() & df['power_kw'].notna()
    x_data = df.loc[valid_mask, feat]
    y_data = df.loc[valid_mask, 'power_kw']
    
    # 计算相关系数
    corr = x_data.corr(y_data)
    
    # 直方图
    ax_hist.hist(x_data, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
    ax_hist.set_xlabel(label, fontsize=11)
    ax_hist.set_ylabel('Frequency', fontsize=11)
    ax_hist.set_title(f'{label} Distribution', fontsize=12)
    ax_hist.grid(axis='y', alpha=0.3)
    
    # 添加统计信息
    stats_text = f'μ={x_data.mean():.2f}\nσ={x_data.std():.2f}\nmin={x_data.min():.2f}\nmax={x_data.max():.2f}'
    ax_hist.text(0.95, 0.95, stats_text, transform=ax_hist.transAxes, 
                 fontsize=9, verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 散点图
    # 随机采样一部分点以避免过密
    sample_size = min(2000, len(x_data))
    if len(x_data) > sample_size:
        sample_idx = np.random.choice(len(x_data), sample_size, replace=False)
        x_sample = x_data.iloc[sample_idx]
        y_sample = y_data.iloc[sample_idx]
    else:
        x_sample = x_data
        y_sample = y_data
    
    ax_scatter.scatter(x_sample, y_sample, alpha=0.3, s=10, color='steelblue')
    ax_scatter.set_xlabel(label, fontsize=11)
    ax_scatter.set_ylabel('power_kw (kW)', fontsize=11)
    ax_scatter.set_title(f'{label} vs power_kw (r={corr:.3f})', fontsize=12)
    ax_scatter.grid(alpha=0.3)
    
    # 添加回归线
    if len(x_sample) > 2:
        z = np.polyfit(x_sample, y_sample, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x_sample.min(), x_sample.max(), 100)
        ax_scatter.plot(x_line, p(x_line), 'r-', linewidth=2, label='线性回归')
        ax_scatter.legend()

plt.tight_layout()

# 保存图片
output_path = Path("data/prediction/step1_preprocessing/charts/03_feature_histogram_scatter.png")
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print(f"\n图表已保存: {output_path}")

# 输出统计摘要
print("\n" + "=" * 60)
print("Feature Statistics Summary")
print("=" * 60)

for feat, label in zip(feature_keys, feature_labels):
    valid_data = df[feat].dropna()
    corr = df[feat].corr(df['power_kw'])
    print(f"\n{label}:")
    print(f"  Count: {len(valid_data)}")
    print(f"  Mean: {valid_data.mean():.2f}")
    print(f"  Std: {valid_data.std():.2f}")
    print(f"  Range: [{valid_data.min():.2f}, {valid_data.max():.2f}]")
    print(f"  Corr with power_kw: {corr:.4f}")
