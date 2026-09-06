"""
生成数据特征箱线图
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
data_path = r"c:\Users\MoYu\Desktop\pv_prediction_scheduling_msc_new\data\data-analysis\data\明月湖_cleaned.csv"
output_path = r"c:\Users\MoYu\Desktop\pv_prediction_scheduling_msc_new\data\data-analysis\charts\feature_boxplots.png"

df = pd.read_csv(data_path)

# 选择数值特征（排除时间戳和分类特征）
feature_cols = [
    'power_kw',
    'ghi', 
    'dhi',
    'temperature_c',
    'apparent_temperature_c',
    'relative_humidity_pct',
    'wind_gust_ms',
    'solar_altitude_deg',
    'uv'
]

feature_labels = [
    'Power (kW)',
    'GHI (W/m2)',
    'DHI (W/m2)', 
    'Temperature (C)',
    'Apparent Temp (C)',
    'Humidity (%)',
    'Wind Gust (m/s)',
    'Solar Altitude (deg)',
    'UV Index'
]

# 提取特征数据
features_data = [df[col].dropna().values for col in feature_cols]

# 创建图表
fig, ax = plt.subplots(figsize=(14, 8))

# 绘制箱线图
bp = ax.boxplot(features_data, patch_artist=True, labels=feature_labels)

# 设置颜色
colors = ['#E53935', '#FFA726', '#FFB74D', '#26A69A', '#4DB6AC', 
          '#42A5F5', '#5C6BC0', '#AB47BC', '#7E57C2']

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# 设置样式
for whisker in bp['whiskers']:
    whisker.set(color='#7570B3', linewidth=1.5, linestyle='--')

for cap in bp['caps']:
    cap.set(color='#7570B3', linewidth=1.5)

for median in bp['medians']:
    median.set(color='white', linewidth=2)

for flier in bp['fliers']:
    flier.set(marker='o', markerfacecolor='#D32F2F', alpha=0.5, markersize=3)

ax.set_ylabel('Value', fontsize=12)
ax.set_title('Feature Boxplot Analysis\n明月湖光伏电站数据', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print(f"箱线图已保存: {output_path}")

# 输出统计信息
print("\n特征统计摘要:")
print("-" * 80)
stats_df = df[feature_cols].describe()
print(stats_df.round(2).to_string())
