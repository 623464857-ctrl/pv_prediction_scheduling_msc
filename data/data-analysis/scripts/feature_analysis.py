"""
明月湖数据集 - 特征分析与可视化
=================================
生成：
  1. 特征-功率相关性热力图
  2. 重要特征的数据分布图
  3. 特征-功率散点图
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import warnings
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

warnings.filterwarnings('ignore')

# ===================== 中文字体设置 =====================
import os

# 优先使用的 Windows 中文字体（按优先级）
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",   # Microsoft YaHei 微软雅黑
    r"C:\Windows\Fonts\simhei.ttf",  # SimHei 黑体
    r"C:\Windows\Fonts\STXIHEI.TTF", # 华文细黑
    r"C:\Windows\Fonts\NotoSansSC-VF.ttf",  # Noto Sans SC
    r"C:\Windows\Fonts\STKAITI.TTF",  # 华文楷体
]

CHINESE_FONT_PATH = None
for fp in FONT_CANDIDATES:
    if os.path.exists(fp):
        CHINESE_FONT_PATH = fp
        break

if CHINESE_FONT_PATH:
    # 全局注册字体
    fm.fontManager.addfont(CHINESE_FONT_PATH)
    font_prop = fm.FontProperties(fname=CHINESE_FONT_PATH)
    font_name = font_prop.get_name()
    plt.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans']
    print(f"[字体] 使用中文字体: {CHINESE_FONT_PATH}  (name: {font_name})")
else:
    font_prop = None
    print("[字体] 未找到中文字体，使用默认字体")

# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False
# 使用 Agg 后端

def make_font(size=10):
    """返回中文字体 FontProperties，供单个文字元素使用"""
    if CHINESE_FONT_PATH:
        return fm.FontProperties(fname=CHINESE_FONT_PATH, size=size)
    return fm.FontProperties(size=size)

# ===================== 路径配置 =====================
DATA_DIR = r"c:\Users\MoYu\Desktop\pv_prediction_scheduling_msc_new\data\data-analysis"
SCRIPT_DIR = DATA_DIR + r"\scripts"
REPORT_DIR = DATA_DIR + r"\reports"
CHART_DIR = DATA_DIR + r"\charts"
DATA_FILE = DATA_DIR + r"\data\明月湖_merged_raw.csv"

# ===================== 加载数据 =====================
print("加载数据...")
df = pd.read_csv(DATA_FILE, encoding='utf-8-sig')
df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f"数据行数: {len(df)}, 列数: {len(df.columns)}")

# ===================== 选择数值特征列 =====================
exclude_cols = ['timestamp', 'part_of_day', 'weather_code', 'weather_description',
                'weather_icon', 'hour_of_day', 'wind_dir_deg']
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
analysis_cols = [c for c in numeric_cols if c not in exclude_cols]

# 确保 power_kw 在首位
if 'power_kw' in analysis_cols:
    analysis_cols.remove('power_kw')
    analysis_cols.insert(0, 'power_kw')

print(f"分析特征数: {len(analysis_cols)}")
print(f"分析特征: {analysis_cols}")

# ===================== 图1: 相关性热力图 =====================
print("\n生成图1: 相关性热力图...")

corr_matrix = df[analysis_cols].corr()

fig, ax = plt.subplots(figsize=(16, 14))
# 使用蓝红色系，power_kw 单独高亮
mask = np.zeros_like(corr_matrix, dtype=bool)
# 不遮罩，显示完整矩阵

# 绘制热力图
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt='.2f',
    cmap='RdBu_r',
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.3,
    cbar_kws={'shrink': 0.75, 'label': 'Pearson Correlation'},
    annot_kws={'size': 7},
    ax=ax
)

# 标题
ax.set_title(
    'Mingyue Lake PV Dataset — Feature Correlation Heatmap\n'
    '(明月湖光伏数据集 — 特征相关性热力图)',
    fontproperties=make_font(13),
    fontweight='bold',
    pad=15
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

plt.tight_layout()
heatmap_path = CHART_DIR + r"\01_correlation_heatmap.png"
plt.savefig(heatmap_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  已保存: {heatmap_path}")

# ===================== 图2: power_kw 与关键特征散点图 =====================
print("\n生成图2: 功率-关键特征散点图...")

key_features = ['ghi', 'dhi', 'temperature_c', 'relative_humidity_pct',
                'cloud_cover_pct', 'solar_altitude_deg', 'solar_radiation_wm2',
                'wind_speed_ms', 'pressure_hpa', 'precip_rate_mmhr']

# 过滤存在的特征
key_features = [f for f in key_features if f in df.columns]
n_features = len(key_features)
n_cols = 3
n_rows = int(np.ceil(n_features / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
axes = axes.flatten()

feature_labels = {
    'ghi': 'GHI (W/m²)',
    'dhi': 'DHI (W/m²)',
    'temperature_c': 'Temperature (°C)',
    'relative_humidity_pct': 'Relative Humidity (%)',
    'cloud_cover_pct': 'Cloud Cover (%)',
    'solar_altitude_deg': 'Solar Altitude (°)',
    'solar_radiation_wm2': 'Solar Radiation (W/m²)',
    'wind_speed_ms': 'Wind Speed (m/s)',
    'pressure_hpa': 'Pressure (hPa)',
    'precip_rate_mmhr': 'Precipitation (mm/h)',
}

feature_labels_cn = {
    'ghi': '水平面总辐射 (W/m²)',
    'dhi': '水平面散射辐射 (W/m²)',
    'temperature_c': '温度 (°C)',
    'relative_humidity_pct': '相对湿度 (%)',
    'cloud_cover_pct': '云量 (%)',
    'solar_altitude_deg': '太阳高度角 (°)',
    'solar_radiation_wm2': '太阳辐射强度 (W/m²)',
    'wind_speed_ms': '风速 (m/s)',
    'pressure_hpa': '气压 (hPa)',
    'precip_rate_mmhr': '降水率 (mm/h)',
}

for i, feat in enumerate(key_features):
    ax = axes[i]
    x = df[feat].values
    y = df['power_kw'].values

    # 采样绘制（数据量大，scatter 太多点太慢）
    sample_size = min(3000, len(df))
    idx = np.random.choice(len(df), sample_size, replace=False)
    x_s, y_s = x[idx], y[idx]

    ax.scatter(x_s, y_s, alpha=0.3, s=8, c='steelblue', rasterized=True)
    ax.set_xlabel(feature_labels.get(feat, feat), fontsize=9)
    ax.set_ylabel('Power (kW)', fontsize=9)
    ax.set_title(f'Power vs {feature_labels_cn.get(feat, feat)}', fontsize=9,
                 fontproperties=make_font(9))

    # 相关系数
    corr = df[feat].corr(df['power_kw'])
    ax.annotate(f'r = {corr:.3f}', xy=(0.05, 0.92), xycoords='axes fraction',
                fontsize=9, color='red',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.grid(True, alpha=0.3)

# 隐藏多余的子图
for j in range(n_features, len(axes)):
    axes[j].set_visible(False)

fig.suptitle(
    'Mingyue Lake PV — Power vs Key Features Scatter\n'
    '(功率与关键特征散点图)',
    fontproperties=make_font(13), fontweight='bold', y=1.01
)
plt.tight_layout()
scatter_path = CHART_DIR + r"\02_power_vs_features_scatter.png"
plt.savefig(scatter_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  已保存: {scatter_path}")

# ===================== 图3: 重要特征的分布直方图 =====================
print("\n生成图3: 特征分布直方图...")

dist_features = ['power_kw', 'ghi', 'dhi', 'temperature_c', 'relative_humidity_pct',
                 'cloud_cover_pct', 'solar_altitude_deg']

dist_labels_cn = {
    'power_kw': '光伏发电功率 (kW)',
    'ghi': '水平面总辐射 GHI (W/m²)',
    'dhi': '水平面散射辐射 DHI (W/m²)',
    'temperature_c': '温度 (°C)',
    'relative_humidity_pct': '相对湿度 (%)',
    'cloud_cover_pct': '云量 (%)',
    'solar_altitude_deg': '太阳高度角 (°)',
}

n_cols = 3
n_rows = int(np.ceil(len(dist_features) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4.5 * n_rows))
axes = axes.flatten()

for i, feat in enumerate(dist_features):
    ax = axes[i]
    data = df[feat].dropna()

    ax.hist(data, bins=50, color='steelblue', alpha=0.7, edgecolor='white', density=True)
    ax.axvline(data.mean(), color='red', linestyle='--', linewidth=1.5, label=f'Mean: {data.mean():.2f}')
    ax.axvline(data.median(), color='orange', linestyle='--', linewidth=1.5, label=f'Median: {data.median():.2f}')
    ax.set_xlabel(dist_labels_cn.get(feat, feat), fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(f'{dist_labels_cn.get(feat, feat)} 分布',
                  fontsize=10, fontweight='bold', fontproperties=make_font(10))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 添加统计信息文本框
    stats_text = (f'Skewness: {data.skew():.2f}\n'
                  f'Kurtosis: {data.kurtosis():.2f}\n'
                  f'N: {len(data):,}')
    ax.text(0.97, 0.72, stats_text, transform=ax.transAxes,
            fontsize=8, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

for j in range(len(dist_features), len(axes)):
    axes[j].set_visible(False)

fig.suptitle(
    'Mingyue Lake PV — Feature Distribution Histograms\n'
    '(特征数据分布直方图)',
    fontproperties=make_font(13), fontweight='bold', y=1.02
)
plt.tight_layout()
hist_path = CHART_DIR + r"\03_feature_distributions.png"
plt.savefig(hist_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  已保存: {hist_path}")

# ===================== 图4: 箱线图（按天气类型） =====================
print("\n生成图4: 按天气类型的功率箱线图...")

# 按天气类型统计
weather_power = df.groupby('weather_description')['power_kw'].agg(['mean', 'median', 'std', 'count']).reset_index()
weather_power = weather_power.sort_values('mean', ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 左：箱线图（Top 10）
top_weather = weather_power.head(10)['weather_description'].tolist()
df_top = df[df['weather_description'].isin(top_weather)]

order = df_top.groupby('weather_description')['power_kw'].median().sort_values(ascending=False).index

ax1 = axes[0]
sns.boxplot(
    data=df_top,
    x='weather_description',
    y='power_kw',
    order=order,
    palette='viridis',
    ax=ax1,
    showfliers=True,
    flierprops={'markersize': 2, 'alpha': 0.3}
)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=35, ha='right', fontsize=8)
ax1.set_xlabel('Weather Type', fontsize=10)
ax1.set_ylabel('Power (kW)', fontsize=10)
ax1.set_title('Power Distribution by Weather Type (Top 10)\n(按天气类型的功率分布)',
                  fontsize=10, fontweight='bold', fontproperties=make_font(10))
ax1.grid(True, alpha=0.3, axis='y')

# 右：均值条形图（所有天气类型）
ax2 = axes[1]
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(weather_power)))
bars = ax2.barh(
    weather_power['weather_description'],
    weather_power['mean'],
    color=colors,
    alpha=0.85
)
ax2.set_xlabel('Mean Power (kW)', fontsize=10)
ax2.set_title('Mean Power by Weather Type\n(各天气类型的平均功率)',
                  fontsize=10, fontweight='bold', fontproperties=make_font(10))
ax2.grid(True, alpha=0.3, axis='x')

# 在条上标注数值
for bar, (_, row) in zip(bars, weather_power.iterrows()):
    width = bar.get_width()
    ax2.text(width + 0.5, bar.get_y() + bar.get_height()/2,
              f'{width:.1f} kW', va='center', fontsize=7)

ax2.invert_yaxis()

plt.tight_layout()
weather_path = CHART_DIR + r"\04_power_by_weather.png"
plt.savefig(weather_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  已保存: {weather_path}")

# ===================== 图5: 功率时序图（每天典型） =====================
print("\n生成图5: 功率时序图...")

df['date'] = df['timestamp'].dt.date

# 选取晴、多云、阴典型各一天
def get_typical_day(weather_type_substr):
    for wtype in weather_power['weather_description']:
        if weather_type_substr.lower() in wtype.lower():
            sub = df[df['weather_description'] == wtype]
            if len(sub) > 0:
                counts = sub.groupby('date').size()
                typical = counts.idxmax()
                return typical, wtype
    return None, None

typical_days = {}
for kw, wt in [('clear', 'Clear Sky'), ('overcast', 'Overcast'), ('cloud', 'Broken')]:
    day, wtype = get_typical_day(kw)
    if day:
        typical_days[day] = wtype

fig, axes = plt.subplots(len(typical_days), 1, figsize=(14, 4 * len(typical_days)), sharex=True)
if len(typical_days) == 1:
    axes = [axes]
axes_iter = iter(axes)

colors_line = {'ghi': 'orange', 'power_kw': 'steelblue'}

for day, wtype in typical_days.items():
    ax = next(axes_iter)
    day_df = df[df['date'] == day].sort_values('timestamp')

    ax2 = ax.twinx()
    l1, = ax.plot(day_df['timestamp'], day_df['power_kw'], color='steelblue',
                   linewidth=1.5, label='Power (kW)', alpha=0.9)
    l2, = ax2.plot(day_df['timestamp'], day_df['ghi'], color='orange',
                    linewidth=1.2, label='GHI (W/m²)', alpha=0.7)

    ax.set_ylabel('Power (kW)', color='steelblue', fontsize=9)
    ax2.set_ylabel('GHI (W/m²)', color='orange', fontsize=9)
    ax.tick_params(axis='y', labelcolor='steelblue')
    ax2.tick_params(axis='y', labelcolor='orange')
    ax.set_title(f'{day} — {wtype}', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(handles=[l1, l2], loc='upper right', fontsize=8)

    for spine in ax.spines.values():
        spine.set_visible(False)
    for spine in ax2.spines.values():
        spine.set_visible(False)

    # 格式化x轴时间
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))

fig.suptitle(
    'Mingyue Lake PV — Daily Power & GHI Time Series (Typical Days)\n'
    '(典型天气日的功率与辐照度时序)',
    fontproperties=make_font(13), fontweight='bold'
)
plt.tight_layout()
timeseries_path = CHART_DIR + r"\05_daily_timeseries.png"
plt.savefig(timeseries_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  已保存: {timeseries_path}")

# ===================== 图6: 云量-GHI-功率 3D关系（散点密度） =====================
print("\n生成图6: 云量-辐照-功率关系图...")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 左：GHI vs 功率，按云量着色
ax1 = axes[0]
sample = df[(df['ghi'] > 0) & (df['power_kw'] > 0)].sample(min(3000, len(df)), random_state=42)
scatter = ax1.scatter(
    sample['ghi'], sample['power_kw'],
    c=sample['cloud_cover_pct'], cmap='coolwarm_r',
    alpha=0.4, s=10, rasterized=True
)
cbar = plt.colorbar(scatter, ax=ax1, shrink=0.8)
cbar.set_label('Cloud Cover (%)', fontsize=9)
ax1.set_xlabel('GHI (W/m²)', fontsize=10)
ax1.set_ylabel('Power (kW)', fontsize=10)
ax1.set_title('GHI vs Power (colored by Cloud Cover)\n(GHI-功率关系，按云量着色)',
                  fontsize=10, fontweight='bold', fontproperties=make_font(10))
ax1.grid(True, alpha=0.3)

# 右：云量 vs 功率，按天气着色
ax2 = axes[1]
# 取白天有效数据
day_df = df[(df['solar_altitude_deg'] > 10) & (df['ghi'] > 50)].copy()
weather_groups = {
    'Clear Sky': ['Clear Sky'],
    'Few/Scattered': ['Few clouds', 'Scattered clouds'],
    'Broken': ['Broken clouds'],
    'Overcast': ['Overcast clouds'],
    'Rain/Storm': ['Light rain', 'Moderate rain', 'Thunderstorm', 'Drizzle', 'Fog'],
}
color_map = {
    'Clear Sky': 'gold',
    'Few/Scattered': 'green',
    'Broken': 'orange',
    'Overcast': 'gray',
    'Rain/Storm': 'blue',
}
for grp_name, grp_weathers in weather_groups.items():
    sub = day_df[day_df['weather_description'].isin(grp_weathers)]
    if len(sub) > 0:
        s = sub.sample(min(500, len(sub)), random_state=42)
        ax2.scatter(s['cloud_cover_pct'], s['power_kw'],
                    alpha=0.4, s=12, label=f'{grp_name} (n={len(sub)})',
                    color=color_map[grp_name], rasterized=True)
ax2.set_xlabel('Cloud Cover (%)', fontsize=10)
ax2.set_ylabel('Power (kW)', fontsize=10)
ax2.set_title('Cloud Cover vs Power (by Weather Type, daytime)\n(云量-功率关系，按天气分组)',
                  fontsize=10, fontweight='bold', fontproperties=make_font(10))
ax2.legend(fontsize=8, loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
ghi_cloud_path = CHART_DIR + r"\06_ghi_cloud_power_relations.png"
plt.savefig(ghi_cloud_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  已保存: {ghi_cloud_path}")

# ===================== 生成文字报告 =====================
print("\n生成文字分析报告...")

report_lines = []
report_lines.append("=" * 65)
report_lines.append("明月湖光伏数据集 — 特征分析报告")
report_lines.append("=" * 65)
report_lines.append("")

# 1. 相关性排名
report_lines.append("【一、功率相关性排名（按 Pearson 相关系数绝对值）】")
corr_with_power = corr_matrix['power_kw'].drop('power_kw').dropna().sort_values(key=abs, ascending=False)
for feat, corr_val in corr_with_power.items():
    n_bars = int(abs(corr_val) * 20)
    bar = '#' * n_bars
    sign = '+' if corr_val >= 0 else '-'
    report_lines.append(f"  {feat:30s}  r={corr_val:+.4f}  {sign}{bar}")
report_lines.append("")

# 2. top positive / negative correlations
top_pos = corr_with_power[corr_with_power > 0].head(5)
top_neg = corr_with_power[corr_with_power < 0].head(5)
report_lines.append("  正相关最强 (Top 5):")
for feat, corr_val in top_pos.items():
    report_lines.append(f"    {feat}: r = {corr_val:+.4f}")
report_lines.append("  负相关最强 (Top 5):")
for feat, corr_val in top_neg.items():
    report_lines.append(f"    {feat}: r = {corr_val:+.4f}")
report_lines.append("")

# 3. 按天气类型的功率统计
report_lines.append("【二、按天气类型的功率统计】")
weather_stats = df.groupby('weather_description')['power_kw'].agg(
    ['count', 'mean', 'median', 'std', 'min', 'max']
).round(3)
weather_stats = weather_stats.sort_values('mean', ascending=False)
report_lines.append(weather_stats.to_string())
report_lines.append("")

# 4. 功率分位数
report_lines.append("【三、功率分位数统计】")
quantiles = df['power_kw'].quantile([0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
for q, v in quantiles.items():
    report_lines.append(f"  {int(q*100):2d}%: {v:.4f} kW")
report_lines.append("")

# 5. 特征统计表
report_lines.append("【四、各特征统计摘要】")
stat_cols = ['power_kw', 'ghi', 'dhi', 'temperature_c', 'relative_humidity_pct',
             'cloud_cover_pct', 'solar_altitude_deg', 'solar_radiation_wm2',
             'wind_speed_ms', 'pressure_hpa', 'precip_rate_mmhr']
stat_labels = ['power_kw', 'ghi', 'dhi', 'temperature', 'humidity',
               'cloud', 'solar_alt', 'solar_rad', 'wind', 'pressure', 'precip']
stats = df[stat_cols].describe().round(2)
stats.index = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
stats.columns = stat_labels
report_lines.append(stats.to_string())
report_lines.append("")

# 6. 关键发现
report_lines.append("【五、关键发现】")
report_lines.append(f"  1. 与功率相关性最强的特征: {corr_with_power.abs().idxmax()} (r={corr_with_power.max():.3f})")
report_lines.append(f"  2. GHI 与功率相关系数: {corr_with_power['ghi']:.4f}")
report_lines.append(f"  3. DHI 与功率相关系数: {corr_with_power['dhi']:.4f}")
report_lines.append(f"  4. 云量与功率相关系数: {corr_with_power['cloud_cover_pct']:.4f}")
report_lines.append(f"  5. 太阳高度角与功率相关系数: {corr_with_power['solar_altitude_deg']:.4f}")
report_lines.append(f"  6. 平均功率: {df['power_kw'].mean():.2f} kW, 最大: {df['power_kw'].max():.2f} kW")
report_lines.append(f"  7. 装机容量利用小时数（日均值）: {df['power_kw'].mean() / 281.6 * 24:.1f} h/天")
report_lines.append(f"  8. 夜间功率>0记录: {len(df[(df['ghi']==0) & (df['power_kw']>0)])} 条（建议修正）")
report_lines.append(f"  9. 白天功率=0记录: {len(df[(df['ghi']>10) & (df['solar_altitude_deg']>0) & (df['power_kw']==0)])} 条（真实停机）")
report_lines.append("")
report_lines.append("【六、生成的图表文件】")
charts = [
    ("01_correlation_heatmap.png", "特征相关性热力图"),
    ("02_power_vs_features_scatter.png", "功率-关键特征散点图"),
    ("03_feature_distributions.png", "特征数据分布直方图"),
    ("04_power_by_weather.png", "按天气类型的功率分布"),
    ("05_daily_timeseries.png", "典型天气日功率与辐照时序"),
    ("06_ghi_cloud_power_relations.png", "云量-辐照-功率关系图"),
]
for fname, desc in charts:
    report_lines.append(f"  {fname}  —  {desc}")

report_content = "\n".join(report_lines)
report_path = REPORT_DIR + r"\feature_analysis_report.txt"
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_content)
print(f"  已保存: {report_path}")

print("\n全部图表和报告生成完毕!")
print(f"图表目录: {CHART_DIR}")
print(f"报告目录: {REPORT_DIR}")
