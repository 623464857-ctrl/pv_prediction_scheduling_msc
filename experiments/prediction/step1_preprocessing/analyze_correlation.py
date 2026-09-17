"""
相关性热力图分析脚本
分析原始特征（去除派生前）的相关性，识别冗余特征
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
OLD_DATA_PATH = Path("data/data-analysis/data/明月湖_cleaned.csv")

df = pd.read_csv(DATA_PATH)
old_df = pd.read_csv(OLD_DATA_PATH)

print("=" * 60)
print("新系统数据概览")
print("=" * 60)
print(f"形状: {df.shape}")
print(f"\n所有列名:\n{list(df.columns)}")

# 定义原始特征列（排除派生的、质量相关的、元数据的）
ORIGINAL_FEATURES = [
    'temperature_c',
    'apparent_temperature_c',
    'relative_humidity_pct',
    'wind_gust_ms',
    'solar_altitude_deg',
    'ghi_wm2',
    'dhi_wm2',
    'dni_wm2',
    'uv_index',
    'atmosphere_hpa',
    'wind_speed_ms',
    'wind_direction_deg',
    'cloud_cover_pct',
    'visibility_km',
]

# 只保留存在的列
available_features = [f for f in ORIGINAL_FEATURES if f in df.columns]
print(f"\n可用原始特征 ({len(available_features)}个):\n{available_features}")

# 提取原始特征数据
X = df[available_features].copy()

# 添加 power_kw 到数据中
if 'power_kw' in df.columns:
    X['power_kw'] = df['power_kw']

# 重命名列以便显示
display_names = {
    'temperature_c': 'temperature_c',
    'apparent_temperature_c': 'apparent_temp',
    'relative_humidity_pct': 'rh',
    'wind_gust_ms': 'wind_gust',
    'solar_altitude_deg': 'solar_alt',
    'ghi_wm2': 'ghi',
    'dhi_wm2': 'dhi',
    'dni_wm2': 'dni',
    'uv_index': 'uv',
    'atmosphere_hpa': 'atmosphere',
    'wind_speed_ms': 'wind_speed',
    'wind_direction_deg': 'wind_dir',
    'cloud_cover_pct': 'cloud',
    'visibility_km': 'visibility',
    'power_kw': 'power_kw',
}

X_display = X.rename(columns=display_names)

# 计算相关性矩阵
corr_matrix = X_display.corr()

# 绘制热力图
fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt='.2f',
    cmap='RdBu_r',
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.5,
    cbar_kws={'shrink': 0.8, 'label': 'Correlation'},
    ax=ax
)

ax.set_title('Feature Correlation Heatmap (Original Features Only)', fontsize=14, pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

# 保存图片
output_path = Path("data/prediction/step1_preprocessing/charts/feature_correlation_heatmap.png")
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\n热力图已保存: {output_path}")

# 分析高相关特征
print("\n" + "=" * 60)
print("高相关性特征分析 (|r| > 0.7)")
print("=" * 60)

high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.7:
            high_corr_pairs.append({
                'feature_1': corr_matrix.columns[i],
                'feature_2': corr_matrix.columns[j],
                'correlation': corr_matrix.iloc[i, j]
            })

if high_corr_pairs:
    high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('correlation', key=abs, ascending=False)
    print(high_corr_df.to_string(index=False))
else:
    print("没有发现高相关性特征对 (|r| > 0.7)")

# 分析极高相关性特征
print("\n" + "=" * 60)
print("极高相关性特征分析 (|r| > 0.9)")
print("=" * 60)

very_high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.9:
            very_high_corr_pairs.append({
                'feature_1': corr_matrix.columns[i],
                'feature_2': corr_matrix.columns[j],
                'correlation': corr_matrix.iloc[i, j]
            })

if very_high_corr_pairs:
    very_high_df = pd.DataFrame(very_high_corr_pairs).sort_values('correlation', key=abs, ascending=False)
    for _, row in very_high_df.iterrows():
        print(f"\n[!] [{row['feature_1']}] <-> [{row['feature_2']}]")
        print(f"   r = {row['correlation']:.3f}")
        print(f"   => 建议删除其中一个")
else:
    print("没有发现极高相关性特征对 (|r| > 0.9)")

# 与目标变量的相关性
print("\n" + "=" * 60)
print("特征与 power_kw 的相关性")
print("=" * 60)

if 'power_kw' in df.columns:
    target_corr = X_display.corrwith(df['power_kw']).abs().sort_values(ascending=False)
    print(target_corr.to_string())

# 特征重要性排名
print("\n" + "=" * 60)
print("特征重要性排名（与功率相关性）")
print("=" * 60)

if 'power_kw' in df.columns:
    corr_with_target = X_display.corrwith(df['power_kw'])
    corr_with_target = corr_with_target.abs().sort_values(ascending=False)
    
    print("\n高相关 (|r| > 0.5):")
    for feat, corr in corr_with_target.items():
        if corr > 0.5:
            print(f"  {feat}: {corr:.3f}")
    
    print("\n中相关 (0.3 < |r| <= 0.5):")
    for feat, corr in corr_with_target.items():
        if 0.3 < corr <= 0.5:
            print(f"  {feat}: {corr:.3f}")
    
    print("\n低相关 (|r| <= 0.3):")
    for feat, corr in corr_with_target.items():
        if corr <= 0.3:
            print(f"  {feat}: {corr:.3f}")

# 冗余特征建议
print("\n" + "=" * 60)
print("冗余特征去除建议")
print("=" * 60)

print("""
分析结论:
1. apparent_temperature_c 与 temperature_c: 高相关 → 建议保留 temperature_c，删除 apparent_temperature_c
2. solar_altitude_deg: 与 GHI 高度相关 → 可保留用于太阳位置特征
3. dni_wm2 与 ghi_wm2: 中等相关 → 可考虑删除 dni，保留 ghi
4. wind_speed_ms 与 wind_gust_ms: 高相关 → 建议保留 wind_gust，删除 wind_speed
""")

# 保存相关性数据
corr_output = Path("data/prediction/step1_preprocessing/processed/feature_correlation_matrix.csv")
corr_matrix.to_csv(corr_output)
print(f"\n相关性矩阵已保存: {corr_output}")
