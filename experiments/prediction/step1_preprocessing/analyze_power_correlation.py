"""
特征与功率相关性分析脚本
生成原始特征与 power_kw 的相关性热力图，分析可去除的特征
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 路径配置
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "prediction" / "step1_preprocessing"
PROCESSED_DIR = DATA_DIR / "processed" / "stations"
CHARTS_DIR = DATA_DIR / "charts"
CHARTS_DIR.mkdir(exist_ok=True)

def analyze_correlation():
    # 读取预处理后的数据
    df = pd.read_csv(PROCESSED_DIR / "Mingyuehu_1_preprocessed.csv")

    # 选择原始特征（排除派生特征、元数据、标志列）
    ORIGINAL_FEATURES = [
        'temperature_c',          # 温度
        'apparent_temperature_c', # 体感温度
        'ghi_wm2',                # 全球水平辐照度
        'dni_wm2',                # 直接法向辐照度
        'dhi_wm2',                # 散射水平辐照度
        'relative_humidity_pct',  # 相对湿度
        'atmosphere_hpa',         # 大气压力
        'wind_speed_ms',          # 风速
        'wind_direction_deg',     # 风向
        'wind_gust_ms',           # 阵风
        'solar_altitude_deg',     # 太阳高度角
        'solar_azimuth_deg',      # 太阳方位角
        'cloud_cover_pct',        # 云量
        'visibility_km',          # 能见度
        'uv_index',               # UV指数
        'hour',                   # 小时
        'power_kw',               # 功率（目标变量）
    ]

    # 只保留存在的列
    features_to_analyze = [f for f in ORIGINAL_FEATURES if f in df.columns]
    df_subset = df[features_to_analyze].copy()

    # 计算相关性矩阵
    corr_matrix = df_subset.corr()

    # 提取与 power_kw 的相关性
    power_corr = corr_matrix['power_kw'].drop('power_kw').sort_values(key=abs, ascending=False)

    print("=" * 60)
    print("特征与 power_kw 的相关性分析")
    print("=" * 60)
    print(f"\n{'特征':<25} {'相关系数':>10} {'|r|':>8} {'可去除?':<8}")
    print("-" * 60)

    # 相关性阈值
    LOW_THRESHOLD = 0.1   # |r| < 0.1 极低相关性
    MED_THRESHOLD = 0.3   # 0.1 <= |r| < 0.3 低相关性

    removable = []
    for feat, corr in power_corr.items():
        abs_corr = abs(corr)
        if abs_corr < LOW_THRESHOLD:
            flag = "✅ 建议去除"
            removable.append(feat)
        elif abs_corr < MED_THRESHOLD:
            flag = "⚠️ 可考虑"
            removable.append(feat)
        else:
            flag = "❌ 保留"
        print(f"{feat:<25} {corr:>10.4f} {abs_corr:>8.4f} {flag:<8}")

    print("-" * 60)
    print(f"\n建议去除的特征 ({len(removable)}个): {removable}")

    # ============= 生成热力图 =============

    # 1. 完整相关性热力图
    fig1, ax1 = plt.subplots(figsize=(14, 12))
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
        ax=ax1
    )
    ax1.set_title('Feature Correlation Heatmap (Original Features)\n原始特征相关性热力图', fontsize=14, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig1.savefig(CHARTS_DIR / "feature_correlation_heatmap_original.png", dpi=150, bbox_inches='tight')
    print(f"\n[保存] {CHARTS_DIR / 'feature_correlation_heatmap_original.png'}")

    # 2. 与 power_kw 相关性的条形图
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    colors = ['green' if abs(c) >= MED_THRESHOLD else ('orange' if abs(c) >= LOW_THRESHOLD else 'red')
               for c in power_corr.values]

    bars = ax2.barh(range(len(power_corr)), power_corr.values, color=colors, edgecolor='black', alpha=0.7)
    ax2.set_yticks(range(len(power_corr)))
    ax2.set_yticklabels(power_corr.index)
    ax2.set_xlabel('Correlation with power_kw', fontsize=12)
    ax2.set_title('Feature Correlation with Power (kw)\n特征与功率的相关性', fontsize=14, pad=20)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax2.axvline(x=LOW_THRESHOLD, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax2.axvline(x=-LOW_THRESHOLD, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax2.axvline(x=MED_THRESHOLD, color='orange', linestyle='--', linewidth=1, alpha=0.7)
    ax2.axvline(x=-MED_THRESHOLD, color='orange', linestyle='--', linewidth=1, alpha=0.7)
    ax2.set_xlim(-1.1, 1.1)

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', edgecolor='black', alpha=0.7, label=f'|r| >= {MED_THRESHOLD} (保留)'),
        Patch(facecolor='orange', edgecolor='black', alpha=0.7, label=f'{LOW_THRESHOLD} <= |r| < {MED_THRESHOLD} (可考虑)'),
        Patch(facecolor='red', edgecolor='black', alpha=0.7, label=f'|r| < {LOW_THRESHOLD} (建议去除)'),
    ]
    ax2.legend(handles=legend_elements, loc='lower right', fontsize=10)
    ax2.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    fig2.savefig(CHARTS_DIR / "power_correlation_barplot.png", dpi=150, bbox_inches='tight')
    print(f"[保存] {CHARTS_DIR / 'power_correlation_barplot.png'}")

    # 3. 高相关特征对热力图（特征间相关性，避免多重共线性）
    # 计算特征间的相关性（排除 power_kw）
    feat_corr_no_power = corr_matrix.drop('power_kw', axis=0).drop('power_kw', axis=1)

    # 找出高相关特征对 (|r| > 0.8)
    high_corr_pairs = []
    for i in range(len(feat_corr_no_power.columns)):
        for j in range(i+1, len(feat_corr_no_power.columns)):
            if abs(feat_corr_no_power.iloc[i, j]) > 0.8:
                high_corr_pairs.append((
                    feat_corr_no_power.columns[i],
                    feat_corr_no_power.columns[j],
                    feat_corr_no_power.iloc[i, j]
                ))

    if high_corr_pairs:
        print("\n" + "=" * 60)
        print("高相关特征对 (|r| > 0.8) - 可能存在多重共线性")
        print("=" * 60)
        for f1, f2, r in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
            print(f"  {f1:<25} <-> {f2:<25} r={r:.4f}")

        # 保存高相关特征对到文件
        pairs_df = pd.DataFrame(high_corr_pairs, columns=['Feature_1', 'Feature_2', 'Correlation'])
        pairs_df.to_csv(DATA_DIR / "processed" / "high_correlation_pairs.csv", index=False)

    plt.show()

    return removable, power_corr

if __name__ == "__main__":
    removable, power_corr = analyze_correlation()
