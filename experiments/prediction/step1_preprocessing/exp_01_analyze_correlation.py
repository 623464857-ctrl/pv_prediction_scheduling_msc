"""
特征相关性分析脚本
整合特征间相关性热力图和特征与功率相关性分析功能

功能：
1. 特征间相关性热力图
2. 特征与功率的相关性分析
3. 高相关特征对检测（多重共线性）
4. 冗余特征去除建议

使用方法：
    python -m experiments.prediction.step1_preprocessing.analyze_correlation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 路径配置
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "prediction" / "step1_preprocessing"
PROCESSED_DIR = DATA_DIR / "processed" / "stations"
CHARTS_DIR = DATA_DIR / "charts"
CHARTS_DIR.mkdir(exist_ok=True)


def load_data(dataset: str = "bozhou") -> pd.DataFrame:
    """加载预处理后的数据"""
    if dataset == "bozhou":
        df = pd.read_csv(PROCESSED_DIR / "Bozhou_1_preprocessed.csv")
    else:
        raise ValueError(f"不支持的数据集: {dataset}")
    return df


def analyze_feature_correlation(
    df: pd.DataFrame,
    output_dir: Optional[Path] = None,
    display_names: Optional[dict] = None,
    show_power: bool = True,
) -> pd.DataFrame:
    """
    分析特征间的相关性
    
    Args:
        df: 数据集
        output_dir: 输出目录
        display_names: 列名显示映射
        show_power: 是否包含功率列
        
    Returns:
        相关性矩阵 DataFrame
    """
    if output_dir is None:
        output_dir = CHARTS_DIR
    
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
    
    # 提取特征数据
    X = df[available_features].copy()
    
    # 添加功率列
    if show_power and 'power_kw' in df.columns:
        X['power_kw'] = df['power_kw']
    
    # 重命名列以便显示
    if display_names is None:
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
    
    X_display = X.rename(columns={k: v for k, v in display_names.items() if k in X.columns})
    
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
    output_path = output_dir / "feature_correlation_heatmap.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[保存] {output_path}")
    
    return corr_matrix


def analyze_power_correlation(
    df: pd.DataFrame,
    output_dir: Optional[Path] = None,
    power_col: str = 'power_kw',
    low_threshold: float = 0.1,
    med_threshold: float = 0.3,
) -> tuple[list, pd.Series]:
    """
    分析特征与功率的相关性
    
    Args:
        df: 数据集
        output_dir: 输出目录
        power_col: 功率列名
        low_threshold: 低相关性阈值
        med_threshold: 中相关性阈值
        
    Returns:
        (可去除特征列表, 与功率相关性序列)
    """
    if output_dir is None:
        output_dir = CHARTS_DIR
    
    # 定义原始特征列
    ORIGINAL_FEATURES = [
        'temperature_c', 'apparent_temperature_c', 'ghi_wm2', 'dni_wm2', 'dhi_wm2',
        'relative_humidity_pct', 'atmosphere_hpa', 'wind_speed_ms', 'wind_direction_deg',
        'wind_gust_ms', 'solar_altitude_deg', 'solar_azimuth_deg', 'cloud_cover_pct',
        'visibility_km', 'uv_index', 'hour',
    ]
    
    # 只保留存在的列
    features_to_analyze = [f for f in ORIGINAL_FEATURES if f in df.columns]
    if power_col not in df.columns:
        print(f"[警告] 找不到功率列: {power_col}")
        return [], pd.Series()
    
    df_subset = df[features_to_analyze + [power_col]].copy()
    
    # 计算相关性矩阵
    corr_matrix = df_subset.corr()
    
    # 提取与功率的相关性
    power_corr = corr_matrix[power_col].drop(power_col).sort_values(key=abs, ascending=False)
    
    print("=" * 60)
    print("特征与功率的相关性分析")
    print("=" * 60)
    print(f"\n{'特征':<25} {'相关系数':>10} {'|r|':>8} {'可去除?':<8}")
    print("-" * 60)
    
    removable = []
    for feat, corr in power_corr.items():
        abs_corr = abs(corr)
        if abs_corr < low_threshold:
            flag = "✅ 建议去除"
            removable.append(feat)
        elif abs_corr < med_threshold:
            flag = "⚠️ 可考虑"
            removable.append(feat)
        else:
            flag = "❌ 保留"
        print(f"{feat:<25} {corr:>10.4f} {abs_corr:>8.4f} {flag:<8}")
    
    print("-" * 60)
    print(f"\n建议去除的特征 ({len(removable)}个): {removable}")
    
    # 生成条形图
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    colors = ['green' if abs(c) >= med_threshold else ('orange' if abs(c) >= low_threshold else 'red')
               for c in power_corr.values]
    
    bars = ax2.barh(range(len(power_corr)), power_corr.values, color=colors, edgecolor='black', alpha=0.7)
    ax2.set_yticks(range(len(power_corr)))
    ax2.set_yticklabels(power_corr.index)
    ax2.set_xlabel('Correlation with power_kw', fontsize=12)
    ax2.set_title('Feature Correlation with Power (kw)\n特征与功率的相关性', fontsize=14, pad=20)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax2.axvline(x=low_threshold, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax2.axvline(x=-low_threshold, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax2.axvline(x=med_threshold, color='orange', linestyle='--', linewidth=1, alpha=0.7)
    ax2.axvline(x=-med_threshold, color='orange', linestyle='--', linewidth=1, alpha=0.7)
    ax2.set_xlim(-1.1, 1.1)
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', edgecolor='black', alpha=0.7, label=f'|r| >= {med_threshold} (保留)'),
        Patch(facecolor='orange', edgecolor='black', alpha=0.7, label=f'{low_threshold} <= |r| < {med_threshold} (可考虑)'),
        Patch(facecolor='red', edgecolor='black', alpha=0.7, label=f'|r| < {low_threshold} (建议去除)'),
    ]
    ax2.legend(handles=legend_elements, loc='lower right', fontsize=10)
    ax2.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    output_path = output_dir / "power_correlation_barplot.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[保存] {output_path}")
    
    return removable, power_corr


def detect_high_correlation_pairs(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.8,
) -> pd.DataFrame:
    """
    检测高相关特征对
    
    Args:
        corr_matrix: 相关性矩阵
        threshold: 高相关阈值
        
    Returns:
        高相关特征对 DataFrame
    """
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > threshold:
                high_corr_pairs.append({
                    'feature_1': corr_matrix.columns[i],
                    'feature_2': corr_matrix.columns[j],
                    'correlation': corr_val
                })
    
    if high_corr_pairs:
        df = pd.DataFrame(high_corr_pairs).sort_values('correlation', key=abs, ascending=False)
        return df
    return pd.DataFrame()


def analyze_redundancy(df: pd.DataFrame) -> None:
    """
    分析特征冗余性并给出建议
    
    Args:
        df: 数据集
    """
    print("\n" + "=" * 60)
    print("冗余特征去除建议")
    print("=" * 60)
    
    # 检测 apparent_temperature_c 与 temperature_c
    if 'apparent_temperature_c' in df.columns and 'temperature_c' in df.columns:
        corr = df['apparent_temperature_c'].corr(df['temperature_c'])
        if abs(corr) > 0.7:
            print(f"1. apparent_temperature_c 与 temperature_c: r={corr:.3f}")
            print("   -> 建议保留 temperature_c，删除 apparent_temperature_c")
    
    # 检测 dni_wm2 与 ghi_wm2
    if 'dni_wm2' in df.columns and 'ghi_wm2' in df.columns:
        corr = df['dni_wm2'].corr(df['ghi_wm2'])
        if abs(corr) > 0.5:
            print(f"2. dni_wm2 与 ghi_wm2: r={corr:.3f}")
            print("   -> 可考虑删除 dni，保留 ghi")
    
    # 检测 wind_speed_ms 与 wind_gust_ms
    if 'wind_speed_ms' in df.columns and 'wind_gust_ms' in df.columns:
        corr = df['wind_speed_ms'].corr(df['wind_gust_ms'])
        if abs(corr) > 0.7:
            print(f"3. wind_speed_ms 与 wind_gust_ms: r={corr:.3f}")
            print("   -> 建议保留 wind_gust，删除 wind_speed")


def analyze_feature_importance(df: pd.DataFrame, power_col: str = 'power_kw') -> None:
    """
    分析特征重要性排名
    
    Args:
        df: 数据集
        power_col: 功率列名
    """
    print("\n" + "=" * 60)
    print("特征重要性排名（与功率相关性）")
    print("=" * 60)
    
    if power_col not in df.columns:
        return
    
    # 获取数值型特征
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if power_col in numeric_cols:
        numeric_cols.remove(power_col)
    
    # 计算与功率的相关性
    corr_with_target = df[numeric_cols].corrwith(df[power_col]).abs().sort_values(ascending=False)
    
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


def run_full_analysis(dataset: str = "bozhou", save_output: bool = True) -> None:
    """
    运行完整的相关性分析
    
    Args:
        dataset: 数据集名称 ('bozhou')
        save_output: 是否保存输出
    """
    print("=" * 60)
    print(f"特征相关性分析 - 数据集: {dataset}")
    print("=" * 60)
    
    # 加载数据
    df = load_data(dataset)
    print(f"\n数据形状: {df.shape}")
    
    # 分析特征间相关性
    print("\n[1] 特征间相关性分析...")
    corr_matrix = analyze_feature_correlation(df, show_power=True)
    
    # 分析特征与功率相关性
    print("\n[2] 特征与功率相关性分析...")
    removable, power_corr = analyze_power_correlation(df)
    
    # 检测高相关特征对
    print("\n[3] 高相关性特征对分析 (|r| > 0.8)...")
    high_corr_df = detect_high_correlation_pairs(corr_matrix, threshold=0.8)
    if len(high_corr_df) > 0:
        print(high_corr_df.to_string(index=False))
    else:
        print("没有发现高相关性特征对 (|r| > 0.8)")
    
    # 极高相关性特征
    print("\n[4] 极高相关性特征对分析 (|r| > 0.9)...")
    very_high_df = detect_high_correlation_pairs(corr_matrix, threshold=0.9)
    if len(very_high_df) > 0:
        for _, row in very_high_df.iterrows():
            print(f"  [!] {row['feature_1']} <-> {row['feature_2']}: r = {row['correlation']:.3f}")
            print(f"      => 建议删除其中一个")
    else:
        print("没有发现极高相关性特征对 (|r| > 0.9)")
    
    # 分析冗余性
    print("\n[5] 冗余特征分析...")
    analyze_redundancy(df)
    
    # 特征重要性排名
    print("\n[6] 特征重要性排名...")
    analyze_feature_importance(df)
    
    # 保存相关性矩阵
    if save_output:
        corr_output = DATA_DIR / "processed" / "feature_correlation_matrix.csv"
        corr_matrix.to_csv(corr_output)
        print(f"\n相关性矩阵已保存: {corr_output}")
    
    print("\n" + "=" * 60)
    print("分析完成!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="特征相关性分析")
    parser.add_argument("--dataset", "-d", type=str, default="bozhou",
                        choices=["bozhou"],
                        help="数据集名称")
    parser.add_argument("--no-save", action="store_true",
                        help="不保存输出文件")
    
    args = parser.parse_args()
    
    run_full_analysis(
        dataset=args.dataset,
        save_output=not args.no_save
    )
