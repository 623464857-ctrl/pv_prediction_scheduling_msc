"""
===============================================================================
光伏发电功率预测 - 数据分析与可视化脚本
===============================================================================
功能：
  1. 数据集基本信息检查
  2. 偏度与分布分析
  3. 异常值检测
  4. 特征相关性分析
  5. 数据可视化（箱线图、周报图表）
===============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import sys
from scipy import stats
from matplotlib.patches import Patch

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==============================================================================
# 路径配置
# ==============================================================================
DATA_DIR = r"c:\Users\MoYu\Desktop\pv_prediction_scheduling_msc_new\data\data-analysis"
DATA_PATH = os.path.join(DATA_DIR, r"data\明月湖_cleaned.csv")
CHART_DIR = os.path.join(DATA_DIR, r"charts")
WEEKLY_DIR = os.path.join(CHART_DIR, r"weekly_feature_comparison")

# ==============================================================================
# 第一部分：数据集基本信息
# ==============================================================================
def show_basic_info():
    """显示数据集基本信息"""
    print("\n" + "=" * 70)
    print("  数据集基本信息")
    print("=" * 70)
    
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"\n  记录数: {len(df):,}")
    print(f"  特征数: {len(df.columns)}")
    
    print(f"\n  列名清单:")
    for i, col in enumerate(df.columns, 1):
        print(f"    {i:2d}. {col}")
    
    # 时间范围
    start = df['timestamp'].min()
    end = df['timestamp'].max()
    days = (end - start).days
    theoretical = days * 96
    
    print(f"\n  时间范围:")
    print(f"    起始: {start}")
    print(f"    结束: {end}")
    print(f"    跨度: {days} 天")
    print(f"    理论记录数: {theoretical}")
    print(f"    实际记录数: {len(df)}")
    print(f"    缺失记录: {theoretical - len(df)}")

# ==============================================================================
# 第二部分：偏度与分布分析
# ==============================================================================
def show_distribution_analysis():
    """显示偏度与分布分析"""
    print("\n" + "=" * 70)
    print("  偏度与分布分析")
    print("=" * 70)
    
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
    
    features = ['power_kw', 'uv', 'ghi', 'temperature_c', 
                'relative_humidity_pct', 'wind_gust_ms', 
                'solar_altitude_deg', 'dhi', 'apparent_temperature_c']
    
    print(f"\n  {'特征':<30} {'偏度':>10} {'峰度':>10} {'状态':>10}")
    print(f"  {'-'*62}")
    
    for feat in features:
        if feat in df.columns:
            skew = stats.skew(df[feat].dropna())
            kurt = stats.kurtosis(df[feat].dropna())
            
            if abs(skew) < 0.5:
                status = "[OK]"
            elif abs(skew) < 1.0:
                status = "[WARN]"
            else:
                status = "[FAIL]"
            
            print(f"  {feat:<30} {skew:>10.3f} {kurt:>10.3f} {status:>10}")
    
    # power_kw 分布详情
    print(f"\n  power_kw 分布详情:")
    zero_count = (df['power_kw'] == 0).sum()
    nonzero_count = (df['power_kw'] > 0).sum()
    print(f"    零值记录: {zero_count:,} ({zero_count/len(df)*100:.1f}%)")
    print(f"    非零记录: {nonzero_count:,} ({nonzero_count/len(df)*100:.1f}%)")
    print(f"    最小值: {df['power_kw'].min():.4f}")
    print(f"    最大值: {df['power_kw'].max():.4f}")
    print(f"    均值: {df['power_kw'].mean():.4f}")
    print(f"    中位数: {df['power_kw'].median():.4f}")
    print(f"    标准差: {df['power_kw'].std():.4f}")

# ==============================================================================
# 第三部分：异常值检测
# ==============================================================================
def show_anomaly_detection():
    """显示异常值检测结果"""
    print("\n" + "=" * 70)
    print("  异常值检测")
    print("=" * 70)
    
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 1. 夜间功率>0
    night_mask = (df['solar_altitude_deg'] < 0) & (df['power_kw'] > 0)
    night_count = night_mask.sum()
    print(f"\n  [1] 夜间异常功率 (solar_alt<0 且 power>0)")
    print(f"      异常记录数: {night_count}")
    if night_count > 0:
        print(f"      最大夜间功率: {df.loc[night_mask, 'power_kw'].max():.4f}")
    
    # 2. 白天GHI=0但功率>0
    day_ghi_zero = (df['solar_altitude_deg'] >= 0) & (df['ghi'] == 0) & (df['power_kw'] > 0)
    day_ghi_zero_count = day_ghi_zero.sum()
    print(f"\n  [2] 白天GHI=0异常 (day & ghi=0 & power>0)")
    print(f"      异常记录数: {day_ghi_zero_count}")
    
    # 3. 白天疑似停机
    day_stop = (df['solar_altitude_deg'] >= 0) & (df['ghi'] > 100) & (df['power_kw'] == 0)
    day_stop_count = day_stop.sum()
    print(f"\n  [3] 白天疑似停机 (day & ghi>100 & power=0)")
    print(f"      疑似停机记录: {day_stop_count}")
    
    # 4. 3σ原则异常值
    print(f"\n  [4] 3σ原则异常值检测")
    print(f"      {'特征':<30} {'异常数':>10} {'占比':>10}")
    print(f"      {'-'*52}")
    
    for col in ['power_kw', 'ghi', 'dhi', 'temperature_c', 'wind_gust_ms']:
        if col in df.columns:
            mean, std = df[col].mean(), df[col].std()
            if std == 0:
                continue
            low, high = mean - 3*std, mean + 3*std
            outliers = df[(df[col] < low) | (df[col] > high)][col]
            if len(outliers) > 0:
                pct = len(outliers)/len(df)*100
                print(f"      {col:<30} {len(outliers):>10} {pct:>9.1f}%")

# ==============================================================================
# 第四部分：特征相关性分析
# ==============================================================================
def show_correlation_analysis():
    """显示特征相关性分析"""
    print("\n" + "=" * 70)
    print("  特征相关性分析 (与 power_kw 的相关系数)")
    print("=" * 70)
    
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
    
    features = ['ghi', 'dhi', 'uv', 'solar_altitude_deg', 
                'temperature_c', 'apparent_temperature_c',
                'relative_humidity_pct', 'wind_gust_ms',
                'cloud_cover_pct', 'weather_score']
    
    features = [f for f in features if f in df.columns]
    
    print(f"\n  {'特征':<30} {'相关系数':>10} {'相关强度':>10}")
    print(f"  {'-'*52}")
    
    correlations = {}
    for feat in features:
        corr = df[feat].corr(df['power_kw'])
        correlations[feat] = corr
        
        if abs(corr) >= 0.7:
            strength = "强正相关" if corr > 0 else "强负相关"
        elif abs(corr) >= 0.4:
            strength = "中等正相关" if corr > 0 else "中等负相关"
        elif abs(corr) >= 0.2:
            strength = "弱正相关" if corr > 0 else "弱负相关"
        else:
            strength = "极弱/无相关"
        
        print(f"  {feat:<30} {corr:>10.4f} {strength:>10}")
    
    # 按相关性排序
    print(f"\n  相关性排名 (Top 5):")
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for i, (feat, corr) in enumerate(sorted_corr[:5], 1):
        print(f"    {i}. {feat}: r = {corr:.4f}")

# ==============================================================================
# 第五部分：天气类型分析
# ==============================================================================
def show_weather_analysis():
    """显示天气类型分析"""
    print("\n" + "=" * 70)
    print("  天气类型分析")
    print("=" * 70)
    
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
    
    if 'weather_description' in df.columns:
        print(f"\n  天气类型分布:")
        weather_counts = df['weather_description'].value_counts()
        for weather, count in weather_counts.items():
            pct = count / len(df) * 100
            mean_power = df[df['weather_description'] == weather]['power_kw'].mean()
            print(f"    {weather:<30} {count:>5} ({pct:>5.1f}%)  平均功率: {mean_power:.3f}")
    
    # is_daytime 分布
    if 'is_daytime' in df.columns:
        day_count = (df['is_daytime'] == 1).sum()
        night_count = (df['is_daytime'] == 0).sum()
        print(f"\n  白天/夜间分布:")
        print(f"    白天: {day_count:,} 条 ({day_count/len(df)*100:.1f}%)")
        print(f"    夜间: {night_count:,} 条 ({night_count/len(df)*100:.1f}%)")

# ==============================================================================
# 第六部分：生成特征箱线图
# ==============================================================================
def generate_boxplots():
    """生成特征箱线图"""
    print("\n" + "=" * 70)
    print("  生成特征箱线图")
    print("=" * 70)
    
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
    
    # 选择数值特征
    feature_cols = [
        'power_kw', 'ghi', 'dhi', 'temperature_c',
        'apparent_temperature_c', 'relative_humidity_pct',
        'wind_gust_ms', 'solar_altitude_deg', 'uv'
    ]
    
    feature_cols = [c for c in feature_cols if c in df.columns]
    features_data = [df[col].dropna().values for col in feature_cols]
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(14, 8))
    bp = ax.boxplot(features_data, patch_artist=True, labels=feature_cols)
    
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
    ax.set_title('Feature Boxplot Analysis\nMingyue Lake PV Dataset', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='x', rotation=30)
    
    plt.tight_layout()
    output_path = os.path.join(CHART_DIR, 'feature_boxplots.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  箱线图已保存: {output_path}")
    
    # 输出统计信息
    print(f"\n  特征统计摘要:")
    print("  " + "-" * 80)
    stats_df = df[feature_cols].describe()
    print(stats_df.round(2).to_string())

# ==============================================================================
# 第七部分：生成周报图表
# ==============================================================================
def generate_weekly_charts():
    """生成按周的对比曲线图"""
    print("\n" + "=" * 70)
    print("  生成周报图表")
    print("=" * 70)
    
    # 创建输出目录
    os.makedirs(WEEKLY_DIR, exist_ok=True)
    
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['week'] = df['timestamp'].dt.isocalendar().week
    df['year_week'] = df['timestamp'].dt.strftime('%Y-W%W')
    
    # 天气颜色映射
    weather_colors = {
        'Clear Sky': '#FFF9C4',
        'Few clouds': '#C8E6C9',
        'Scattered clouds': '#B2DFDB',
        'Broken clouds': '#B3E5FC',
        'Overcast clouds': '#E0E0E0',
        'Fog': '#D7CCC8',
        'Drizzle': '#B0BEC5',
        'Light rain': '#90CAF9',
        'Moderate rain': '#64B5F6',
        'Heavy rain': '#42A5F5',
        'Thunderstorm with drizzle': '#CE93D8',
        'Thunderstorm with light rain': '#BA68C8',
        'Thunderstorm with rain': '#9C27B0',
        'Thunderstorm with heavy rain': '#7B1FA2',
    }
    
    weeks = sorted(df['year_week'].unique())
    print(f"  共找到 {len(weeks)} 周数据")
    
    # 要绑定的特征
    features = {
        'power_kw': {'label': 'Power (kW)', 'color': '#E53935'},
        'ghi': {'label': 'GHI (W/m2)', 'color': '#FFA726'},
        'temperature_c': {'label': 'Temp (C)', 'color': '#26A69A'},
        'solar_altitude_deg': {'label': 'Solar Alt (deg)', 'color': '#5C6BC0'},
        'uv': {'label': 'UV Index', 'color': '#AB47BC'},
    }
    
    for week in weeks:
        week_data = df[df['year_week'] == week].copy()
        
        if len(week_data) == 0:
            continue
        
        fig, axes = plt.subplots(len(features), 1, figsize=(16, 3*len(features)), sharex=True)
        if len(features) == 1:
            axes = [axes]
        
        fig.suptitle(f'Week {week} - PV Power & Key Features\n({week_data["timestamp"].min().strftime("%Y-%m-%d")} to {week_data["timestamp"].max().strftime("%Y-%m-%d")})', 
                     fontsize=14, fontweight='bold', y=1.02)
        
        start_date = week_data['timestamp'].min()
        end_date = week_data['timestamp'].max()
        
        for idx, (feat_name, feat_info) in enumerate(features.items()):
            ax = axes[idx]
            
            # 绘制天气背景带
            current_time = start_date
            while current_time <= end_date:
                mask = week_data['timestamp'] == current_time
                if mask.any():
                    weather = week_data.loc[mask, 'weather_description'].values[0]
                    color = weather_colors.get(weather, '#FFFFFF')
                    next_time = current_time + pd.Timedelta(minutes=15)
                    ax.axvspan(current_time, next_time, alpha=0.5, color=color, linewidth=0)
                current_time += pd.Timedelta(minutes=15)
            
            # 绘制特征曲线
            ax.plot(week_data['timestamp'], week_data[feat_name], 
                    color=feat_info['color'], linewidth=1.2, label=feat_info['label'])
            
            ax.set_ylabel(feat_info['label'], fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xlim(start_date, end_date)
            ax.legend(loc='upper right', fontsize=8)
        
        # 设置x轴格式
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        axes[-1].xaxis.set_major_locator(mdates.DayLocator())
        axes[-1].xaxis.set_minor_locator(mdates.HourLocator(interval=6))
        plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha='right')
        axes[-1].set_xlabel('Time', fontsize=11)
        
        # 添加天气图例
        legend_elements = [Patch(facecolor=color, alpha=0.7, label=weather) 
                          for weather, color in weather_colors.items() 
                          if weather in week_data['weather_description'].values]
        if legend_elements:
            fig.legend(handles=legend_elements, loc='lower center', ncol=min(7, len(legend_elements)), 
                       fontsize=8, bbox_to_anchor=(0.5, -0.02), title='Weather Types')
        
        plt.tight_layout()
        
        filename = f"week_{week.replace('-', '_')}.png"
        filepath = os.path.join(WEEKLY_DIR, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  已保存: {filename}")
    
    print(f"\n  完成！共生成 {len(weeks)} 张周报图表")
    print(f"  保存位置: {WEEKLY_DIR}")

# ==============================================================================
# 主函数
# ==============================================================================
def main():
    """数据分析主函数"""
    print("\n" + "=" * 70)
    print("  光伏发电功率预测 - 数据分析与可视化")
    print("=" * 70)
    
    # 显示分析结果
    show_basic_info()
    show_distribution_analysis()
    show_anomaly_detection()
    show_correlation_analysis()
    show_weather_analysis()
    
    # 生成图表
    generate_boxplots()
    generate_weekly_charts()
    
    print("\n" + "=" * 70)
    print("  分析完成！")
    print("=" * 70)

if __name__ == "__main__":
    main()
