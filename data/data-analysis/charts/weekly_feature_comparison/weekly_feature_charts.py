"""
按周生成光伏发电与重要特征的对比曲线图
每个星期一张图，背景按天气类型着色
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from matplotlib.patches import Patch
import numpy as np

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
data_path = r"c:\Users\MoYu\Desktop\pv_prediction_scheduling_msc_new\data\data-analysis\data\明月湖_cleaned.csv"
output_dir = r"c:\Users\MoYu\Desktop\pv_prediction_scheduling_msc_new\data\data-analysis\charts\weekly_feature_comparison"

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 读取数据
df = pd.read_csv(data_path)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['week'] = df['timestamp'].dt.isocalendar().week
df['year_week'] = df['timestamp'].dt.strftime('%Y-W%W')

# 天气类型颜色映射（浅色背景）
weather_colors = {
    'Clear Sky': '#FFF9C4',           # 浅黄
    'Few clouds': '#C8E6C9',          # 浅绿
    'Scattered clouds': '#B2DFDB',    # 浅青
    'Broken clouds': '#B3E5FC',       # 浅蓝
    'Overcast clouds': '#E0E0E0',     # 浅灰
    'Fog': '#D7CCC8',                 # 浅棕
    'Drizzle': '#B0BEC5',             # 蓝灰
    'Light rain': '#90CAF9',          # 浅蓝
    'Moderate rain': '#64B5F6',       # 中蓝
    'Heavy rain': '#42A5F5',          # 深蓝
    'Thunderstorm with drizzle': '#CE93D8',      # 浅紫
    'Thunderstorm with light rain': '#BA68C8',   # 紫
    'Thunderstorm with rain': '#9C27B0',          # 深紫
    'Thunderstorm with heavy rain': '#7B1FA2',   # 更深紫
}

# 获取所有唯一的星期
weeks = df['year_week'].unique()
weeks = sorted(weeks)

print(f"共找到 {len(weeks)} 周数据")
print(f"输出目录: {output_dir}")

# 要绑定的特征
features = {
    'power_kw': {'label': '光伏发电 (kW)', 'color': '#E53935', 'ylim': (0, None)},
    'ghi': {'label': 'GHI (W/m2)', 'color': '#FFA726', 'ylim': (0, None)},
    'temperature_c': {'label': '温度 (°C)', 'color': '#26A69A', 'ylim': (None, None)},
    'solar_altitude_deg': {'label': '太阳高度角 (°)', 'color': '#5C6BC0', 'ylim': (None, None)},
    'uv': {'label': 'UV指数', 'color': '#AB47BC', 'ylim': (0, None)},
}

for week in weeks:
    week_data = df[df['year_week'] == week].copy()
    
    if len(week_data) == 0:
        continue
    
    # 创建图表
    fig, axes = plt.subplots(len(features), 1, figsize=(16, 3*len(features)), sharex=True)
    if len(features) == 1:
        axes = [axes]
    
    fig.suptitle(f'星期 {week} 光伏发电与重要特征对比\n({week_data["timestamp"].min().strftime("%Y-%m-%d")} 至 {week_data["timestamp"].max().strftime("%Y-%m-%d")})', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    # 获取该周的日期范围
    start_date = week_data['timestamp'].min()
    end_date = week_data['timestamp'].max()
    
    # 为每个子图添加天气背景
    for idx, (feat_name, feat_info) in enumerate(features.items()):
        ax = axes[idx]
        
        # 绘制天气背景带
        current_time = start_date
        while current_time <= end_date:
            # 获取该时间点的天气
            mask = week_data['timestamp'] == current_time
            if mask.any():
                weather = week_data.loc[mask, 'weather_description'].values[0]
                color = weather_colors.get(weather, '#FFFFFF')
                
                # 绘制背景矩形（15分钟间隔）
                next_time = current_time + pd.Timedelta(minutes=15)
                ax.axvspan(current_time, next_time, alpha=0.5, color=color, linewidth=0)
            
            current_time += pd.Timedelta(minutes=15)
        
        # 绘制特征曲线
        ax.plot(week_data['timestamp'], week_data[feat_name], 
                color=feat_info['color'], linewidth=1.2, label=feat_info['label'])
        
        ax.set_ylabel(feat_info['label'], fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(start_date, end_date)
        
        if feat_info['ylim'][0] is not None:
            ax.set_ylim(bottom=feat_info['ylim'][0])
        if feat_info['ylim'][1] is not None:
            ymax = week_data[feat_name].max() * 1.1
            ax.set_ylim(top=ymax)
        
        # 添加图例
        ax.legend(loc='upper right', fontsize=8)
    
    # 设置x轴格式
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    axes[-1].xaxis.set_major_locator(mdates.DayLocator())
    axes[-1].xaxis.set_minor_locator(mdates.HourLocator(interval=6))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    axes[-1].set_xlabel('时间', fontsize=11)
    
    # 添加图例说明天气颜色
    legend_elements = [Patch(facecolor=color, alpha=0.7, label=weather) 
                       for weather, color in weather_colors.items() if weather in week_data['weather_description'].values]
    if legend_elements:
        fig.legend(handles=legend_elements, loc='lower center', ncol=min(7, len(legend_elements)), 
                   fontsize=8, bbox_to_anchor=(0.5, -0.02), title='天气类型背景色')
    
    plt.tight_layout()
    
    # 保存图表
    filename = f"week_{week.replace('-', '_')}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"已保存: {filename}")

print(f"\n完成！共生成 {len(weeks)} 张周报图表")
print(f"保存位置: {output_dir}")
