"""
===============================================================================
光伏发电功率预测 - 数据预处理脚本
===============================================================================
功能：整合所有数据预处理步骤
作者：MSC Project
日期：2026-09-06
===============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import PowerTransformer, StandardScaler

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ==============================================================================
# 路径配置
# ==============================================================================
DATA_PATH = r"c:\Users\MoYu\Desktop\pv_prediction_scheduling_msc_new\data\data-analysis\data\明月湖_cleaned.csv"
OUTPUT_PATH = r"c:\Users\MoYu\Desktop\pv_prediction_scheduling_msc_new\data\data-analysis\data\明月湖_cleaned.csv"
CHART_DIR = r"c:\Users\MoYu\Desktop\pv_prediction_scheduling_msc_new\data\data-analysis\charts"

# ==============================================================================
# 第一部分：天气描述编码
# ==============================================================================
def encode_weather_description(df):
    """
    将天气描述转换为云量分数（序数编码）
    
    设计原理：分数越高表示天气遮挡越严重，对光伏发电影响越大
    
    编码规则：
    - 0-4: 晴朗到阴天（无降水）
    - 5-8: 有降水天气（毛毛雨到大雨）
    - 9-10: 雷暴天气（最严重影响）
    """
    weather_score_map = {
        # ===== 晴天 =====
        'Clear Sky': 0,
        
        # ===== 少云 =====
        'Few clouds': 1,
        
        # ===== 疏云 =====
        'Scattered clouds': 2,
        
        # ===== 破裂云 =====
        'Broken clouds': 3,
        
        # ===== 阴天/雾 =====
        'Overcast clouds': 4,
        'Fog': 4,
        
        # ===== 有降水 =====
        'Drizzle': 5,           # 毛毛雨
        'Light rain': 6,        # 小雨
        'Moderate rain': 7,     # 中雨
        'Heavy rain': 8,        # 大雨
        
        # ===== 雷暴 =====
        'Thunderstorm with drizzle': 9,
        'Thunderstorm with light rain': 9,
        'Thunderstorm with rain': 10,
        'Thunderstorm with heavy rain': 10
    }
    
    # 应用编码
    df['weather_score'] = df['weather_description'].map(weather_score_map)
    
    # 检查未映射值
    unmapped = df['weather_score'].isna().sum()
    if unmapped > 0:
        print(f"  [警告] {unmapped} 条记录未找到对应编码")
    
    print(f"  ✓ 天气编码完成: weather_description → weather_score (0-10)")
    return df

# ==============================================================================
# 第二部分：白天/夜间分离
# ==============================================================================
def separate_day_night(df):
    """
    根据太阳高度角分离白天和夜间数据
    
    原理：
    - 白天（solar_altitude_deg >= 0）：太阳在地平线以上
    - 夜间（solar_altitude_deg < 0）：太阳在地平线以下
    
    处理：
    - 夜间功率直接置零（物理意义：夜间无光伏发电）
    - 新增 is_daytime 标记列
    """
    # 分离白天和夜间数据
    df_day = df[df['solar_altitude_deg'] >= 0].copy()
    df_night = df[df['solar_altitude_deg'] < 0].copy()
    
    print(f"  ✓ 数据分离完成")
    print(f"    - 白天记录: {len(df_day):,} 条 ({len(df_day)/len(df)*100:.1f}%)")
    print(f"    - 夜间记录: {len(df_night):,} 条 ({len(df_night)/len(df)*100:.1f}%)")
    
    # 夜间功率置零
    df_night['power_kw'] = 0
    
    # 标记
    df_day['is_daytime'] = 1
    df_night['is_daytime'] = 0
    
    # 合并数据
    df = pd.concat([df_day, df_night], ignore_index=True)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df

# ==============================================================================
# 第三部分：偏态分布校正
# ==============================================================================
def correct_skewness(df):
    """
    根据偏度程度选择合适的变换方法
    
    偏度等级标准：
    - |偏度| < 0.5: 正常，无需处理
    - 0.5 ≤ |偏度| < 1.0: 中等偏态，使用温和方法（平方根）
    - 1.0 ≤ |偏度| < 2.0: 高度偏态，使用较强方法（对数/Yeo-Johnson）
    - |偏度| ≥ 2.0: 极度偏态，使用最强方法（Yeo-Johnson）
    
    变换方法选择：
    - Yeo-Johnson: 自适应参数，适合极度偏态和含零值数据
    - 对数变换 (log1p): 适合高度右偏，公式 z = log(1+x)
    - 平方根变换 (sqrt): 适合中等右偏，公式 z = √x
    """
    
    # 定义各特征的变换方法
    # key: 特征名, value: (变换方法, 原始偏度, 偏态类型)
    transformation_config = {
        'power_kw': {
            'method': 'yeo-johnson',
            'original_skew': 2.345,
            'reason': '极度右偏，需要最强变换'
        },
        'uv': {
            'method': 'yeo-johnson',
            'original_skew': 1.867,
            'reason': '高度右偏，含大量零值'
        },
        'ghi': {
            'method': 'yeo-johnson',
            'original_skew': 0.636,
            'reason': '中等右偏，含零值'
        },
        'temperature_c': {
            'method': 'sqrt',
            'original_skew': 0.841,
            'reason': '中等右偏，温和变换即可'
        },
        'wind_gust_ms': {
            'method': 'log1p',
            'original_skew': 1.137,
            'reason': '高度右偏'
        },
        'relative_humidity_pct': {
            'method': 'yeo-johnson',
            'original_skew': -1.027,
            'reason': '高度左偏'
        }
    }
    
    print("  偏态校正详情：")
    print("  " + "-" * 75)
    print(f"  {'特征':<25} {'原始偏度':>10} {'变换方法':>18} {'最终偏度':>10}")
    print("  " + "-" * 75)
    
    for feat, config in transformation_config.items():
        orig_skew = stats.skew(df[feat].dropna())
        
        if config['method'] == 'yeo-johnson':
            # Yeo-Johnson 变换
            pt = PowerTransformer(method='yeo-johnson', standardize=True)
            df[feat] = pt.fit_transform(df[feat].values.reshape(-1, 1)).flatten()
            method_name = 'Yeo-Johnson'
            
        elif config['method'] == 'log1p':
            # 对数变换: log(1+x)
            df[feat] = np.log1p(df[feat])
            method_name = '对数 log(1+x)'
            
        elif config['method'] == 'sqrt':
            # 平方根变换
            df[feat] = np.sqrt(df[feat])
            method_name = '平方根 sqrt'
        
        new_skew = stats.skew(df[feat].dropna())
        print(f"  {feat:<25} {orig_skew:>10.3f} {method_name:>18} {new_skew:>10.3f}")
    
    print("  " + "-" * 75)
    print("  ✓ 偏态校正完成")
    
    return df

# ==============================================================================
# 第四部分：特征标准化
# ==============================================================================
def scale_features(df):
    """
    Z-score 标准化：将数据转换为均值为0，标准差为1
    
    公式：z = (x - μ) / σ
    
    注意：
    - 仅对连续数值特征进行标准化
    - 二值标记列（is_daytime, is_potential_shutdown）不进行标准化
    - 时间列（hour_of_day）不进行标准化
    """
    features_to_scale = [
        'temperature_c',
        'apparent_temperature_c',
        'relative_humidity_pct',
        'wind_gust_ms',
        'solar_altitude_deg',
        'dhi',
        'power_kw',
        'uv',
        'ghi',
        'weather_score'
    ]
    
    scaler = StandardScaler()
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    
    print("  ✓ 特征标准化完成 (Z-score)")
    print(f"    标准化特征数: {len(features_to_scale)}")
    
    return df

# ==============================================================================
# 第五部分：删除冗余列
# ==============================================================================
def remove_redundant_columns(df):
    """
    删除预处理过程中产生的冗余列和无需保留的列
    
    删除规则：
    - 旧变换中间结果 (*_transformed)
    - YJ变换中间结果 (*_yj)
    - 无预测价值的列 (weather_icon)
    - 已编码的原列 (weather_description)
    """
    columns_to_drop = [
        # 旧的变换结果（已被新变换替代）
        'power_kw_transformed',
        'uv_transformed',
        'wind_gust_ms_transformed',
        'temperature_c_transformed',
        'ghi_transformed',
        # YJ变换结果（已标准化）
        'power_kw_yj',
        'uv_yj',
        'ghi_yj',
        # 无用列
        'weather_description',
        'weather_icon'
    ]
    
    # 只删除存在的列
    existing_cols = [c for c in columns_to_drop if c in df.columns]
    df = df.drop(columns=existing_cols)
    
    print(f"  ✓ 冗余列清理完成")
    print(f"    已删除 {len(existing_cols)} 列")
    
    return df

# ==============================================================================
# 第六部分：数据验证
# ==============================================================================
def validate_data(df):
    """
    验证预处理结果
    """
    print("\n" + "=" * 75)
    print("数据验证报告")
    print("=" * 75)
    
    # 基础信息
    print(f"\n  [1] 基础信息")
    print(f"      记录数: {len(df):,}")
    print(f"      特征数: {len(df.columns)}")
    
    # 缺失值检查
    print(f"\n  [2] 缺失值检查")
    missing = df.isnull().sum().sum()
    print(f"      缺失值总数: {missing}")
    
    # 重复值检查
    print(f"\n  [3] 重复值检查")
    duplicates = df.duplicated().sum()
    print(f"      重复行数: {duplicates}")
    
    # 偏度检查
    print(f"\n  [4] 偏度检查")
    key_features = ['power_kw', 'uv', 'ghi', 'temperature_c', 
                    'relative_humidity_pct', 'wind_gust_ms', 
                    'solar_altitude_deg', 'dhi', 'apparent_temperature_c']
    
    print(f"      {'特征':<30} {'偏度':>10} {'状态':>10}")
    print(f"      {'-'*52}")
    
    all_ok = True
    for feat in key_features:
        if feat in df.columns:
            skew = stats.skew(df[feat].dropna())
            if abs(skew) < 0.5:
                status = "✅ 正常"
            elif abs(skew) < 1.0:
                status = "⚠️ 轻微"
                all_ok = False
            else:
                status = "❌ 需处理"
                all_ok = False
            print(f"      {feat:<30} {skew:>10.3f} {status:>10}")
    
    print(f"\n  结论: {'✅ 所有特征偏度正常' if all_ok else '⚠️ 部分特征偏度异常'}")
    
    # 最终列清单
    print(f"\n  [5] 最终列清单")
    print(f"      {'序号':>4}  {'列名':<30} {'处理状态':<20}")
    print(f"      {'-'*56}")
    for i, col in enumerate(df.columns, 1):
        if col == 'timestamp':
            status = '保留'
        elif col in ['is_daytime', 'is_potential_shutdown', 'hour_of_day']:
            status = '二值/时间标记'
        elif col in ['power_kw', 'uv', 'ghi', 'temperature_c', 'relative_humidity_pct']:
            status = '变换+标准化'
        elif col == 'weather_score':
            status = '编码+标准化'
        else:
            status = '标准化'
        print(f"      {i:>4}  {col:<30} {status:<20}")
    
    return df

# ==============================================================================
# 主函数
# ==============================================================================
def main():
    """
    数据预处理主流程
    
    处理步骤：
    1. 天气描述编码 → 云量分数
    2. 白天/夜间分离
    3. 偏态分布校正（分类变换）
    4. 特征标准化（Z-score）
    5. 删除冗余列
    6. 数据验证
    """
    print("=" * 75)
    print("光伏发电功率预测 - 数据预处理")
    print("=" * 75)
    
    # 读取数据
    print("\n[Step 1] 读取数据...")
    df = pd.read_csv(DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"  ✓ 数据加载完成: {len(df):,} 条记录, {len(df.columns)} 列")
    
    # Step 2: 天气编码
    print("\n[Step 2] 天气描述编码...")
    df = encode_weather_description(df)
    
    # Step 3: 白天/夜间分离
    print("\n[Step 3] 白天/夜间分离...")
    df = separate_day_night(df)
    
    # Step 4: 偏态校正
    print("\n[Step 4] 偏态分布校正...")
    df = correct_skewness(df)
    
    # Step 5: 特征标准化
    print("\n[Step 5] 特征标准化...")
    df = scale_features(df)
    
    # Step 6: 删除冗余列
    print("\n[Step 6] 删除冗余列...")
    df = remove_redundant_columns(df)
    
    # Step 7: 保存数据
    print("\n[Step 7] 保存数据...")
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"  ✓ 数据已保存: {OUTPUT_PATH}")
    
    # Step 8: 验证
    df = validate_data(df)
    
    print("\n" + "=" * 75)
    print("预处理完成！")
    print("=" * 75)
    
    return df

# ==============================================================================
# 入口
# ==============================================================================
if __name__ == "__main__":
    df = main()
