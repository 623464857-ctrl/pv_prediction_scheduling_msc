"""
===============================================================================
数据预处理状态检查脚本
===============================================================================
功能：检查当前数据集的预处理状态
===============================================================================
"""

import pandas as pd
import numpy as np
from scipy import stats

# 路径配置
DATA_PATH = r"c:\Users\MoYu\Desktop\pv_prediction_scheduling_msc_new\data\data-analysis\data\明月湖_cleaned.csv"

# 读取数据
df = pd.read_csv(DATA_PATH)

def print_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def check_basic_info():
    """基础信息检查"""
    print_header("基础信息")
    print(f"  记录数: {len(df):,}")
    print(f"  特征数: {len(df.columns)}")
    print(f"\n  列名清单:")
    for i, col in enumerate(df.columns, 1):
        print(f"    {i:2d}. {col}")

def check_missing_values():
    """缺失值检查"""
    print_header("缺失值检查")
    missing = df.isnull().sum()
    total_missing = missing.sum()
    
    if total_missing == 0:
        print("  [OK] 无缺失值")
    else:
        print(f"  [WARN] 缺失值总数: {total_missing}")
        for col, val in missing.items():
            if val > 0:
                print(f"      {col}: {val}")

def check_duplicates():
    """重复值检查"""
    print_header("重复值检查")
    duplicates = df.duplicated().sum()
    print(f"  重复行数: {duplicates}")

def check_skewness():
    """偏度检查"""
    print_header("偏度检查")
    
    # 需要检查的特征
    features = ['power_kw', 'uv', 'ghi', 'temperature_c', 
                'relative_humidity_pct', 'wind_gust_ms', 
                'solar_altitude_deg', 'dhi', 'apparent_temperature_c']
    
    print(f"  {'特征':<30} {'偏度':>10} {'状态':>15}")
    print(f"  {'-'*58}")
    
    all_normal = True
    for feat in features:
        if feat in df.columns:
            skew = stats.skew(df[feat].dropna())
            
            if abs(skew) < 0.5:
                status = "[OK]"
            elif abs(skew) < 1.0:
                status = "[WARN]"
                all_normal = False
            else:
                status = "[FAIL]"
                all_normal = False
            
            print(f"  {feat:<30} {skew:>10.3f} {status:>15}")
    
    if all_normal:
        print(f"\n  [OK] 所有特征偏度正常 (|偏度| < 0.5)")
    else:
        print(f"\n  [WARN] 部分特征偏度异常，建议重新处理")

def check_encoding():
    """编码检查"""
    print_header("编码状态检查")
    
    # 天气编码
    if 'weather_score' in df.columns:
        print(f"  [OK] weather_score: 已编码")
        print(f"      范围: {df['weather_score'].min():.1f} - {df['weather_score'].max():.1f}")
        print(f"      均值: {df['weather_score'].mean():.2f}")
    else:
        print(f"  [FAIL] weather_score: 未编码")
    
    if 'weather_description' in df.columns:
        print(f"  [WARN] weather_description: 仍存在（建议删除）")
    else:
        print(f"  [OK] weather_description: 已删除")
    
    # 天气图标
    if 'weather_icon' in df.columns:
        print(f"  [WARN] weather_icon: 仍存在（建议删除）")
    else:
        print(f"  [OK] weather_icon: 已删除")

def check_redundant_columns():
    """冗余列检查"""
    print_header("冗余列检查")
    
    # 检查 *_transformed 列
    transformed_cols = [c for c in df.columns if 'transformed' in c]
    # 检查 *_yj 列
    yj_cols = [c for c in df.columns if '_yj' in c]
    
    redundant = transformed_cols + yj_cols
    
    if redundant:
        print(f"  [WARN] 发现冗余列:")
        for c in redundant:
            print(f"      - {c}")
    else:
        print(f"  [OK] 无冗余列")

def check_day_night():
    """白天/夜间分离检查"""
    print_header("白天/夜间分离检查")
    
    if 'is_daytime' in df.columns:
        day_count = (df['is_daytime'] == 1).sum()
        night_count = (df['is_daytime'] == 0).sum()
        print(f"  [OK] is_daytime: 已添加")
        print(f"      白天: {day_count:,} 条 ({day_count/len(df)*100:.1f}%)")
        print(f"      夜间: {night_count:,} 条 ({night_count/len(df)*100:.1f}%)")
    else:
        print(f"  [FAIL] is_daytime: 未添加")

def check_scaling():
    """标准化检查"""
    print_header("标准化检查")
    
    # 应该已标准化的特征
    features = ['temperature_c', 'apparent_temperature_c', 'relative_humidity_pct',
                'wind_gust_ms', 'solar_altitude_deg', 'dhi', 
                'power_kw', 'uv', 'ghi', 'weather_score']
    
    print(f"  {'特征':<30} {'均值':>10} {'标准差':>10}")
    print(f"  {'-'*52}")
    
    all_scaled = True
    for feat in features:
        if feat in df.columns:
            mean_val = df[feat].mean()
            std_val = df[feat].std()
            
            # 判断是否已标准化（均值接近0，标准差接近1）
            is_scaled = abs(mean_val) < 0.1 and abs(std_val - 1) < 0.1
            
            if is_scaled:
                status = "[OK]"
            else:
                status = "[WARN]"
                all_scaled = False
            
            print(f"  {feat:<30} {mean_val:>10.3f} {std_val:>10.3f} {status}")
    
    if all_scaled:
        print(f"\n  [OK] 所有特征已标准化")
    else:
        print(f"\n  [WARN] 部分特征未标准化")

def print_summary():
    """汇总报告"""
    print_header("预处理状态汇总")
    
    checks = []
    
    # 检查1: 缺失值
    if df.isnull().sum().sum() == 0:
        checks.append(("缺失值", True, "无"))
    else:
        checks.append(("缺失值", False, f"{df.isnull().sum().sum()}个"))
    
    # 检查2: 偏度
    features = ['power_kw', 'uv', 'ghi', 'temperature_c', 'relative_humidity_pct']
    skew_issues = []
    for feat in features:
        if feat in df.columns:
            skew = abs(stats.skew(df[feat].dropna()))
            if skew >= 0.5:
                skew_issues.append(feat)
    if not skew_issues:
        checks.append(("偏度校正", True, "正常"))
    else:
        checks.append(("偏度校正", False, f"{len(skew_issues)}个异常"))
    
    # 检查3: 天气编码
    if 'weather_score' in df.columns and 'weather_description' not in df.columns:
        checks.append(("天气编码", True, "已完成"))
    else:
        checks.append(("天气编码", False, "未完成"))
    
    # 检查4: 白天/夜间分离
    if 'is_daytime' in df.columns:
        checks.append(("白天/夜间分离", True, "已完成"))
    else:
        checks.append(("白天/夜间分离", False, "未完成"))
    
    # 检查5: 冗余列
    redundant = [c for c in df.columns if 'transformed' in c or '_yj' in c]
    if not redundant:
        checks.append(("冗余列清理", True, "已完成"))
    else:
        checks.append(("冗余列清理", False, f"{len(redundant)}个"))
    
    # 打印检查结果
    print(f"\n  {'检查项':<20} {'状态':>10} {'说明':>20}")
    print(f"  {'-'*52}")
    for name, passed, detail in checks:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {name:<20} {status:>10} {detail:>20}")
    
    # 总体评价
    all_passed = all(c[1] for c in checks)
    print(f"\n  {'='*50}")
    if all_passed:
        print(f"  总体评价: [PASS] 预处理已完成，可以进行建模")
    else:
        print(f"  总体评价: [WARN] 预处理未完成，请检查上述问题")

def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("  数据预处理状态检查")
    print("=" * 70)
    print(f"\n  数据文件: {DATA_PATH}")
    
    check_basic_info()
    check_missing_values()
    check_duplicates()
    check_skewness()
    check_encoding()
    check_redundant_columns()
    check_day_night()
    check_scaling()
    print_summary()
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
