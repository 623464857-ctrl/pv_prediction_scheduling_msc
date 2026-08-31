"""
Site1 数据优化脚本 - EXP-P01-v2
======================
功能：
1. 湿度数据质量分层
2. 辐照-功率异常检测
3. 功率突变平滑
4. 设备停机检测
5. WRF辐照偏差校正
6. 新增质量特征
7. 物理一致性验证

运行：
python experiments/prediction/step1_data_cleaning_alignment/run_exp_p01_v2_optimization.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# ============================================================================
# 路径与常量
# ============================================================================
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[3]
RAW_SITE1 = PROJECT_ROOT / "data" / "raw" / "Solar station site 1 (Nominal capacity-50MW).csv"
WRF_PATH = PROJECT_ROOT / "data" / "prediction" / "step1_preprocessing" / "processed" / "stations" / "Site_1_wrf_features.csv"
OUT_DIR = PROJECT_ROOT / "data" / "prediction" / "step1_preprocessing" / "processed" / "stations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CAPACITY_MW = 50.0
EXPECTED_FREQ = "15min"

# ============================================================================
# 日志设置
# ============================================================================
def setup_logging() -> logging.Logger:
    LOG_DIR = PROJECT_ROOT / "logs" / "prediction" / "step1_data_cleaning_alignment"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "EXP-P01-v2_optimization.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )
    return logging.getLogger("EXP-P01-v2")


# ============================================================================
# 1. 辐照-功率异常检测
# ============================================================================
def detect_irradiance_power_anomalies(df: pd.DataFrame, capacity_mw: float) -> pd.DataFrame:
    """
    检测辐照-功率不一致的异常
    
    异常类型：
    1. 晴空无功率：辐照>500但功率<5%容量
    2. 阴天高功率：辐照<100但功率>40%容量  
    3. 夜间发电：辐照<5但功率>1%容量
    4. 严重晴空无功率：辐照>700但功率<10%容量（中午时段）
    """
    df = df.copy()
    
    # 辐照和功率
    gti = df['total_irradiance_wm2']
    power = df['power_mw']
    hour = df['hour']
    
    # 异常1: 晴空无功率
    clear_sky = gti > 500
    no_power = power < 0.05 * capacity_mw
    df['anomaly_clear_sky_no_power'] = (clear_sky & no_power).astype(np.int8)
    
    # 异常2: 阴天高功率
    cloudy = gti < 100
    high_power = power > 0.4 * capacity_mw
    df['anomaly_cloudy_high_power'] = (cloudy & high_power).astype(np.int8)
    
    # 异常3: 夜间发电
    night = gti < 5
    power_exists = power > 0.01 * capacity_mw
    df['anomaly_night_power'] = (night & power_exists).astype(np.int8)
    
    # 异常4: 中午晴空无功率（更严重）
    noon_hours = hour.between(10, 14)
    severe_clear_no_power = (gti > 700) & (power < 0.1 * capacity_mw) & noon_hours
    df['anomaly_severe_clear_no_power'] = severe_clear_no_power.astype(np.int8)
    
    # 计算异常严重程度
    df['irradiance_power_anomaly_level'] = 0  # 正常
    df.loc[df['anomaly_night_power'] == 1, 'irradiance_power_anomaly_level'] = 3  # 最严重
    df.loc[df['anomaly_severe_clear_no_power'] == 1, 'irradiance_power_anomaly_level'] = 2  # 严重
    df.loc[(df['anomaly_clear_sky_no_power'] == 1) | (df['anomaly_cloudy_high_power'] == 1), 
           'irradiance_power_anomaly_level'] = 1  # 轻度
    
    return df


# ============================================================================
# 2. 功率突变平滑
# ============================================================================
def smooth_power_anomalies(df: pd.DataFrame, capacity_mw: float, threshold_pct: float = 0.30) -> pd.DataFrame:
    """
    检测并平滑功率突变
    
    1. 检测突变点：15分钟变化>threshold_pct*容量
    2. 如果是突变（非渐变），进行约束或平滑
    """
    df = df.copy()
    
    # 计算功率变化率
    df['power_diff'] = df['power_mw'].diff()
    df['power_diff_pct'] = df['power_diff'].abs() / capacity_mw
    
    # 检测突变
    df['power_sudden_change'] = 0
    df.loc[df['power_diff_pct'] > threshold_pct, 'power_sudden_change'] = 1
    
    # 标记：渐变(0) vs 突变(1)
    # 日间且辐照充足时的突变更可疑
    daytime = df['total_irradiance_wm2'] > 200
    df.loc[(df['power_sudden_change'] == 1) & daytime, 'power_sudden_change_suspicious'] = 1
    df['power_sudden_change_suspicious'] = df.get('power_sudden_change_suspicious', 0).fillna(0).astype(np.int8)
    
    # 对突变进行处理：替换为前后时刻均值
    power_corrected = df['power_mw'].copy()
    for idx in df[df['power_sudden_change'] == 1].index:
        if idx > 0 and idx < len(df) - 1:
            prev_power = df.loc[idx - 1, 'power_mw']
            next_power = df.loc[idx + 1, 'power_mw']
            # 如果前后功率相近，取均值；否则取较小的
            if abs(prev_power - next_power) < 0.2 * capacity_mw:
                power_corrected.loc[idx] = (prev_power + next_power) / 2
            else:
                power_corrected.loc[idx] = min(prev_power, next_power)
    
    df['power_mw_corrected'] = power_corrected
    
    return df


# ============================================================================
# 3. 设备停机检测
# ============================================================================
def detect_equipment_outage(df: pd.DataFrame, capacity_mw: float, min_zero_hours: int = 24) -> pd.DataFrame:
    """
    检测设备停机：
    1. 连续零功率 > min_zero_hours
    2. 返回停机时段列表并标记
    """
    df = df.copy()
    min_zero_points = min_zero_hours * 4  # 15分钟间隔
    
    # 识别连续零功率段
    zero_mask = df['power_mw'] < 0.01 * capacity_mw
    
    # 找出停机开始和结束
    df['in_outage'] = 0
    in_outage = False
    start_idx = None
    
    for idx in range(len(df)):
        if zero_mask.iloc[idx] and not in_outage:
            in_outage = True
            start_idx = idx
        elif not zero_mask.iloc[idx] and in_outage:
            duration = idx - start_idx
            if duration >= min_zero_points:
                df.loc[start_idx:idx-1, 'in_outage'] = 1
            in_outage = False
    
    # 检查结尾是否有停机
    if in_outage:
        duration = len(df) - start_idx
        if duration >= min_zero_points:
            df.loc[start_idx:, 'in_outage'] = 1
    
    return df


# ============================================================================
# 4. 湿度数据质量分层
# ============================================================================
def classify_humidity_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    湿度数据质量分层
    
    质量等级：
    3 = 高质量：湿度在合理范围(20-80%)且变化平稳
    2 = 中等质量：湿度在(10-100%)范围
    1 = 低质量：湿度<10%或>90%
    0 = 不可用：湿度>100%（物理不可能）或湿度数据本身缺失
    """
    df = df.copy()
    rh = df['relative_humidity_pct']
    
    # 基础质量分层
    df['humidity_quality_tier'] = 3  # 默认高质量
    
    df.loc[rh > 100, 'humidity_quality_tier'] = 0  # 物理不可能
    df.loc[rh < 10, 'humidity_quality_tier'] = 1   # 极端干燥
    df.loc[(rh >= 10) & (rh < 20), 'humidity_quality_tier'] = 1  # 很低湿度
    df.loc[(rh >= 20) & (rh < 50), 'humidity_quality_tier'] = 2  # 中等
    df.loc[rh >= 90, 'humidity_quality_tier'] = 1  # 很高湿度
    
    # 如果湿度被标记为invalid（>100），设为0
    if 'relative_humidity_pct_invalid_flag' in df.columns:
        df.loc[df['relative_humidity_pct_invalid_flag'] == 1, 'humidity_quality_tier'] = 0
    
    return df


# ============================================================================
# 5. 辐照质量评估
# ============================================================================
def evaluate_irradiance_quality(df: pd.DataFrame, capacity_mw: float) -> pd.DataFrame:
    """
    辐照数据质量评估
    
    1. 计算理论最大GHI（简化版：与太阳高度角相关）
    2. 辐照利用率 = 实际/理论
    3. 波动度 = 滚动标准差/均值
    """
    df = df.copy()
    
    # 简化理论最大GHI（中午满辐照约1000 W/m²）
    hour = df['hour'] + df['minute'] / 60
    df['theoretical_max_gti'] = 1000 * np.maximum(0, np.sin(np.pi * (hour - 6) / 12))
    df['theoretical_max_gti'] = df['theoretical_max_gti'].clip(lower=0)
    
    # 辐照利用率
    df['gti_utilization'] = df['total_irradiance_wm2'] / (df['theoretical_max_gti'] + 1)
    df['gti_utilization'] = df['gti_utilization'].clip(lower=0, upper=1.5)
    
    # 辐照波动度（滚动4步，约1小时）
    gti = df['total_irradiance_wm2']
    roll_mean = gti.rolling(4, min_periods=1).mean()
    roll_std = gti.rolling(4, min_periods=1).std()
    df['gti_volatility'] = roll_std / (roll_mean + 1)
    
    # 辐照质量等级
    df['irradiance_quality_tier'] = 3  # 默认高质量
    
    # 低质量：超过理论值10%以上（除非是早晚时刻）
    df.loc[df['gti_utilization'] > 1.1, 'irradiance_quality_tier'] = 1
    
    # 中等质量：利用率<20%且是日间（可能辐照传感器问题）
    daytime = df['hour'].between(9, 17)
    low_util_daytime = (df['gti_utilization'] < 0.2) & daytime
    df.loc[low_util_daytime, 'irradiance_quality_tier'] = 2
    
    # 低质量：波动剧烈（云层快速变化可能是正常的，这里标记但不删除）
    high_volatility = df['gti_volatility'] > 1.0
    df.loc[high_volatility, 'irradiance_quality_tier'] = 2
    
    return df


# ============================================================================
# 6. WRF辐照偏差校正（忽略时间偏移）
# ============================================================================
def correct_wrf_irradiance(df_obs: pd.DataFrame, df_wrf: pd.DataFrame, 
                           window_days: int = 7) -> pd.DataFrame:
    """
    使用滑动窗口对WRF辐照进行偏差校正
    
    方法：
    1. 计算每日实测-WRF偏差
    2. 使用平滑偏差校正WRF
    """
    df_merged = pd.merge(
        df_obs[['timestamp', 'total_irradiance_wm2', 'hour']],
        df_wrf[['timestamp', 'wrf_gti_wm2']],
        on='timestamp', how='inner',
        suffixes=('_obs', '_wrf')
    )
    
    # 只对日间数据进行校正
    daytime = df_merged['hour'].between(8, 17)
    df_daytime = df_merged[daytime].copy()
    
    # 计算日累积辐照比值
    daily_ratio = df_daytime.groupby(df_daytime['timestamp'].dt.date).apply(
        lambda x: (x['total_irradiance_wm2'].sum() + 1) / (x['wrf_gti_wm2'].sum() + 1)
    )
    
    # 平滑偏差（滚动窗口）
    daily_ratio.index = pd.to_datetime(daily_ratio.index)
    daily_ratio_smooth = daily_ratio.rolling(window_days, center=True, min_periods=1).mean()
    
    # 应用校正因子
    df_merged['wrf_gti_corrected'] = df_merged['wrf_gti_wm2'].copy()
    for date, ratio in daily_ratio_smooth.items():
        mask = df_merged['timestamp'].dt.date == date
        df_merged.loc[mask, 'wrf_gti_corrected'] = df_merged.loc[mask, 'wrf_gti_wm2'] * ratio
    
    return df_merged


# ============================================================================
# 7. 综合数据质量评分
# ============================================================================
def compute_overall_quality_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算综合数据质量评分（0-1）
    
    考虑因素：
    - 辐照-功率异常级别
    - 湿度质量等级
    - 辐照质量等级
    - 是否在设备停机期间
    - 是否被插值填充
    """
    df = df.copy()
    
    # 基础分数
    quality_score = 1.0
    
    # 辐照-功率异常扣分
    anomaly_penalty = {
        0: 0.0,   # 无异常
        1: 0.1,   # 轻度
        2: 0.3,   # 严重
        3: 0.5    # 明显错误
    }
    for level, penalty in anomaly_penalty.items():
        mask = df['irradiance_power_anomaly_level'] == level
        quality_score = quality_score - mask.astype(float) * penalty * (quality_score > 0)
    
    # 湿度质量扣分
    humidity_penalty = {3: 0.0, 2: 0.05, 1: 0.15, 0: 0.3}
    for tier, penalty in humidity_penalty.items():
        mask = df['humidity_quality_tier'] == tier
        quality_score = quality_score - mask.astype(float) * penalty * (quality_score > 0)
    
    # 辐照质量扣分
    irradiance_penalty = {3: 0.0, 2: 0.05, 1: 0.15}
    for tier, penalty in irradiance_penalty.items():
        mask = df['irradiance_quality_tier'] == tier
        quality_score = quality_score - mask.astype(float) * penalty * (quality_score > 0)
    
    # 设备停机扣分
    if 'in_outage' in df.columns:
        quality_score = quality_score - df['in_outage'].astype(float) * 0.2 * (quality_score > 0)
    
    # 插值数据扣分
    if 'imputed_feature_count' in df.columns:
        imputed_mask = df['imputed_feature_count'] > 2
        quality_score = quality_score - imputed_mask.astype(float) * 0.1 * (quality_score > 0)
    
    # 限制范围
    df['overall_quality_score'] = quality_score.clip(lower=0, upper=1)
    
    # 质量分级
    df['quality_grade'] = 'A'  # 优质
    df.loc[df['overall_quality_score'] < 0.7, 'quality_grade'] = 'B'  # 可用
    df.loc[df['overall_quality_score'] < 0.5, 'quality_grade'] = 'C'  # 低质
    df.loc[df['overall_quality_score'] < 0.3, 'quality_grade'] = 'D'  # 差
    
    return df


# ============================================================================
# 8. 物理一致性验证
# ============================================================================
def validate_physics_consistency(df: pd.DataFrame, capacity_mw: float, 
                                 logger: logging.Logger) -> dict:
    """
    物理一致性检验清单
    """
    issues = {}
    
    # 1. 夜间无功率（严重）
    night_mask = df['total_irradiance_wm2'] < 5
    night_power = df.loc[night_mask, 'power_mw']
    night_power_count = (night_power > 0.01 * capacity_mw).sum()
    issues['夜间存在非零功率'] = night_power_count
    if night_power_count > 0:
        logger.warning(f"夜间存在非零功率: {night_power_count}条")
    
    # 2. 功率超过容量（轻微，1.05倍以内可接受）
    over_capacity = (df['power_mw'] > 1.05 * capacity_mw).sum()
    issues['超容量功率'] = over_capacity
    if over_capacity > 0:
        logger.warning(f"存在超容量功率: {over_capacity}条")
    
    # 3. 晴空高辐照应有发电（严重）
    clear_mask = (df['total_irradiance_wm2'] > 700) & (df['hour'].between(10, 14))
    clear_no_power = df.loc[clear_mask, 'power_mw'] < 0.3 * capacity_mw
    clear_no_power_count = clear_no_power.sum()
    issues['晴空低发电'] = clear_no_power_count
    if clear_no_power_count > 0:
        logger.warning(f"晴空高辐照但低发电: {clear_no_power_count}条")
    
    # 4. 辐照超过理论最大值
    theoretical_max = df['theoretical_max_gti'] * 1.05
    over_theoretical = (df['total_irradiance_wm2'] > theoretical_max).sum()
    issues['辐照超理论值'] = over_theoretical
    
    # 5. 功率为负（不应该有）
    negative_power = (df['power_mw'] < 0).sum()
    issues['负功率'] = negative_power
    if negative_power > 0:
        logger.warning(f"存在负功率: {negative_power}条")
    
    # 6. 湿度超过100%（物理不可能）
    rh_over_100 = (df['relative_humidity_pct'] > 100).sum()
    issues['湿度超100%'] = rh_over_100
    if rh_over_100 > 0:
        logger.warning(f"湿度超过100%: {rh_over_100}条")
    
    return issues


# ============================================================================
# 9. 质量统计报告
# ============================================================================
def generate_quality_report(df: pd.DataFrame, capacity_mw: float) -> dict:
    """
    生成数据质量统计报告
    """
    report = {}
    
    # 基础统计
    report['总数据量'] = len(df)
    report['时间范围'] = f"{df['timestamp'].min()} 到 {df['timestamp'].max()}"
    
    # 质量分布
    quality_dist = df['overall_quality_score'].describe()
    report['质量分均值'] = quality_dist['mean']
    report['质量分中位数'] = quality_dist['50%']
    report['质量分标准差'] = quality_dist['std']
    
    # 质量等级分布
    grade_counts = df['quality_grade'].value_counts()
    report['质量等级分布'] = grade_counts.to_dict()
    
    # 异常检测统计
    report['辐照-功率异常_夜间发电'] = df['anomaly_night_power'].sum()
    report['辐照-功率异常_晴空无功率'] = df['anomaly_clear_sky_no_power'].sum()
    report['辐照-功率异常_阴天高功率'] = df['anomaly_cloudy_high_power'].sum()
    report['辐照-功率异常_严重晴空'] = df['anomaly_severe_clear_no_power'].sum()
    
    # 设备停机
    if 'in_outage' in df.columns:
        report['设备停机时段数'] = df['in_outage'].sum()
    
    # 湿度质量
    report['湿度高质量占比'] = (df['humidity_quality_tier'] == 3).mean() * 100
    report['湿度低质量占比'] = (df['humidity_quality_tier'] <= 1).mean() * 100
    
    # 辐照质量
    report['辐照高质量占比'] = (df['irradiance_quality_tier'] == 3).mean() * 100
    report['辐照低质量占比'] = (df['irradiance_quality_tier'] <= 1).mean() * 100
    
    # 容量利用率
    report['最大功率'] = df['power_mw'].max()
    report['容量利用率'] = df['power_mw'].max() / capacity_mw * 100
    
    return report


# ============================================================================
# 主函数
# ============================================================================
def main():
    logger = setup_logging()
    logger.info("=" * 80)
    logger.info("EXP-P01-v2: Site1 数据优化")
    logger.info("=" * 80)
    
    # =========================================================================
    # Step 1: 读取原始预处理数据
    # =========================================================================
    logger.info("[Step 1] 读取预处理后的Site1数据...")
    
    site1_path = OUT_DIR / "Site_1_preprocessed.csv"
    if not site1_path.exists():
        logger.error(f"找不到预处理数据: {site1_path}")
        logger.error("请先运行 run_exp_p01_preprocessing.py 生成预处理数据")
        sys.exit(1)
    
    df = pd.read_csv(site1_path, parse_dates=['timestamp'])
    logger.info(f"读取数据: {len(df)} 行, {len(df.columns)} 列")
    
    # =========================================================================
    # Step 2: 辐照-功率异常检测
    # =========================================================================
    logger.info("[Step 2] 辐照-功率异常检测...")
    df = detect_irradiance_power_anomalies(df, CAPACITY_MW)
    
    anomaly_counts = {
        '夜间发电': df['anomaly_night_power'].sum(),
        '晴空无功率': df['anomaly_clear_sky_no_power'].sum(),
        '阴天高功率': df['anomaly_cloudy_high_power'].sum(),
        '严重晴空': df['anomaly_severe_clear_no_power'].sum()
    }
    logger.info(f"  检测到异常: {anomaly_counts}")
    
    # =========================================================================
    # Step 3: 功率突变检测
    # =========================================================================
    logger.info("[Step 3] 功率突变检测与平滑...")
    df = smooth_power_anomalies(df, CAPACITY_MW)
    sudden_count = df['power_sudden_change'].sum()
    suspicious_count = df.get('power_sudden_change_suspicious', pd.Series([0])).sum()
    logger.info(f"  检测到突变: {sudden_count} 次, 其中可疑(白天): {suspicious_count} 次")
    
    # =========================================================================
    # Step 4: 设备停机检测
    # =========================================================================
    logger.info("[Step 4] 设备停机检测...")
    df = detect_equipment_outage(df, CAPACITY_MW)
    outage_count = df['in_outage'].sum()
    logger.info(f"  检测到停机期间: {outage_count} 个时间点")
    
    # =========================================================================
    # Step 5: 湿度质量分层
    # =========================================================================
    logger.info("[Step 5] 湿度数据质量分层...")
    df = classify_humidity_quality(df)
    humidity_dist = df['humidity_quality_tier'].value_counts().sort_index()
    logger.info(f"  湿度质量分布: {humidity_dist.to_dict()}")
    
    # =========================================================================
    # Step 6: 辐照质量评估
    # =========================================================================
    logger.info("[Step 6] 辐照数据质量评估...")
    df = evaluate_irradiance_quality(df, CAPACITY_MW)
    irradiance_dist = df['irradiance_quality_tier'].value_counts().sort_index()
    logger.info(f"  辐照质量分布: {irradiance_dist.to_dict()}")
    
    # =========================================================================
    # Step 7: WRF辐照偏差校正
    # =========================================================================
    logger.info("[Step 7] WRF辐照偏差校正...")
    if WRF_PATH.exists():
        df_wrf = pd.read_csv(WRF_PATH, parse_dates=['timestamp'])
        df_wrf_corrected = correct_wrf_irradiance(df, df_wrf)
        logger.info(f"  WRF校正完成")
    else:
        logger.warning(f"  WRF文件不存在: {WRF_PATH}, 跳过WRF校正")
    
    # =========================================================================
    # Step 8: 综合质量评分
    # =========================================================================
    logger.info("[Step 8] 计算综合数据质量评分...")
    df = compute_overall_quality_score(df)
    quality_mean = df['overall_quality_score'].mean()
    quality_grade_dist = df['quality_grade'].value_counts()
    logger.info(f"  平均质量分: {quality_mean:.4f}")
    logger.info(f"  质量等级分布: {quality_grade_dist.to_dict()}")
    
    # =========================================================================
    # Step 9: 物理一致性验证
    # =========================================================================
    logger.info("[Step 9] 物理一致性验证...")
    issues = validate_physics_consistency(df, CAPACITY_MW, logger)
    
    # =========================================================================
    # Step 10: 生成质量报告
    # =========================================================================
    logger.info("[Step 10] 生成数据质量报告...")
    report = generate_quality_report(df, CAPACITY_MW)
    
    print("\n" + "=" * 80)
    print("Site1 数据优化报告")
    print("=" * 80)
    for key, value in report.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        elif isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # =========================================================================
    # Step 11: 保存优化后的数据
    # =========================================================================
    logger.info("[Step 11] 保存优化后的数据...")
    output_path = OUT_DIR / "Site_1_optimized.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"  已保存到: {output_path}")
    
    # 保存质量报告
    report_path = OUT_DIR / "Site_1_quality_report.csv"
    pd.DataFrame([report]).to_csv(report_path, index=False)
    logger.info(f"  质量报告已保存到: {report_path}")
    
    logger.info("=" * 80)
    logger.info("EXP-P01-v2 完成!")
    logger.info("=" * 80)
    
    return df, report


if __name__ == "__main__":
    df_optimized, report = main()
