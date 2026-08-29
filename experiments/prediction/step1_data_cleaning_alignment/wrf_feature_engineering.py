"""
Site 4 Weather Forecast Feature Engineering
============================================
实验编号: EXP-P01-WRF
功能: 将Open-Meteo天气预报数据(温度、湿度、天气代码)合并到 Site 4 预处理数据中
输入:
    - data/raw/open-meteo-40.81N114.80E769m.csv (天气预报)
    - data/prediction/step1_preprocessing/processed/stations/Site_4_preprocessed.csv (站点数据)
输出:
    - data/prediction/step1_preprocessing/processed/stations/Site_4_with_wrf.csv
    - data/prediction/step1_preprocessing/processed/stations/Site_4_wrf_features.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('logs/prediction/wrf_feature_engineering.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)
logger = logging.getLogger(__name__)

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH
while PROJECT_ROOT.name != 'experiments' and PROJECT_ROOT.parent != PROJECT_ROOT:
    PROJECT_ROOT = PROJECT_ROOT.parent
PROJECT_ROOT = PROJECT_ROOT.parent

RAW_WRF_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "prediction" / "step1_preprocessing" / "processed" / "stations"
DEFAULT_SITE_ID = 4


def site_paths(site_id: int) -> tuple[Path, Path, Path]:
    key = f"Site_{site_id}"
    return (
        PROCESSED_DIR / f"{key}_preprocessed.csv",
        PROCESSED_DIR / f"{key}_with_wrf.csv",
        PROCESSED_DIR / f"{key}_wrf_features.csv",
    )


def load_weather_forecast() -> pd.DataFrame:
    """加载Open-Meteo天气预报数据（不含dew_point，使用湿球温度估算）"""
    wrf_path = RAW_WRF_DIR / "open-meteo-40.81N114.80E769m.csv"
    logger.info(f"加载天气预报数据: {wrf_path}")

    # 新CSV格式: timestamp, tsi, gti, temperature_2m, relative_humidity, weather_code
    df = pd.read_csv(wrf_path, on_bad_lines='skip')

    # 重命名列
    df.columns = ['timestamp', 'tsi_wm2', 'gti_wm2',
                  'wrf_temperature_c', 'wrf_relative_humidity_pct', 'wrf_weather_code']

    # 转换时间戳
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601', errors='coerce')
    df = df.dropna(subset=['timestamp'])

    # 转换为数值类型
    for col in ['wrf_temperature_c', 'wrf_relative_humidity_pct', 'wrf_weather_code',
                'tsi_wm2', 'gti_wm2']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 填充缺失值
    df['wrf_weather_code'] = df['wrf_weather_code'].fillna(0).astype(int)
    df['wrf_relative_humidity_pct'] = df['wrf_relative_humidity_pct'].fillna(50)
    df['wrf_temperature_c'] = df['wrf_temperature_c'].ffill().bfill()
    df['tsi_wm2'] = df['tsi_wm2'].fillna(0)
    df['gti_wm2'] = df['gti_wm2'].fillna(0)

    logger.info(f"天气预报数据: {len(df)} 行, 时间范围 {df['timestamp'].min()} 到 {df['timestamp'].max()}")
    logger.info(f"温度范围: {df['wrf_temperature_c'].min():.1f} ~ {df['wrf_temperature_c'].max():.1f} °C")
    logger.info(f"湿度范围: {df['wrf_relative_humidity_pct'].min():.0f} ~ {df['wrf_relative_humidity_pct'].max():.0f} %")
    logger.info(f"天气代码分布: {df['wrf_weather_code'].value_counts().to_dict()}")

    return df


def load_site_data(site_id: int = DEFAULT_SITE_ID) -> pd.DataFrame:
    """加载指定站点预处理数据"""
    site_path, _, _ = site_paths(site_id)
    logger.info(f"加载 Site {site_id} 数据: {site_path}")

    df = pd.read_csv(site_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    logger.info(f"Site {site_id} 数据: {len(df)} 行, 时间范围 {df['timestamp'].min()} 到 {df['timestamp'].max()}")

    return df


def estimate_dew_point(temp_c: pd.Series, rh_pct: pd.Series) -> pd.Series:
    """使用 Magnus 公式估算露点温度（温度低于0°C使用改进公式）"""
    gamma = np.log(rh_pct / 100.0) + (17.62 * temp_c) / (243.12 + temp_c)
    dew_point = (243.12 * gamma) / (17.62 - gamma)
    return dew_point


def create_wrf_features(wrf_df: pd.DataFrame) -> pd.DataFrame:
    """基于天气预报创建光伏预测特征"""

    T = wrf_df['wrf_temperature_c']
    RH = wrf_df['wrf_relative_humidity_pct']
    weather_code = wrf_df['wrf_weather_code']
    gti = wrf_df['gti_wm2']
    tsi = wrf_df['tsi_wm2']

    # ── 1. 露点温度估算 ────────────────────────────────────────────────
    wrf_df['wrf_dew_point_c'] = estimate_dew_point(T, RH)

    # ── 2. 温度特征 ─────────────────────────────────────────────────────
    wrf_df['wrf_temperature_squared'] = T ** 2
    wrf_df['wrf_temperature_cubed'] = T ** 3
    wrf_df['wrf_temperature_abs'] = T.abs()

    # 温升率（相对于前一时步，°C/15min）
    wrf_df['wrf_temperature_diff'] = T.diff().fillna(0)
    wrf_df['wrf_temperature_ema_4'] = T.ewm(span=4, adjust=False).mean()

    # ── 3. 湿度特征 ─────────────────────────────────────────────────────
    wrf_df['wrf_humidity_squared'] = RH ** 2
    wrf_df['wrf_humidity_diff'] = RH.diff().fillna(0)

    # 湿度等级（0-4）
    wrf_df['wrf_humidity_level'] = pd.cut(RH, bins=[0, 30, 50, 70, 90, 100],
                                            labels=[0, 1, 2, 3, 4]).astype(float).fillna(2)

    # ── 4. 温湿度交互指数 ───────────────────────────────────────────────
    # 热指数（Heat Index, Rothfusz 回归）
    hi = (
        0.5 * (T + 61.0 + (T - 68.0) * 1.2 + RH * 0.094)
    )
    # 高温时用完整公式
    mask_high = T >= 27
    hi_full = (
        -8.785
        + 1.611 * T
        + 2.339 * RH
        - 0.146 * T * RH
        - 0.012 * T ** 2
        - 0.0164 * RH ** 2
        + 0.002 * T ** 2 * RH
        + 0.001 * T * RH ** 2
        - 0.0000393 * T ** 2 * RH ** 2
    )
    wrf_df['wrf_heat_index'] = np.where(mask_high, hi_full, hi)

    # 舒不适指数（温度×湿度）
    wrf_df['wrf_temp_humidity_product'] = T * RH

    # 归一化温湿度指数
    wrf_df['wrf_discomfort_index'] = (T - 0.55 * (1 - RH / 100) * (T - 14.5))

    # ── 5. 大气稳定度指标 ──────────────────────────────────────────────
    # 露点温度差（温度-露点），越大越不稳定
    wrf_df['wrf_dew_point_spread'] = T - wrf_df['wrf_dew_point_c']
    wrf_df['wrf_dew_point_spread_sq'] = wrf_df['wrf_dew_point_spread'] ** 2

    # K指数（大气垂直稳定度）
    # K = T_850 - T_500 + Td_850 - (T_700 - Td_700)
    # 这里没有分层数据，用简化：温度日较差 + 湿度日较差近似
    wrf_df['wrf_atmospheric_stability'] = (
        wrf_df['wrf_dew_point_spread'] * 0.5 + RH * 0.1 - T.abs() * 0.05
    )

    # ── 6. 天气代码分类 ─────────────────────────────────────────────────
    # WMO天气代码分组
    wrf_df['wrf_weather_category'] = pd.cut(
        weather_code,
        bins=[-1, 0, 3, 48, 57, 67, 77, 82, 86, 99, 999],
        labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    ).astype(float).fillna(0).astype(int)

    # 晴朗（天气代码0）
    wrf_df['wrf_clear_sky_flag'] = (weather_code == 0).astype(int)

    # 有云/多云（1-3）
    wrf_df['wrf_cloudy_flag'] = ((weather_code >= 1) & (weather_code <= 3)).astype(int)

    # 雾/霾（45-48）
    wrf_df['wrf_fog_flag'] = ((weather_code >= 45) & (weather_code <= 48)).astype(int)

    # 毛毛雨（51-57）
    wrf_df['wrf_drizzle_flag'] = ((weather_code >= 51) & (weather_code <= 57)).astype(int)

    # 降雨（61-67）
    wrf_df['wrf_rain_flag'] = ((weather_code >= 61) & (weather_code <= 67)).astype(int)

    # 降雪（71-77）
    wrf_df['wrf_snow_flag'] = ((weather_code >= 71) & (weather_code <= 77)).astype(int)

    # 阵雨（80-82）
    wrf_df['wrf_shower_flag'] = ((weather_code >= 80) & (weather_code <= 82)).astype(int)

    # 雷暴（95-99）
    wrf_df['wrf_thunderstorm_flag'] = (weather_code >= 95).astype(int)

    # 任何降水（毛毛雨/雨/雪/阵雨）
    wrf_df['wrf_any_precipitation_flag'] = (
        ((weather_code >= 51) & (weather_code <= 86)) | (weather_code >= 95)
    ).astype(int)

    # 云量估算（基于天气代码）
    cloud_cover_map = {
        0: 0, 1: 1, 2: 3, 3: 5,
        45: 9, 48: 9,
        51: 7, 53: 8, 55: 9, 56: 9, 57: 9,
        61: 7, 63: 8, 65: 9, 66: 9, 67: 9,
        71: 6, 73: 7, 75: 8, 77: 6,
        80: 6, 81: 7, 82: 9,
        85: 7, 86: 8,
        95: 8, 96: 9, 99: 10,
    }
    wrf_df['wrf_cloud_cover_oktas'] = weather_code.map(
        lambda x: cloud_cover_map.get(int(x), 5)
    )

    # 天空遮蔽比例（0-1）
    wrf_df['wrf_cloud_cover_ratio'] = wrf_df['wrf_cloud_cover_oktas'] / 10.0

    # ── 7. 辐照度特征 ───────────────────────────────────────────────────
    # 总太阳辐照度（预报）
    wrf_df['wrf_tsi_wm2'] = tsi
    # 倾斜面总辐照度（预报）
    wrf_df['wrf_gti_wm2'] = gti
    # 晴空辐照度估算（用温度反推的晴空指数）
    wrf_df['wrf_clearness_index'] = np.where(tsi > 0, gti / (tsi + 1e-6), 0)
    wrf_df['wrf_clearness_index'] = wrf_df['wrf_clearness_index'].clip(0, 1)

    # 云层衰减因子（天气代码影响辐照度）
    cloud_attenuation = {
        0: 1.0, 1: 0.95, 2: 0.85, 3: 0.75,
        45: 0.3, 48: 0.2,
        51: 0.7, 53: 0.6, 55: 0.5, 56: 0.4, 57: 0.3,
        61: 0.5, 63: 0.4, 65: 0.3, 66: 0.25, 67: 0.2,
        71: 0.6, 73: 0.5, 75: 0.4, 77: 0.5,
        80: 0.6, 81: 0.5, 82: 0.35,
        85: 0.5, 86: 0.4,
        95: 0.3, 96: 0.2, 99: 0.1,
    }
    wrf_df['wrf_cloud_attenuation_factor'] = weather_code.map(
        lambda x: cloud_attenuation.get(int(x), 0.7)
    )

    # 辐照度变化（用于检测云层移动）
    wrf_df['wrf_gti_diff'] = gti.diff().fillna(0)
    wrf_df['wrf_gti_ema_4'] = gti.ewm(span=4, adjust=False).mean()
    wrf_df['wrf_gti_std_8'] = gti.rolling(8, min_periods=1).std()

    # ── 8. 温度-辐照交互特征（光伏预测核心） ───────────────────────────
    # 高温对光伏效率的影响（温度每升高1°C，效率约下降0.4%）
    wrf_df['wrf_pv_temperature_factor'] = 1.0 - 0.004 * np.maximum(T - 25, 0)
    # 晴空指数与温度的交互
    wrf_df['wrf_clearness_temperature'] = wrf_df['wrf_clearness_index'] * T
    # 云量与温度的交互
    wrf_df['wrf_cloud_cover_temperature'] = wrf_df['wrf_cloud_cover_ratio'] * T
    # 辐照度与湿度的交互
    wrf_df['wrf_gti_humidity'] = gti * (1 - RH / 100)

    # ── 9. 联合云-气象指标 ─────────────────────────────────────────────
    # 云层-降水联合指标
    wrf_df['wrf_cloud_precip_index'] = (
        wrf_df['wrf_cloud_cover_ratio'] + wrf_df['wrf_any_precipitation_flag'] * 0.5
    )

    # 气象不确定性指数（基于天气代码范围）
    uncertain_codes = [3, 45, 48, 51, 53, 61, 63, 80, 82, 85, 95, 96]
    wrf_df['wrf_weather_uncertainty'] = weather_code.apply(
        lambda x: 0.2 if x in uncertain_codes else 0.1
    )

    # 温度突变指标（可能导致辐照度剧变）
    wrf_df['wrf_temperature_surge'] = (
        wrf_df['wrf_temperature_diff'].abs() + wrf_df['wrf_humidity_diff'].abs() * 0.5
    )

    # ── 10. 时序滚动特征（短期趋势） ──────────────────────────────────
    wrf_df['wrf_temperature_rolling_std_4'] = T.rolling(4, min_periods=1).std()
    wrf_df['wrf_humidity_rolling_std_4'] = RH.rolling(4, min_periods=1).std()
    wrf_df['wrf_gti_rolling_max_4'] = gti.rolling(4, min_periods=1).max()

    n_features = len([c for c in wrf_df.columns if c not in ['timestamp', 'tsi_wm2', 'gti_wm2']])
    logger.info(f"创建了 {n_features} 个WRF特征")

    return wrf_df


def merge_and_save(site_df: pd.DataFrame, wrf_df: pd.DataFrame, output_dir: Path, site_id: int = DEFAULT_SITE_ID):
    """合并数据并保存"""

    logger.info(f"合并天气预报特征到 Site {site_id} 数据...")

    merged = site_df.merge(wrf_df, on='timestamp', how='left')

    n_original = len(site_df)
    n_merged = len(merged)
    n_wrf_matched = merged['wrf_temperature_c'].notna().sum()

    logger.info(f"合并结果: {n_merged} 行, 天气预报匹配 {n_wrf_matched}/{n_original} ({100*n_wrf_matched/n_original:.1f}%)")

    if n_wrf_matched < n_original * 0.9:
        logger.warning("天气预报数据覆盖率较低，可能存在时间戳对齐问题！")

    _, output_path, features_path = site_paths(site_id)
    merged.to_csv(output_path, index=False)
    logger.info(f"已保存: {output_path}")

    wrf_feature_cols = [col for col in merged.columns if col.startswith('wrf_')]
    wrf_features_df = merged[['timestamp'] + wrf_feature_cols]
    wrf_features_df.to_csv(features_path, index=False)
    logger.info(f"已保存: {features_path}")

    logger.info(f"\n=== WRF特征列表 ({len(wrf_feature_cols)}个) ===")
    for col in sorted(wrf_feature_cols):
        vals = pd.to_numeric(merged[col], errors='coerce')
        null_count = vals.isna().sum()
        logger.info(f"  - {col}: min={vals.min():.3f}, max={vals.max():.3f}, mean={vals.mean():.3f}, null={null_count}")

    return merged


def main():
    parser = argparse.ArgumentParser(description="WRF/Open-Meteo 特征工程")
    parser.add_argument("--site-id", type=int, default=DEFAULT_SITE_ID, help="站点 ID（默认 4）")
    args = parser.parse_args()
    site_id = args.site_id

    logger.info("=" * 60)
    logger.info(f"Site {site_id} WRF特征工程启动")
    logger.info("=" * 60)

    wrf_df = load_weather_forecast()
    wrf_df = create_wrf_features(wrf_df)
    site_df = load_site_data(site_id)

    output_dir = PROCESSED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_df = merge_and_save(site_df, wrf_df, output_dir, site_id=site_id)

    logger.info("\n=== WRF特征统计 ===")
    wrf_cols = [col for col in merged_df.columns if col.startswith('wrf_')]
    for col in wrf_cols:
        vals = pd.to_numeric(merged_df[col], errors='coerce')
        logger.info(f"{col}: min={vals.min():.3f}, max={vals.max():.3f}, mean={vals.mean():.3f}, null={vals.isna().sum()}")

    logger.info("\n" + "=" * 60)
    logger.info(f"Site {site_id} WRF特征工程完成")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
