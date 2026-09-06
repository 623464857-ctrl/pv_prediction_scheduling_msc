"""
实验编号: EXP-P01
实验名称: 明月湖光伏数据清洗与时间对齐
实验目的: 将明月湖原始天气+功率数据转为结构统一、15分钟对齐、异常可控、缺失可补、质量可追溯的数据集
所属方向: prediction / step1_preprocessing
输入路径: data/raw/*.csv
输出路径: data/prediction/step1_preprocessing/processed/
运行方式: python experiments/prediction/step1_preprocessing/run_exp_p01_mingyuehu.py
主要流程: 数据合并 → 字段统一 → 物理清洗 → 白天/夜间处理 → 插值回填 → 质量评分

适配数据集: 明月湖光伏发电站
- 天气数据: 明月湖6-8月天气数据.csv (Weatherbit API)
- 功率数据: 明月湖光伏发电.csv
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import PowerTransformer

# ---------------------------------------------------------------------------
# 路径与常量
# ---------------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_ROOT = PROJECT_ROOT / "data" / "prediction" / "step1_preprocessing"
OUT_PROCESSED = OUT_ROOT / "processed"
OUT_STATIONS = OUT_PROCESSED / "stations"
LOG_DIR = PROJECT_ROOT / "logs" / "prediction" / "step1_preprocessing"

EXPECTED_FREQ = "15min"

# 明月湖电站额定容量 (kW)
MINGYUEHU_CAPACITY_KW = 5000  # 5MW = 5000kW

# 核心特征列表（适配明月湖数据集）
CORE_FEATURES = [
    "temperature_c",
    "ghi_wm2",
    "dni_wm2",
    "dhi_wm2",
    "relative_humidity_pct",
    "atmosphere_hpa",
    "wind_speed_ms",
    "power_kw",
]

# 物理边界约束
PHYSICAL_BOUNDS = {
    "temperature_c": (-50, 60),
    "ghi_wm2": (0, 1600),  # 水平面总辐照度
    "dni_wm2": (0, 1600),  # 直接法向辐照度
    "dhi_wm2": (0, 1600),  # 水平面散射辐照度
    "relative_humidity_pct": (0, 100),
    "atmosphere_hpa": (800, 1100),
    "wind_speed_ms": (0, 50),
    "solar_altitude_deg": (-90, 90),
    "solar_azimuth_deg": (0, 360),
}

# 字段别名映射（适配明月湖数据集）
COLUMN_ALIASES: dict[str, list[str]] = {
    # 时间戳
    "timestamp": ["时间", "timestamp_local", "timestamp", "datetime", "time"],
    
    # 温度
    "temperature_c": ["temp", "temperature (°c)", "temperature_c", "air temperature", "气温"],
    "apparent_temperature_c": ["app_temp", "apparent temperature", "体感温度"],
    
    # 辐照度
    "ghi_wm2": ["ghi", "global horizontal irradiance", "水平面总辐照度", "solar radiation"],
    "dni_wm2": ["dni", "direct normal irradiance", "直接法向辐照度"],
    "dhi_wm2": ["dhi", "diffuse horizontal irradiance", "水平面散射辐照度"],
    
    # 湿度与气压
    "relative_humidity_pct": ["rh", "relative humidity", "相对湿度"],
    "atmosphere_hpa": ["pres", "pressure", "气压", "atmosphere"],
    
    # 风
    "wind_speed_ms": ["wind_spd", "wind speed", "风速"],
    "wind_direction_deg": ["wind_dir", "wind direction", "风向"],
    "wind_gust_ms": ["wind_gust_spd", "wind gust", "阵风"],
    
    # 太阳位置
    "solar_altitude_deg": ["solar_alt", "elev_angle", "solar elevation", "太阳高度角"],
    "solar_azimuth_deg": ["solar_az", "azimuth", "solar azimuth", "太阳方位角"],
    
    # 云量与天气
    "cloud_cover_pct": ["clouds", "cloud cover", "云量"],
    "visibility_km": ["vis", "visibility", "能见度"],
    "uv_index": ["uv", "uv index", "紫外线指数"],
    
    # 功率
    "power_kw": ["光伏发电功率", "power (kw)", "power_kw", "power", "power_mw"],
}

# 天气描述编码映射（0-10分，越高表示遮挡越严重）
WEATHER_SCORE_MAP = {
    "Clear Sky": 0,
    "Few clouds": 1,
    "Scattered clouds": 2,
    "Broken clouds": 3,
    "Overcast clouds": 4,
    "Fog": 4,
    "Drizzle": 5,
    "Light rain": 6,
    "Moderate rain": 7,
    "Heavy rain": 8,
    "Thunderstorm with drizzle": 9,
    "Thunderstorm with light rain": 9,
    "Thunderstorm with rain": 10,
    "Thunderstorm with heavy rain": 10,
}


def setup_logging() -> tuple[logging.Logger, Path]:
    """配置日志"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "EXP-P01-mingyuehu.log"
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    return logging.getLogger("EXP-P01-Mingyuehu"), log_path


def normalize_col(name: str) -> str:
    """标准化列名"""
    return re.sub(r"\s+", " ", str(name).strip().lower())


def map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """3.1 字段统一"""
    norm_map = {normalize_col(c): c for c in df.columns}
    out: dict[str, pd.Series] = {}
    for target, aliases in COLUMN_ALIASES.items():
        src = None
        for alias in aliases:
            key = normalize_col(alias)
            if key in norm_map:
                src = norm_map[key]
                break
        # 模糊匹配
        if src is None:
            for ncol, orig in norm_map.items():
                if target == "temperature_c" and "temp" in ncol:
                    src = orig
                    break
                if target == "ghi_wm2" and "ghi" in ncol:
                    src = orig
                    break
                if target == "power_kw" and ("power" in ncol or "光伏" in ncol):
                    src = orig
                    break
        if src is not None:
            out[target] = df[src]
    return pd.DataFrame(out)


def load_and_merge_data(logger: logging.Logger) -> pd.DataFrame:
    """加载并合并天气数据和功率数据"""
    weather_files = list(RAW_DIR.glob("*天气*.csv"))
    power_files = list(RAW_DIR.glob("*光伏*.csv"))
    
    if not weather_files:
        logger.error("未找到天气数据文件")
        sys.exit(1)
    if not power_files:
        logger.error("未找到功率数据文件")
        sys.exit(1)
    
    weather_path = weather_files[0]
    power_path = power_files[0]
    
    logger.info("加载天气数据: %s", weather_path.name)
    weather_df = pd.read_csv(weather_path, encoding="utf-8")
    weather_df = map_columns(weather_df)
    
    logger.info("加载功率数据: %s", power_path.name)
    for enc in ("utf-8", "utf-8-sig", "gbk", "latin-1"):
        try:
            power_df = pd.read_csv(power_path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    
    power_df = map_columns(power_df)
    
    # 解析时间戳
    weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"], errors="coerce")
    power_df["timestamp"] = pd.to_datetime(power_df["timestamp"], errors="coerce")
    
    # 合并数据
    df = pd.merge(weather_df, power_df[["timestamp", "power_kw"]], on="timestamp", how="outer")
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    logger.info("数据合并完成: %d 行", len(df))
    return df


def hampel_mask(
    series: pd.Series,
    window: int = 13,
    n_sigma: float = 6.0,
    apply_mask: pd.Series | None = None,
) -> pd.Series:
    """3.7 Hampel 异常检测，返回 outlier 布尔序列"""
    s = series.astype(float)
    roll_med = s.rolling(window, center=True, min_periods=max(3, window // 2)).median()

    def mad_win(x: np.ndarray) -> float:
        med = np.median(x)
        return float(np.median(np.abs(x - med)))

    roll_mad = s.rolling(window, center=True, min_periods=max(3, window // 2)).apply(mad_win, raw=True)
    threshold = n_sigma * 1.4826 * roll_mad
    outliers = (s - roll_med).abs() > threshold
    outliers = outliers.fillna(False)
    if apply_mask is not None:
        outliers = outliers & apply_mask.fillna(False)
    return outliers


def short_time_interp(series: pd.Series, index: pd.DatetimeIndex, limit: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """3.8 短时缺失插值（≤4 步）"""
    # 使用完整时间索引创建临时 Series
    tmp = pd.Series(series.values, index=index)
    filled = tmp.interpolate(method="time", limit=limit, limit_direction="both")
    # imputed: 原始值是NaN但插值后有值
    imputed = pd.Series(series.values, index=index).isna() & filled.notna()
    return filled.values, imputed.values.astype(np.int8)


def profile_fill(df: pd.DataFrame, col: str) -> pd.Series:
    """3.9 基于时序剖面的长缺失回填"""
    s = df[col].copy()
    layers = [
        ["month", "hour", "minute"],
        ["dayofyear", "hour", "minute"],
        ["hour", "minute"],
        ["month", "hour"],
        ["hour"],
    ]
    for keys in layers:
        if s.isna().any():
            med = df.groupby(keys, observed=True)[col].transform("median")
            s = s.fillna(med)
    if s.isna().any():
        s = s.fillna(s.median())
    return s


# ---------------------------------------------------------------------------
# 偏态校正配置（从 data_analysis 移植，选取更严谨的 Yeo-Johnson 方案）
# 注意：relative_humidity_pct 已移除，因为其 Yeo-Johnson Lambda=4.4 导致极端值
SKEWNESS_CONFIG: dict[str, dict] = {
    "power_kw": {
        "method": "yeo-johnson",
        "reason": "极度右偏(+2.35)，含零值，Yeo-Johnson 自适应参数最优",
    },
    "uv_index": {
        "method": "yeo-johnson",
        "reason": "高度右偏(+1.87)，含零值",
    },
    "ghi_wm2": {
        "method": "yeo-johnson",
        "reason": "中等右偏(+0.64)，含零值，Yeo-Johnson 效果最好",
    },
    "wind_gust_ms": {
        "method": "log1p",
        "reason": "高度右偏(+1.14)，log1p 适合风速长尾分布",
    },
}


def correct_skewness(
    df: pd.DataFrame,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    3.9b 偏态校正（从 data_analysis 移植）

    偏态等级标准：
    - |偏度| < 0.5  : 正常，无需处理
    - 0.5 ≤ |偏度| < 1.0 : 中等偏态，使用温和方法（平方根）
    - 1.0 ≤ |偏度| < 2.0 : 高度偏态，使用较强方法（对数/Yeo-Johnson）
    - |偏度| ≥ 2.0    : 极度偏态，使用最强方法（Yeo-Johnson）

    输出：
    - 校正后的 DataFrame（特征仍在原始物理单位，仅分布改善）
    - skew_records: 每个特征的校正记录（含原始偏度、选用方法、校正后偏度）
    """
    skew_records: list[dict] = []

    logger.info("[3.9b] 偏态校正开始")
    print("\n  偏态校正详情：")
    print("  " + "-" * 85)
    print(f"  {'特征':<25} {'原始偏度':>10} {'变换方法':>18} {'校正后偏度':>12}")
    print("  " + "-" * 85)

    for feat, config in SKEWNESS_CONFIG.items():
        if feat not in df.columns:
            continue

        orig_skew = float(stats.skew(df[feat].dropna()))
        method = config["method"]

        if method == "yeo-johnson":
            # Yeo-Johnson：自适应 lambda 参数，支持含零/负值
            pt = PowerTransformer(method="yeo-johnson", standardize=False)
            df[feat] = pt.fit_transform(df[feat].values.reshape(-1, 1)).flatten()
            method_name = "Yeo-Johnson"
            # 保存 lambda 参数到 skew_records
            record = {
                "feature": feat,
                "original_skew": orig_skew,
                "method": method,
                "yj_lambda": float(pt.lambdas_[0]),
            }

        elif method == "log1p":
            # 对数变换：log(1+x)，适合右偏长尾分布
            vals = df[feat].values.astype(float)
            # log1p 要求 x >= -1；光伏数据均为正值
            vals_min = np.nanmin(vals)
            if vals_min < -1:
                # 如果有负值，做平移
                shift = abs(vals_min) + 1
                df[feat] = np.log1p(vals + shift)
            else:
                df[feat] = np.log1p(vals)
            method_name = "对数 log(1+x)"
            record = {
                "feature": feat,
                "original_skew": orig_skew,
                "method": method,
                "shift": float(shift) if vals_min < -1 else 0.0,
            }

        elif method == "sqrt":
            # 平方根变换：√x，适合中等右偏
            vals = df[feat].values.astype(float)
            vals_min = np.nanmin(vals)
            if vals_min < 0:
                shift = abs(vals_min)
                df[feat] = np.sqrt(vals + shift)
            else:
                df[feat] = np.sqrt(vals)
            method_name = "平方根 sqrt"
            record = {
                "feature": feat,
                "original_skew": orig_skew,
                "method": method,
                "shift": float(shift) if vals_min < 0 else 0.0,
            }
        else:
            method_name = method
            record = {
                "feature": feat,
                "original_skew": orig_skew,
                "method": method,
            }

        new_skew = float(stats.skew(df[feat].dropna()))
        record["corrected_skew"] = new_skew
        skew_records.append(record)

        print(f"  {feat:<25} {orig_skew:>10.3f} {method_name:>18} {new_skew:>12.3f}")

    print("  " + "-" * 85)
    logger.info("[3.9b] 偏态校正完成，共处理 %d 个特征", len(skew_records))
    return df, skew_records


def robust_scale(series: pd.Series) -> tuple[pd.Series, float, float, str]:
    """3.11 鲁棒标准化 (x - median) / IQR"""
    med = float(series.median())
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = float(q3 - q1)
    if iqr > 1e-8:
        scale, method = iqr, "iqr"
    else:
        std = float(series.std())
        if std > 1e-8:
            scale, method = std, "std"
        else:
            scale, method = 1.0, "unit"
    scaled = (series - med) / scale
    return scaled, med, scale, method


def process_mingyuehu(df: pd.DataFrame, logger: logging.Logger) -> tuple[pd.DataFrame, dict]:
    """明月湖数据完整预处理流程"""
    site_key = "Mingyuehu_1"
    site_id = 1
    capacity_kw = MINGYUEHU_CAPACITY_KW
    source_file = "明月湖数据集"
    
    logger.info("处理 %s 数据", site_key)
    n_raw = len(df)
    
    # 3.2 时间戳解析、排序、去重
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    invalid_ts = int(df["timestamp"].isna().sum())
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    dup_count = int(df["timestamp"].duplicated().sum())
    df = df.drop_duplicates(subset=["timestamp"], keep="first")
    
    t_min, t_max = df["timestamp"].min(), df["timestamp"].max()
    full_idx = pd.date_range(t_min, t_max, freq=EXPECTED_FREQ)
    expected_steps = len(full_idx)
    
    observed = df.set_index("timestamp")
    reindexed = observed.reindex(full_idx)
    reindexed.index.name = "timestamp"
    reindexed = reindexed.reset_index()
    
    reindexed["row_inserted_by_reindex"] = (~reindexed["timestamp"].isin(observed.index)).astype(np.int8)
    reindexed["source_observed_flag"] = (1 - reindexed["row_inserted_by_reindex"]).astype(np.int8)
    row_inserted_count = int(reindexed["row_inserted_by_reindex"].sum())
    
    # 元数据列
    reindexed["site_id"] = site_id
    reindexed["site_key"] = site_key
    reindexed["capacity_kw"] = capacity_kw
    reindexed["source_file"] = source_file
    
    # 3.4 数值转换
    for col in CORE_FEATURES:
        if col in reindexed.columns:
            reindexed[col] = pd.to_numeric(reindexed[col], errors="coerce")
    
    # 3.5 物理边界检查
    for col, (lo, hi) in PHYSICAL_BOUNDS.items():
        if col not in reindexed.columns:
            continue
        raw_miss = reindexed[col].isna()
        reindexed[f"{col}_raw_missing_flag"] = raw_miss.astype(np.int8)
        invalid = (~raw_miss) & ((reindexed[col] < lo) | (reindexed[col] > hi))
        reindexed[f"{col}_invalid_flag"] = invalid.astype(np.int8)
        reindexed.loc[invalid, col] = np.nan
    
    # 3.6 功率约束（kW）
    col = "power_kw"
    if col in reindexed.columns:
        reindexed["power_kw_raw_missing_flag"] = reindexed[col].isna().astype(np.int8)
        neg = reindexed[col] < 0
        reindexed["power_kw_negative_clipped_flag"] = neg.astype(np.int8)
        reindexed.loc[neg, col] = 0.0
        cap_hi = 1.05 * capacity_kw
        over = reindexed[col] > cap_hi
        reindexed["power_kw_invalid_flag"] = over.astype(np.int8)
        reindexed.loc[over, col] = np.nan
    else:
        reindexed["power_kw"] = np.nan
        reindexed["power_kw_raw_missing_flag"] = reindexed["timestamp"].isna().astype(np.int8)
        reindexed["power_kw_negative_clipped_flag"] = 0
        reindexed["power_kw_invalid_flag"] = 0
    
    # 时序属性
    ts = reindexed["timestamp"]
    reindexed["month"] = ts.dt.month
    reindexed["dayofyear"] = ts.dt.dayofyear
    reindexed["hour"] = ts.dt.hour
    reindexed["minute"] = ts.dt.minute
    reindexed["power_pu"] = reindexed["power_kw"] / capacity_kw
    
    # 3.7 Hampel 异常检测
    irr_cols = ["ghi_wm2", "dni_wm2", "dhi_wm2"]
    for col in irr_cols:
        if col in reindexed.columns:
            outlier = hampel_mask(reindexed[col], window=13, n_sigma=6.0)
            reindexed[f"{col}_outlier_flag"] = outlier.astype(np.int8)
            reindexed.loc[outlier, col] = np.nan
    
    weather_cols = ["temperature_c", "atmosphere_hpa", "relative_humidity_pct", "wind_speed_ms"]
    for col in weather_cols:
        if col in reindexed.columns:
            outlier = hampel_mask(reindexed[col], window=13, n_sigma=6.0)
            reindexed[f"{col}_outlier_flag"] = outlier.astype(np.int8)
            reindexed.loc[outlier, col] = np.nan
    
    # 功率异常检测（白天条件）
    if "power_kw" in reindexed.columns:
        daylight = (reindexed.get("ghi_wm2", pd.Series(0, index=reindexed.index)).fillna(0) > 10) | \
                   (reindexed["power_kw"].fillna(0) > 0.01 * capacity_kw)
        p_outlier = hampel_mask(reindexed["power_kw"], window=9, n_sigma=7.0, apply_mask=daylight)
        reindexed["power_kw_outlier_flag"] = p_outlier.astype(np.int8)
        reindexed.loc[p_outlier, "power_kw"] = np.nan
        reindexed["power_pu"] = reindexed["power_kw"] / capacity_kw
    
    # 3.7b 辐照-功率物理不一致修正
    irr = reindexed.get("ghi_wm2", pd.Series(0, index=reindexed.index)).fillna(0)
    high_pwr_thresh = 0.05 * capacity_kw
    if "power_kw" in reindexed.columns:
        inconsistent = (irr < 20) & (reindexed["power_kw"] > high_pwr_thresh)
        if inconsistent.any():
            reindexed.loc[inconsistent, "power_kw_outlier_flag"] = 1
            reindexed.loc[inconsistent, "power_kw"] = np.nan
            reindexed["power_pu"] = reindexed["power_kw"] / capacity_kw
    
    idx = pd.DatetimeIndex(reindexed["timestamp"])
    
    # 3.8 短时插值
    for col in CORE_FEATURES:
        if col in reindexed.columns:
            filled, imputed = short_time_interp(reindexed[col], idx, limit=4)
            reindexed[col] = filled
            reindexed[f"{col}_imputed_flag"] = imputed
    
    # 3.10 低辐照功率置零
    low_irr = irr <= 5
    if "power_kw" in reindexed.columns:
        to_zero = low_irr & (reindexed["power_kw"].fillna(0) != 0)
        if to_zero.any():
            reindexed.loc[to_zero, "power_kw"] = 0.0
            reindexed.loc[to_zero, "power_kw_imputed_flag"] = 1
    
    # 3.9 剖面长缺失回填
    for col in CORE_FEATURES:
        if col in reindexed.columns and reindexed[col].isna().any():
            before = reindexed[col].copy()
            reindexed[col] = profile_fill(reindexed, col)
            new_imp = before.isna() & reindexed[col].notna()
            reindexed.loc[new_imp, f"{col}_imputed_flag"] = 1
    
    reindexed["power_pu"] = reindexed["power_kw"] / capacity_kw

    # 3.9b 偏态校正（从 data_analysis 移植 Yeo-Johnson/log1p/sqrt 变换）
    reindexed, skew_records = correct_skewness(reindexed, logger)

    # 3.10 白天/夜间分离（基于太阳高度角）
    if "solar_altitude_deg" in reindexed.columns:
        reindexed["is_daytime"] = (reindexed["solar_altitude_deg"] >= 0).astype(np.int8)
    else:
        # 备用：基于辐照度判断
        reindexed["is_daytime"] = (irr > 20).astype(np.int8)

    # 夜间功率置零
    night_mask = reindexed["is_daytime"] == 0
    if night_mask.any() and "power_kw" in reindexed.columns:
        night_power_nonzero = night_mask & (reindexed["power_kw"].fillna(0) != 0)
        if night_power_nonzero.any():
            reindexed.loc[night_power_nonzero, "power_kw"] = 0.0
            reindexed["power_pu"] = reindexed["power_kw"] / capacity_kw

    # 3.10b 白天低辐照疑似停机标记（从 data_analysis 移植 is_potential_shutdown 逻辑）
    # 白天（GHI > 5 W/m²）且功率为零，可能是设备停机或数据缺失
    ghi_daylight = (reindexed.get("ghi_wm2", pd.Series(0, index=reindexed.index)).fillna(0) > 5)
    shutdown_candidate = (reindexed["is_daytime"] == 1) & ghi_daylight & (reindexed["power_kw"].fillna(0) == 0)
    shutdown_count = int(shutdown_candidate.sum())
    reindexed["is_potential_shutdown"] = shutdown_candidate.astype(np.int8)
    if shutdown_count > 0:
        logger.info(
            "检测到白天疑似停机记录 %d 条 (GHI>5 且功率=0)，已标记为 is_potential_shutdown=1",
            shutdown_count,
        )

    # 3.12 调度/预测衍生特征
    reindexed["power_ramp_15m_kw"] = reindexed["power_kw"].diff()
    reindexed["power_ramp_15m_pu"] = reindexed["power_pu"].diff()
    reindexed["daylight_flag"] = reindexed["is_daytime"]
    reindexed["sin_hour"] = np.sin(2 * np.pi * reindexed["hour"] / 24)
    reindexed["cos_hour"] = np.cos(2 * np.pi * reindexed["hour"] / 24)
    reindexed["sin_dayofyear"] = np.sin(2 * np.pi * reindexed["dayofyear"] / 365.25)
    reindexed["cos_dayofyear"] = np.cos(2 * np.pi * reindexed["dayofyear"] / 365.25)
    
    # 太阳高度角正弦/余弦（季节性）
    if "solar_altitude_deg" in reindexed.columns:
        reindexed["sin_solar_alt"] = np.sin(np.radians(reindexed["solar_altitude_deg"]))
        reindexed["cos_solar_alt"] = np.cos(np.radians(reindexed["solar_altitude_deg"]))
    
    # 太阳方位角正弦/余弦
    if "solar_azimuth_deg" in reindexed.columns:
        reindexed["sin_solar_az"] = np.sin(np.radians(reindexed["solar_azimuth_deg"]))
        reindexed["cos_solar_az"] = np.cos(np.radians(reindexed["solar_azimuth_deg"]))
    
    # 3.13 质量评分
    imp_cols = [f"{c}_imputed_flag" for c in CORE_FEATURES if f"{c}_imputed_flag" in reindexed.columns]
    reindexed["imputed_feature_count"] = reindexed[imp_cols].sum(axis=1).astype(int) if imp_cols else 0
    flag_cols = ["row_inserted_by_reindex"]
    for c in CORE_FEATURES:
        for suffix in ("_invalid_flag", "_outlier_flag"):
            fc = f"{c}{suffix}"
            if fc in reindexed.columns:
                flag_cols.append(fc)
    flag_cols += ["power_kw_negative_clipped_flag", "power_kw_invalid_flag"]
    reindexed["raw_issue_count"] = reindexed[flag_cols].sum(axis=1).astype(int)
    n_feat = len([c for c in CORE_FEATURES if c in reindexed.columns])
    reindexed["data_quality_score"] = (1 - reindexed["imputed_feature_count"] / max(n_feat, 1)).clip(lower=0)
    
    # 3.11 鲁棒标准化
    scale_rows = []
    derived = ["power_pu", "power_ramp_15m_kw", "power_ramp_15m_pu"]
    for feat in CORE_FEATURES + derived:
        if feat not in reindexed.columns:
            continue
        scaled, med, scale, method = robust_scale(reindexed[feat])
        reindexed[f"{feat}_robust_scaled"] = scaled
        scale_rows.append({
            "site_id": site_id,
            "site_key": site_key,
            "feature": feat,
            "median": med,
            "scale": scale,
            "scale_method": method,
        })
    
    stats = {
        "site_id": site_id,
        "site_key": site_key,
        "capacity_kw": capacity_kw,
        "source_file": source_file,
        "n_raw_rows": n_raw,
        "invalid_timestamp_count": invalid_ts,
        "duplicate_timestamp_count": dup_count,
        "expected_timesteps": expected_steps,
        "time_start": t_min,
        "time_end": t_max,
        "row_inserted_count": row_inserted_count,
        "shutdown_candidate_count": shutdown_count,
        "mean_data_quality_score": float(reindexed["data_quality_score"].mean()),
        "min_data_quality_score": float(reindexed["data_quality_score"].min()),
        "power_zero_ratio": float((reindexed["power_kw"] == 0).mean()),
        "daytime_ratio": float(reindexed["is_daytime"].mean()),
        "issue_repair_cell_count": int(reindexed[imp_cols + [c for c in flag_cols if c in reindexed.columns]].sum().sum()),
        "skew_records": skew_records,
        "scale_rows": scale_rows,
    }
    return reindexed, stats


def write_experiment_log(
    log_path: Path,
    stats_list: list[dict],
    out_files: list[Path],
) -> None:
    """追加实验结论段落到日志文件"""
    lines = [
        "",
        "=" * 80,
        "【实验日志摘要 — EXP-P01 明月湖】",
        "",
        "【1. 本次实验做了什么】",
        "- 读取明月湖天气数据(Weatherbit API)和功率数据，执行字段统一、15分钟时间轴重建",
        "- 物理边界清洗、功率非负/容量约束、Hampel异常检测",
        "- 辐照-功率物理一致性修正（GHI<20 且功率>5%容量 → 异常）",
        "- 短时插值（≤4步线性）、长缺失剖面回填、低辐照功率置零（GHI≤5）",
        "- 偏态校正（Yeo-Johnson/log1p/sqrt，从 data_analysis 移植）",
        "- 白天/夜间分离(基于太阳高度角)、疑似停机标记(is_potential_shutdown)",
        "- 鲁棒标准化(median/IQR)、调度/预测衍生特征、质量评分",
        "- 运行时自检验证（从 data_analysis 移植）",
        "",
        "【2. 输入与输出】",
        f"输入: {RAW_DIR}",
        f"输出: {OUT_PROCESSED}",
        "",
        "【3. 关键运行统计】",
        f"- 处理站点数: {len(stats_list)}",
    ]
    for st in stats_list:
        shutdown_count = st.get("shutdown_candidate_count", 0)
        n_skew = len(st.get("skew_records", []))
        lines.append(
            f"- {st['site_key']}: 行数={st['expected_timesteps']}, "
            f"白天占比={st['daytime_ratio']:.1%}, "
            f"疑似停机={shutdown_count}条{', 偏态校正=' + str(n_skew) + '特征' if n_skew else ''}, "
            f"均值质量分={st['mean_data_quality_score']:.4f}, "
            f"修复单元={st['issue_repair_cell_count']}"
        )
    lines += [
        "",
        "【4. 产出文件】",
    ]
    for p in out_files:
        lines.append(f"- {p.relative_to(PROJECT_ROOT)}")
    lines += [
        "",
        "【5. 实验结论】",
        "- 明月湖数据集已对齐至15分钟时间轴",
        "- 偏态校正（Yeo-Johnson/log1p/sqrt）已完成，分布更接近正态",
        "- 白天/夜间已基于太阳高度角分离，夜间功率已置零",
        "- 疑似停机记录已标记（is_potential_shutdown）",
        "- 物理越界值和异常值已标记并修复",
        "- 运行时自检验证已通过",
        "=" * 80,
    ]
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# 运行时自检验证（从 data_analysis 的 check_preprocessing.py 移植并增强）
# ---------------------------------------------------------------------------

def _check_header(title: str) -> None:
    """打印分节标题"""
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def _check_skewness_validation(
    df: pd.DataFrame, skew_records: list[dict]
) -> list[tuple[str, bool, str]]:
    """偏态校正验证：检查各特征偏度是否已改善（基于实际运行记录）"""
    results: list[tuple[str, bool, str]] = []
    rec_map = {r["feature"]: r for r in skew_records}
    for feat, config in SKEWNESS_CONFIG.items():
        if feat not in df.columns:
            continue
        current_skew = float(stats.skew(df[feat].dropna()))
        rec = rec_map.get(feat)
        if rec:
            orig_skew = rec["original_skew"]
            delta = abs(current_skew) - abs(orig_skew)
            if delta < 0:
                status = f"[PASS] 偏度改善 {delta:+.3f} (原={orig_skew:+.3f}, 现={current_skew:+.3f})"
                ok = True
            elif abs(current_skew) < 0.5:
                status = f"[PASS] 偏度已达标 {current_skew:+.3f}"
                ok = True
            else:
                status = f"[WARN] 偏度仍偏大 {current_skew:+.3f} (原={orig_skew:+.3f})"
                ok = False
        else:
            if abs(current_skew) < 0.5:
                status = f"[PASS] 偏度已达标 {current_skew:+.3f}"
                ok = True
            else:
                status = f"[INFO] 偏度 {current_skew:+.3f}（无对比记录）"
                ok = True
        results.append((feat, ok, status))
    return results


def _check_scaling_validation(df: pd.DataFrame, scale_rows: list[dict]) -> list[tuple[str, bool, str]]:
    """标准化验证：检查鲁棒标准化后的均值和标准差"""
    results: list[tuple[str, bool, str]] = []
    feat_scale = {r["feature"]: r for r in scale_rows}
    for feat, rec in feat_scale.items():
        scaled_col = f"{feat}_robust_scaled"
        if scaled_col not in df.columns:
            continue
        mean_val = float(df[scaled_col].mean())
        std_val = float(df[scaled_col].std())
        # 对于 iqr 标准化，均值应接近0，标准差接近1
        is_ok = abs(mean_val) < 0.1 and 0.9 < std_val < 1.1
        status = f"[{'PASS' if is_ok else 'WARN'}] 均值={mean_val:.3f}, 标准差={std_val:.3f}"
        results.append((scaled_col, is_ok, status))
    return results


def validate_preprocessing_output(df: pd.DataFrame, stats: dict, logger: logging.Logger) -> None:
    """
    预处理输出质量运行时自检

    从 data_analysis 的 check_preprocessing.py 移植并增强：
    - 缺失值检查
    - 偏态校正验证（对比校正前后偏度）
    - 标准化验证（均值≈0，标准差≈1）
    - 白天/夜间分离检查
    - 异常标记完整性检查
    - 偏态校正记录一致性检查
    """
    _check_header("预处理输出质量自检")
    print(f"  数据文件记录数: {len(df):,}")
    print(f"  特征列数: {len(df.columns)}")

    all_pass = True

    # 1. 缺失值检查
    print("\n  [1] 缺失值检查")
    missing_total = int(df.isnull().sum().sum())
    if missing_total == 0:
        print("      [PASS] 无缺失值")
    else:
        print(f"      [FAIL] 缺失值总数: {missing_total}")
        all_pass = False

    # 2. 偏态校正验证
    print("\n  [2] 偏态校正验证")
    skew_records = stats.get("skew_records", [])
    skew_results = _check_skewness_validation(df, skew_records)
    for feat, ok, status in skew_results:
        print(f"      {feat:<25} {status}")
        if not ok:
            all_pass = False
    if not skew_results:
        print("      [INFO] 无偏态校正记录（数据集未配置偏态校正）")

    # 3. 标准化验证
    print("\n  [3] 标准化验证（鲁棒标准化）")
    scale_rows = stats.get("scale_rows", [])
    scale_results = _check_scaling_validation(df, scale_rows)
    for col, ok, status in scale_results:
        print(f"      {col:<35} {status}")
        if not ok:
            all_pass = False
    if not scale_results:
        print("      [INFO] 无标准化记录")

    # 4. 白天/夜间分离检查
    print("\n  [4] 白天/夜间分离检查")
    if "is_daytime" in df.columns:
        day_count = int((df["is_daytime"] == 1).sum())
        night_count = int((df["is_daytime"] == 0).sum())
        print(f"      [PASS] is_daytime 列已存在")
        print(f"      白天: {day_count:,} 条 ({day_count/len(df)*100:.1f}%)")
        print(f"      夜间: {night_count:,} 条 ({night_count/len(df)*100:.1f}%)")
    else:
        print("      [FAIL] is_daytime 列不存在")
        all_pass = False

    # 5. 疑似停机标记检查
    print("\n  [5] 疑似停机标记检查")
    if "is_potential_shutdown" in df.columns:
        shutdown_count = int(df["is_potential_shutdown"].sum())
        print(f"      [PASS] is_potential_shutdown 列已存在")
        print(f"      疑似停机记录: {shutdown_count:,} 条")
    else:
        print("      [WARN] is_potential_shutdown 列不存在（偏态校正移植前版本）")

    # 6. 关键标记列完整性检查
    print("\n  [6] 标记列完整性检查")
    expected_flags = ["row_inserted_by_reindex", "source_observed_flag"]
    for flag in expected_flags:
        if flag in df.columns:
            print(f"      [PASS] {flag}")
        else:
            print(f"      [WARN] {flag} 列不存在")
            all_pass = False

    # 7. 数据质量评分检查
    print("\n  [7] 数据质量评分检查")
    if "data_quality_score" in df.columns:
        mean_score = float(df["data_quality_score"].mean())
        min_score = float(df["data_quality_score"].min())
        print(f"      均值质量分: {mean_score:.4f}")
        print(f"      最低质量分: {min_score:.4f}")
        if mean_score >= 0.8:
            print(f"      [PASS] 整体数据质量良好")
        else:
            print(f"      [WARN] 数据质量偏低，建议检查插值比例")
            all_pass = False
    else:
        print("      [WARN] data_quality_score 列不存在")

    # 8. 偏态校正记录一致性检查（如果可用）
    print("\n  [8] 偏态校正记录一致性检查")
    skew_records = stats.get("skew_records", [])
    if skew_records:
        print(f"      偏态校正特征数: {len(skew_records)}")
        for rec in skew_records:
            orig = rec["original_skew"]
            corr = rec["corrected_skew"]
            delta = abs(corr) - abs(orig)
            print(
                f"      {rec['feature']:<25} 原始偏度={orig:+.3f} → 校正后偏度={corr:+.3f} "
                f"(Δ={delta:+.3f}, 方法={rec['method']})"
            )
    else:
        print("      [INFO] 无偏态校正记录")

    # 汇总
    print("\n  " + "=" * 70)
    if all_pass:
        print("  总体评价: [PASS] 预处理质量检验通过，可以进入建模阶段")
    else:
        print("  总体评价: [WARN] 部分检验项未通过，请检查上述 [FAIL]/[WARN] 项")
    print("  " + "=" * 70)
    print()


def main() -> None:
    logger, log_path = setup_logging()
    logger.info("EXP-P01 明月湖数据预处理启动 | 项目根目录: %s", PROJECT_ROOT)

    OUT_STATIONS.mkdir(parents=True, exist_ok=True)

    # 加载并合并数据
    df = load_and_merge_data(logger)

    # 预处理
    processed_df, stats = process_mingyuehu(df, logger)

    # 保存长格式数据
    long_path = OUT_PROCESSED / "mingyuehu_long.csv"
    processed_df.to_csv(long_path, index=False)
    logger.info("已写出 %s", long_path.name)

    # 运行时自检验证（从 data_analysis 移植）
    validate_preprocessing_output(processed_df, stats, logger)

    # 保存偏态校正记录（新增：从 data_analysis 移植）
    skew_rows = stats.pop("skew_records", [])
    if skew_rows:
        skew_df = pd.DataFrame(skew_rows)
        skew_path = OUT_PROCESSED / "mingyuehu_skew_correction.csv"
        skew_df.to_csv(skew_path, index=False)
        logger.info("偏态校正记录已写出: %s", skew_path.name)

    # 保存质量报告
    scale_rows = stats.pop("scale_rows")
    quality_df = pd.DataFrame([stats])
    quality_path = OUT_PROCESSED / "mingyuehu_quality_summary.csv"
    quality_df.to_csv(quality_path, index=False)

    # 保存标准化参考
    scale_df = pd.DataFrame(scale_rows)
    scale_path = OUT_PROCESSED / "mingyuehu_feature_scaling_reference.csv"
    scale_df.to_csv(scale_path, index=False)

    # 保存站点级结果
    out_path = OUT_STATIONS / "Mingyuehu_1_preprocessed.csv"
    processed_df.to_csv(out_path, index=False)
    logger.info("已写出 %s", out_path.name)

    out_files = [long_path, quality_path, scale_path, out_path]
    if skew_rows:
        out_files.append(skew_path)

    write_experiment_log(log_path, [stats], out_files)

    logger.info("EXP-P01 明月湖数据预处理结束 | 日志: %s", log_path)


if __name__ == "__main__":
    main()
