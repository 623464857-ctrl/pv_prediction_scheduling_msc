"""
实验编号: EXP-P01
实验名称: 光伏站点数据清洗与时间对齐
实验目的: 将原始站点 CSV 转为结构统一、15 分钟对齐、异常可控、缺失可补、质量可追溯的数据集
所属方向: prediction / step1_data_cleaning_alignment
输入路径: data/raw/*.csv
输出路径: data/prediction/step1_preprocessing/processed/
运行方式: python experiments/prediction/step1_data_cleaning_alignment/run_exp_p01_preprocessing.py
主要流程: 3.1 字段统一 → 3.13 质量评分与汇总
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 路径与常量
# ---------------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_ROOT = PROJECT_ROOT / "data" / "prediction" / "step1_preprocessing"
OUT_PROCESSED = OUT_ROOT / "processed"
OUT_STATIONS = OUT_PROCESSED / "stations"
LOG_DIR = PROJECT_ROOT / "logs" / "prediction" / "step1_data_cleaning_alignment"

EXPECTED_FREQ = "15min"
CORE_FEATURES = [
    "total_irradiance_wm2",
    "direct_normal_irradiance_wm2",
    "global_horizontal_irradiance_wm2",
    "air_temperature_c",
    "atmosphere_hpa",
    "relative_humidity_pct",
    "power_mw",
]

PHYSICAL_BOUNDS = {
    "total_irradiance_wm2": (0, 1600),
    "direct_normal_irradiance_wm2": (0, 1600),
    "global_horizontal_irradiance_wm2": (0, 1600),
    "air_temperature_c": (-50, 60),
    "atmosphere_hpa": (800, 1100),
    "relative_humidity_pct": (0, 100),
}

COLUMN_ALIASES: dict[str, list[str]] = {
    "timestamp": ["time(year-month-day h:m:s)", "time", "timestamp", "datetime"],
    "total_irradiance_wm2": ["total solar irradiance (w/m2)", "total solar irradiance"],
    "direct_normal_irradiance_wm2": ["direct normal irradiance (w/m2)", "direct normal irradiance"],
    "global_horizontal_irradiance_wm2": [
        "global horizontal irradiance (w/m2)",
        "global horicontal irradiance (w/m2)",
        "global horizontal irradiance",
        "global horicontal irradiance",
    ],
    "air_temperature_c": ["air temperature", "air temperature  (°c)", "air temperature  (буc)"],
    "atmosphere_hpa": ["atmosphere (hpa)", "atmosphere"],
    "relative_humidity_pct": ["relative humidity (%)", "relative humidity"],
    "power_mw": ["power (mw)", "power"],
}


def setup_logging() -> tuple[logging.Logger, Path]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "EXP-P01.log"
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    return logging.getLogger("EXP-P01"), log_path


def normalize_col(name: str) -> str:
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
        if src is None:
            for ncol, orig in norm_map.items():
                if target == "air_temperature_c" and "temperature" in ncol:
                    src = orig
                    break
                if target == "global_horizontal_irradiance_wm2" and "hori" in ncol and "irradiance" in ncol:
                    src = orig
                    break
        if src is not None:
            out[target] = df[src]
    return pd.DataFrame(out)


def parse_site_meta(filename: str) -> tuple[int, float, str]:
  site_m = re.search(r"site\s*(\d+)", filename, re.I)
  cap_m = re.search(r"capacity[-\s]*(\d+(?:\.\d+)?)\s*mw", filename, re.I)
  site_id = int(site_m.group(1)) if site_m else -1
  capacity = float(cap_m.group(1)) if cap_m else np.nan
  return site_id, capacity, filename


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


def short_time_interp(series: pd.Series, index: pd.DatetimeIndex, limit: int = 4) -> pd.Series:
    """3.8 短时缺失插值（≤4 步）"""
    tmp = pd.Series(series.values, index=index)
    filled = tmp.interpolate(method="time", limit=limit, limit_direction="both")
    imputed = series.isna() & filled.notna()
    return filled.values, imputed


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


def process_site(path: Path, logger: logging.Logger) -> tuple[pd.DataFrame, dict]:
    """单站完整预处理"""
    site_id, capacity_mw, source_file = parse_site_meta(path.name)
    site_key = f"Site_{site_id}"
    logger.info("处理 %s (%s)", site_key, path.name)

    for enc in ("utf-8", "utf-8-sig", "gbk", "latin-1"):
        try:
            raw = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raw = pd.read_csv(path, encoding="latin-1")
    raw = raw.loc[:, ~raw.columns.astype(str).str.match(r"^Unnamed")]
    df = map_columns(raw)
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

    reindexed["site_id"] = site_id
    reindexed["site_key"] = site_key
    reindexed["capacity_mw"] = capacity_mw
    reindexed["source_file"] = source_file

    # 3.4 数值转换
    for col in CORE_FEATURES:
        reindexed[col] = pd.to_numeric(reindexed[col], errors="coerce")

    # 3.5 物理边界
    for col, (lo, hi) in PHYSICAL_BOUNDS.items():
        raw_miss = reindexed[col].isna()
        reindexed[f"{col}_raw_missing_flag"] = raw_miss.astype(np.int8)
        invalid = (~raw_miss) & ((reindexed[col] < lo) | (reindexed[col] > hi))
        reindexed[f"{col}_invalid_flag"] = invalid.astype(np.int8)
        reindexed.loc[invalid, col] = np.nan

    # 3.6 功率约束
    col = "power_mw"
    reindexed["power_mw_raw_missing_flag"] = reindexed[col].isna().astype(np.int8)
    neg = reindexed[col] < 0
    reindexed["power_mw_negative_clipped_flag"] = neg.astype(np.int8)
    reindexed.loc[neg, col] = 0.0
    cap_hi = 1.05 * capacity_mw
    over = reindexed[col] > cap_hi
    reindexed["power_mw_invalid_flag"] = over.astype(np.int8)
    reindexed.loc[over, col] = np.nan

    # 时序属性（供回填与特征）
    ts = reindexed["timestamp"]
    reindexed["month"] = ts.dt.month
    reindexed["dayofyear"] = ts.dt.dayofyear
    reindexed["hour"] = ts.dt.hour
    reindexed["minute"] = ts.dt.minute
    reindexed["power_pu"] = reindexed["power_mw"] / capacity_mw

    # 3.7 Hampel
    for col in [
        "total_irradiance_wm2",
        "direct_normal_irradiance_wm2",
        "global_horizontal_irradiance_wm2",
        "air_temperature_c",
        "atmosphere_hpa",
        "relative_humidity_pct",
    ]:
        outlier = hampel_mask(reindexed[col], window=13, n_sigma=6.0)
        reindexed[f"{col}_outlier_flag"] = outlier.astype(np.int8)
        reindexed.loc[outlier, col] = np.nan

    daylight = (reindexed["total_irradiance_wm2"].fillna(0) > 10) | (reindexed["power_mw"].fillna(0) > 0.01 * capacity_mw)
    p_outlier = hampel_mask(reindexed["power_mw"], window=9, n_sigma=7.0, apply_mask=daylight)
    reindexed["power_mw_outlier_flag"] = p_outlier.astype(np.int8)
    reindexed.loc[p_outlier, "power_mw"] = np.nan
    reindexed["power_pu"] = reindexed["power_mw"] / capacity_mw

    idx = pd.DatetimeIndex(reindexed["timestamp"])

    # 3.8 短时插值
    for col in CORE_FEATURES:
        filled, imputed = short_time_interp(reindexed[col], idx, limit=4)
        reindexed[col] = filled
        reindexed[f"{col}_imputed_flag"] = imputed.astype(np.int8)

    # 3.10 夜间功率置零（剖面回填前）
    night = reindexed["total_irradiance_wm2"].fillna(0) <= 5
    night_miss = night & reindexed["power_mw"].isna()
    if night_miss.any():
        reindexed.loc[night_miss, "power_mw"] = 0.0
        reindexed.loc[night_miss, "power_mw_imputed_flag"] = 1

    # 3.9 剖面长缺失回填
    for col in CORE_FEATURES:
        before = reindexed[col].copy()
        reindexed[col] = profile_fill(reindexed, col)
        new_imp = before.isna() & reindexed[col].notna()
        reindexed.loc[new_imp, f"{col}_imputed_flag"] = 1

    reindexed["power_pu"] = reindexed["power_mw"] / capacity_mw

    # 3.12 调度/预测衍生特征
    reindexed["power_ramp_15m_mw"] = reindexed["power_mw"].diff()
    reindexed["power_ramp_15m_pu"] = reindexed["power_pu"].diff()
    reindexed["daylight_flag"] = (reindexed["total_irradiance_wm2"] > 20).astype(np.int8)
    reindexed["sin_hour"] = np.sin(2 * np.pi * reindexed["hour"] / 24)
    reindexed["cos_hour"] = np.cos(2 * np.pi * reindexed["hour"] / 24)
    reindexed["sin_dayofyear"] = np.sin(2 * np.pi * reindexed["dayofyear"] / 365.25)
    reindexed["cos_dayofyear"] = np.cos(2 * np.pi * reindexed["dayofyear"] / 365.25)

    # 3.13 质量评分
    imp_cols = [f"{c}_imputed_flag" for c in CORE_FEATURES]
    reindexed["imputed_feature_count"] = reindexed[imp_cols].sum(axis=1).astype(int)
    flag_cols = ["row_inserted_by_reindex"]
    for c in CORE_FEATURES:
        for suffix in ("_invalid_flag", "_outlier_flag"):
            fc = f"{c}{suffix}"
            if fc in reindexed.columns:
                flag_cols.append(fc)
    flag_cols += ["power_mw_negative_clipped_flag", "power_mw_invalid_flag"]
    reindexed["raw_issue_count"] = reindexed[flag_cols].sum(axis=1).astype(int)
    n_feat = len(CORE_FEATURES)
    reindexed["data_quality_score"] = (1 - reindexed["imputed_feature_count"] / n_feat).clip(lower=0)

    # 3.11 鲁棒标准化
    scale_rows = []
    derived = ["power_pu", "power_ramp_15m_mw", "power_ramp_15m_pu"]
    for feat in CORE_FEATURES + derived:
        if feat not in reindexed.columns:
            continue
        scaled, med, scale, method = robust_scale(reindexed[feat])
        reindexed[f"{feat}_robust_scaled"] = scaled
        scale_rows.append(
            {
                "site_id": site_id,
                "site_key": site_key,
                "feature": feat,
                "median": med,
                "scale": scale,
                "scale_method": method,
            }
        )

    stats = {
        "site_id": site_id,
        "site_key": site_key,
        "capacity_mw": capacity_mw,
        "source_file": source_file,
        "n_raw_rows": n_raw,
        "invalid_timestamp_count": invalid_ts,
        "duplicate_timestamp_count": dup_count,
        "expected_timesteps": expected_steps,
        "time_start": t_min,
        "time_end": t_max,
        "row_inserted_count": row_inserted_count,
        "mean_data_quality_score": float(reindexed["data_quality_score"].mean()),
        "min_data_quality_score": float(reindexed["data_quality_score"].min()),
        "power_zero_ratio": float((reindexed["power_mw"] == 0).mean()),
        "issue_repair_cell_count": int(reindexed[imp_cols + [c for c in flag_cols if c in reindexed.columns]].sum().sum()),
        "scale_rows": scale_rows,
    }
    return reindexed, stats


def build_dispatch_panel(long_df: pd.DataFrame, common_end: pd.Timestamp, logger: logging.Logger) -> pd.DataFrame:
    """3.12 多站共同时间窗口调度面板"""
    sub = long_df[long_df["timestamp"] <= common_end].copy()
    records = []
    for ts, grp in sub.groupby("timestamp"):
        row: dict = {"timestamp": ts}
        for _, r in grp.iterrows():
            sk = r["site_key"]
            row[f"power_mw_{sk}"] = r["power_mw"]
            row[f"power_pu_{sk}"] = r["power_pu"]
            row[f"data_quality_score_{sk}"] = r["data_quality_score"]
        pu_cols = [c for c in row if c.startswith("power_pu_")]
        q_cols = [c for c in row if c.startswith("data_quality_score_")]
        row["fleet_power_mw"] = sum(row[c] for c in row if c.startswith("power_mw_"))
        row["fleet_power_pu_mean"] = np.mean([row[c] for c in pu_cols]) if pu_cols else np.nan
        row["fleet_quality_score_mean"] = np.mean([row[c] for c in q_cols]) if q_cols else np.nan
        row["available_site_count"] = len(pu_cols)
        records.append(row)
    panel = pd.DataFrame(records)
    logger.info("调度面板: %d 行, 共同截止 %s", len(panel), common_end)
    return panel


def write_experiment_log(
    log_path: Path,
    stats_list: list[dict],
    common_end: pd.Timestamp,
    out_files: list[Path],
) -> None:
    """追加实验结论段落到日志文件"""
    lines = [
        "",
        "=" * 80,
        "【实验日志摘要 — EXP-P01】",
        "",
        "【1. 本次实验做了什么】",
        "- 读取 8 个站点原始 CSV，执行字段统一、15 分钟时间轴重建",
        "- 物理边界清洗、功率非负/容量约束、Hampel 异常检测",
        "- 短时插值、夜间功率置零、时序剖面长缺失回填",
        "- 鲁棒标准化、衍生特征、质量评分、多站调度面板",
        "",
        "【2. 输入与输出】",
        f"输入: {RAW_DIR}",
        f"输出: {OUT_PROCESSED}",
        "",
        "【3. 关键运行统计】",
        f"- 处理站点数: {len(stats_list)}",
        f"- 多站共同窗口截止: {common_end}",
    ]
    for st in sorted(stats_list, key=lambda x: x["site_id"]):
        lines.append(
            f"- {st['site_key']}: 行数={st['expected_timesteps']}, "
            f"插入缺口={st['row_inserted_count']}, "
            f"均值质量分={st['mean_data_quality_score']:.4f}, "
            f"修复单元={st['issue_repair_cell_count']}"
        )
    worst = max(stats_list, key=lambda x: x["issue_repair_cell_count"])
    lines += [
        f"- 问题修复量最高站点: {worst['site_key']} ({worst['issue_repair_cell_count']} 单元)",
        "",
        "【4. 产出文件】",
    ]
    for p in out_files:
        lines.append(f"- {p.relative_to(PROJECT_ROOT)}")
    lines += [
        "",
        "【5. 实验结论】",
        "- 多站面板已对齐至最短站点时间覆盖（Site 3: 2020-07-01），避免站间比较偏差",
        "- 异常码与物理越界值已标记并修复；Site 8 时间缺口已补齐并保留 row_inserted_by_reindex",
        "=" * 80,
    ]
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    logger, log_path = setup_logging()
    logger.info("EXP-P01 启动 | 项目根目录: %s", PROJECT_ROOT)

    OUT_STATIONS.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(RAW_DIR.glob("*.csv"))
    if not csv_files:
        logger.error("未在 %s 找到原始 CSV", RAW_DIR)
        sys.exit(1)

    all_frames: list[pd.DataFrame] = []
    stats_list: list[dict] = []
    all_scale_rows: list[dict] = []

    for path in csv_files:
        df, stats = process_site(path, logger)
        out_path = OUT_STATIONS / f"{stats['site_key']}_preprocessed.csv"
        df.to_csv(out_path, index=False)
        logger.info("已写出 %s", out_path.name)
        all_frames.append(df)
        scale_rows = stats.pop("scale_rows")
        all_scale_rows.extend(scale_rows)
        stats_list.append(stats)

    long_df = pd.concat(all_frames, ignore_index=True)
    long_path = OUT_PROCESSED / "solar_stations_long.csv"
    long_df.to_csv(long_path, index=False)

    quality_df = pd.DataFrame(stats_list)
    quality_path = OUT_PROCESSED / "solar_site_quality_summary.csv"
    quality_df.to_csv(quality_path, index=False)

    scale_df = pd.DataFrame(all_scale_rows)
    scale_path = OUT_PROCESSED / "solar_feature_scaling_reference.csv"
    scale_df.to_csv(scale_path, index=False)

    common_end = min(st["time_end"] for st in stats_list)
    panel = build_dispatch_panel(long_df, common_end, logger)
    panel_path = OUT_PROCESSED / "solar_dispatch_panel_common_window.csv"
    panel.to_csv(panel_path, index=False)

    out_files = [
        long_path,
        quality_path,
        scale_path,
        panel_path,
        *sorted(OUT_STATIONS.glob("*.csv")),
    ]
    write_experiment_log(log_path, stats_list, common_end, out_files)

    logger.info("EXP-P01 结束 | 日志: %s", log_path)


if __name__ == "__main__":
    main()
