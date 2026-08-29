"""
Site 1 原始数据质量评估脚本
任务一：辐照度-功率物理一致性
任务二：湿度数据合理性
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "Solar station site 1 (Nominal capacity-50MW).csv"
PREPROCESSED_PATH = (
    PROJECT_ROOT / "data" / "prediction" / "step1_preprocessing" / "processed" / "stations" / "Site_1_preprocessed.csv"
)
OUT_DIR = PROJECT_ROOT / "data" / "prediction" / "step1_preprocessing" / "reports" / "site1_quality_assessment"
CAPACITY_MW = 50.0
IRRADIANCE_COL = "total_irradiance_wm2"
POWER_COL = "power_mw"
RH_COL = "relative_humidity_pct"
MIN_IRR_FOR_DEVIATION = 200.0
POWER_DEVIATION_FRAC = 0.05


def setup_chinese_font():
    for name in ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC"]:
        if name in {f.name for f in fm.fontManager.ttflist}:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def _is_lfs_pointer(path: Path) -> bool:
    if not path.exists():
        return True
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        head = f.read(80)
    return head.startswith("version https://git-lfs.github.com/spec/v1")


def load_raw_site1() -> tuple[pd.DataFrame, str]:
    """优先加载原始 CSV；若 LFS 未就绪则使用预处理中 source_observed 记录。"""
    if not _is_lfs_pointer(RAW_PATH):
        for enc in ("utf-8", "gbk", "latin-1"):
            try:
                raw = pd.read_csv(RAW_PATH, encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            raw = pd.read_csv(RAW_PATH, encoding="latin-1")
        raw = raw.loc[:, ~raw.columns.astype(str).str.match(r"^Unnamed")]
        alias = {
            "timestamp": ["time(year-month-day h:m:s)", "time", "timestamp", "datetime"],
            IRRADIANCE_COL: ["total solar irradiance (w/m2)", "total solar irradiance"],
            POWER_COL: ["power (mw)", "power"],
            RH_COL: ["relative humidity (%)", "relative humidity"],
        }
        col_map = {}
        lower_cols = {c.lower().strip(): c for c in raw.columns}
        for std, names in alias.items():
            for n in names:
                if n.lower() in lower_cols:
                    col_map[lower_cols[n.lower()]] = std
                    break
        df = raw.rename(columns=col_map)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        df["capacity_mw"] = CAPACITY_MW
        df["data_source"] = "raw_csv"
        return df, "raw_csv"

    df = pd.read_csv(PREPROCESSED_PATH, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    if "source_observed_flag" in df.columns:
        df = df[df["source_observed_flag"] == 1].copy()
    df["data_source"] = "preprocessed_observed"
    return df, "preprocessed_observed"


def detect_anomaly_windows(df: pd.DataFrame, window: int = 4) -> pd.DataFrame:
    """检测异常时段：辐照升功率降 / 辐照稳功率大幅波动。"""
    d = df.copy()
    d["d_irr"] = d[IRRADIANCE_COL].diff()
    d["d_pwr"] = d[POWER_COL].diff()
    d["irr_std"] = d[IRRADIANCE_COL].rolling(window, min_periods=window).std()
    d["pwr_std"] = d[POWER_COL].rolling(window, min_periods=window).std()
    d["irr_mean"] = d[IRRADIANCE_COL].rolling(window, min_periods=window).mean()

    # 仅白天有效辐照段
    daylight = d[IRRADIANCE_COL] > 50

    inv_trend = daylight & (d["d_irr"] > 20) & (d["d_pwr"] < -0.5)
    stable_irr_volatile_pwr = (
        daylight
        & (d["irr_std"] < 15)
        & (d["irr_mean"] > 100)
        & (d["pwr_std"] > 1.5)
    )

    records = []
    for mask, kind in [
        (inv_trend, "功率随辐照度上升而下降"),
        (stable_irr_volatile_pwr, "辐照度稳定但功率大幅波动"),
    ]:
        sub = d.loc[mask, ["timestamp", IRRADIANCE_COL, POWER_COL]].copy()
        sub["anomaly_type"] = kind
        records.append(sub)
    if not records:
        return pd.DataFrame(columns=["timestamp", IRRADIANCE_COL, POWER_COL, "anomaly_type"])
    return pd.concat(records, ignore_index=True)


def hourly_daily_correlations(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """逐小时 / 逐日相关系数。"""
    valid = df[(df[IRRADIANCE_COL] > 0) & (df[POWER_COL].notna())].copy()
    valid["date"] = valid["timestamp"].dt.date
    valid["hour_block"] = valid["timestamp"].dt.floor("h")

    hourly_rows = []
    for ts, g in valid.groupby("hour_block"):
        if len(g) < 3:
            continue
        r = g[IRRADIANCE_COL].corr(g[POWER_COL])
        hourly_rows.append({"period_start": ts, "period_end": ts + pd.Timedelta(hours=1), "n": len(g), "corr": r, "granularity": "hourly"})
    hourly = pd.DataFrame(hourly_rows)

    daily_rows = []
    for dt, g in valid.groupby("date"):
        if len(g) < 8:
            continue
        r = g[IRRADIANCE_COL].corr(g[POWER_COL])
        daily_rows.append({"period_start": pd.Timestamp(dt), "period_end": pd.Timestamp(dt) + pd.Timedelta(days=1), "n": len(g), "corr": r, "granularity": "daily"})
    daily = pd.DataFrame(daily_rows)
    return hourly, daily


def physical_deviation_stats(df: pd.DataFrame) -> dict:
    threshold_pwr = CAPACITY_MW * POWER_DEVIATION_FRAC
    mask = (df[IRRADIANCE_COL] > MIN_IRR_FOR_DEVIATION) & (df[POWER_COL] <= threshold_pwr)
    n_dev = int(mask.sum())
    n_total = len(df)
    return {
        "threshold_irradiance_wm2": MIN_IRR_FOR_DEVIATION,
        "threshold_power_mw": threshold_pwr,
        "deviation_count": n_dev,
        "total_count": n_total,
        "deviation_ratio_pct": 100.0 * n_dev / n_total if n_total else 0.0,
    }


def rh_descriptive_stats(series: pd.Series) -> dict:
    s = series.dropna()
    return {
        "min": float(s.min()) if len(s) else np.nan,
        "max": float(s.max()) if len(s) else np.nan,
        "mean": float(s.mean()) if len(s) else np.nan,
        "median": float(s.median()) if len(s) else np.nan,
        "std": float(s.std()) if len(s) else np.nan,
        "q25": float(s.quantile(0.25)) if len(s) else np.nan,
        "q75": float(s.quantile(0.75)) if len(s) else np.nan,
        "count": int(len(s)),
    }


def find_sticky_periods(series: pd.Series, timestamps: pd.Series, high: bool = True, min_hours: float = 6.0) -> pd.DataFrame:
    """连续 min_hours 以上 RH>95% 或 RH<5%。"""
    ts = pd.to_datetime(timestamps)
    vals = series.values
    threshold = 95.0 if high else 5.0
    cond = vals > threshold if high else vals < threshold
    rows = []
    i = 0
    n = len(vals)
    while i < n:
        if not cond[i] or np.isnan(vals[i]):
            i += 1
            continue
        j = i + 1
        while j < n and cond[j] and not np.isnan(vals[j]):
            j += 1
        duration_h = (ts.iloc[j - 1] - ts.iloc[i]).total_seconds() / 3600.0
        if duration_h >= min_hours:
            rows.append({
                "start": ts.iloc[i],
                "end": ts.iloc[j - 1],
                "duration_hours": duration_h,
                "type": "高湿黏滞(>95%)" if high else "低湿黏滞(<5%)",
                "mean_rh": float(np.nanmean(vals[i:j])),
            })
        i = j
    return pd.DataFrame(rows)


def plot_task1(df: pd.DataFrame, out_dir: Path, overall_corr: float):
    out_dir.mkdir(parents=True, exist_ok=True)
    daylight = df[df[IRRADIANCE_COL] > 0].copy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].scatter(daylight[IRRADIANCE_COL], daylight[POWER_COL], s=3, alpha=0.25, c="steelblue")
    axes[0].set_xlabel("总辐射照度 (W/m²)")
    axes[0].set_ylabel("实际功率 (MW)")
    axes[0].set_title(f"辐照度-功率散点图 (r={overall_corr:.3f})")
    axes[0].grid(True, alpha=0.3)

    sample = df.iloc[:: max(1, len(df) // 8000)]
    ax2 = axes[1]
    ax2.plot(sample["timestamp"], sample[IRRADIANCE_COL] / 100, color="orange", alpha=0.7, label="辐照度/100")
    ax2.plot(sample["timestamp"], sample[POWER_COL], color="green", alpha=0.7, label="功率 (MW)")
    ax2.set_xlabel("时间")
    ax2.set_ylabel("归一化对比")
    ax2.set_title("辐照度与功率时间序列对比")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(out_dir / "task1_scatter_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 标注异常时段（选代表性月份）
    anomalies = detect_anomaly_windows(df)
    if len(anomalies) > 0:
        month = anomalies["timestamp"].dt.to_period("M").value_counts().index[0]
        sub = df[df["timestamp"].dt.to_period("M") == month].copy()
        fig, ax = plt.subplots(figsize=(16, 5))
        ax2t = ax.twinx()
        ax.plot(sub["timestamp"], sub[IRRADIANCE_COL], color="orange", alpha=0.6, label="辐照度")
        ax2t.plot(sub["timestamp"], sub[POWER_COL], color="green", alpha=0.6, label="功率")
        an_sub = anomalies[anomalies["timestamp"].dt.to_period("M") == month]
        for _, row in an_sub.head(50).iterrows():
            ax.axvline(row["timestamp"], color="red", alpha=0.15, linewidth=0.8)
        ax.set_title(f"异常时段标注示例 ({month})")
        ax.set_xlabel("时间")
        ax.set_ylabel("辐照度 (W/m²)")
        ax2t.set_ylabel("功率 (MW)")
        fig.savefig(out_dir / "task1_anomaly_annotated.png", dpi=150, bbox_inches="tight")
        plt.close()


def plot_task2_rh_hist(rh: pd.Series, out_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.arange(-2.5, 105, 5)
    ax.hist(rh.dropna(), bins=bins, edgecolor="black", alpha=0.75, color="teal")
    ax.axvline(10, color="red", linestyle="--", label="10%")
    ax.axvline(90, color="red", linestyle="--", label="90%")
    ax.set_xlabel("相对湿度 (%)")
    ax.set_ylabel("样本数")
    ax.set_title("湿度分布直方图 (区间宽度 5%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "task2_rh_histogram.png", dpi=150, bbox_inches="tight")
    plt.close()


def build_report(
    df: pd.DataFrame,
    source: str,
    overall_corr: float,
    dev_stats: dict,
    hourly: pd.DataFrame,
    daily: pd.DataFrame,
    anomalies: pd.DataFrame,
    rh_stats: dict,
    rh_invalid: int,
    rh_high: int,
    rh_low: int,
    sticky: pd.DataFrame,
) -> str:
    n = len(df)
    t0, t1 = df["timestamp"].min(), df["timestamp"].max()
    low_h = hourly[hourly["corr"] < 0.6] if len(hourly) else pd.DataFrame()
    low_d = daily[daily["corr"] < 0.6] if len(daily) else pd.DataFrame()

    inv_ratio = dev_stats["deviation_ratio_pct"]
    invalid_rh_pct = 100.0 * rh_invalid / n if n else 0
    extreme_rh_pct = 100.0 * (rh_high + rh_low) / n if n else 0

    # 判定
    power_ok = overall_corr >= 0.7 and inv_ratio <= 5.0
    rh_ok = invalid_rh_pct <= 1.0 and extreme_rh_pct <= 30.0

    if power_ok and rh_ok:
        verdict = "基本合格"
        action = "建议继续使用 Site 1 数据集，但对异常时段进行剔除或插补后再建模。"
    elif not power_ok and not rh_ok:
        verdict = "不合格（功率-辐照与湿度均存在显著问题）"
        action = "建议优先更换数据集；更换紧迫性：**高**。"
    elif not power_ok:
        verdict = "部分不合格（功率-辐照物理一致性不达标）"
        action = "建议更换数据集或仅使用高相关子集；更换紧迫性：**中高**。"
    else:
        verdict = "部分不合格（湿度数据显著异常）"
        action = "若功率一致性良好，可保留 Site 1 并剔除/修复湿度字段；否则建议更换；更换紧迫性：**中**。"

    lines = [
        "# Site 1 数据质量评估报告",
        "",
        "## 数据概览",
        "",
        f"| 项目 | 值 |",
        f"|------|-----|",
        f"| 数据来源 | {source} |",
        f"| 样本总数 | {n:,} |",
        f"| 时间范围 | {t0} ~ {t1} |",
        f"| 额定容量 | {CAPACITY_MW} MW |",
        f"| 关键字段 | `{IRRADIANCE_COL}`（总辐射照度）, `{POWER_COL}`（实际功率）, `{RH_COL}`（相对湿度） |",
        f"| 采样频率 | 约 15 分钟（对齐后） |",
        "",
        "## 任务一：辐照度-功率物理一致性",
        "",
        f"### 全样本 Pearson 相关系数: **{overall_corr:.4f}**",
        "",
        "![散点与时序对比](task1_scatter_timeseries.png)",
        "",
        "### 物理背离样本统计",
        "",
        f"| 指标 | 值 |",
        f"|------|-----|",
        f"| 判定条件 | 辐照度 > {MIN_IRR_FOR_DEVIATION} W/m² 且 功率 ≤ {dev_stats['threshold_power_mw']} MW（额定 5%） |",
        f"| 背离样本数 | {dev_stats['deviation_count']:,} |",
        f"| 占总样本比例 | {inv_ratio:.2f}% |",
        "",
        f"### 相关系数 < 0.6 的逐小时时段（共 {len(low_h)} 个）",
        "",
    ]
    if len(low_h):
        lines.append("| 时段起始 | 时段结束 | 样本数 | 相关系数 |")
        lines.append("|----------|----------|--------|----------|")
        for _, r in low_h.head(30).iterrows():
            lines.append(f"| {r['period_start']} | {r['period_end']} | {int(r['n'])} | {r['corr']:.3f} |")
        if len(low_h) > 30:
            lines.append(f"| ... | 另有 {len(low_h)-30} 个时段见 low_corr_hourly.csv | | |")
    else:
        lines.append("无（所有有效小时段相关系数均 ≥ 0.6）")

    lines += [
        "",
        f"### 相关系数 < 0.6 的逐日时段（共 {len(low_d)} 天）",
        "",
    ]
    if len(low_d):
        lines.append("| 日期 | 样本数 | 相关系数 |")
        lines.append("|------|--------|----------|")
        for _, r in low_d.iterrows():
            lines.append(f"| {r['period_start'].date()} | {int(r['n'])} | {r['corr']:.3f} |")
    else:
        lines.append("无（所有有效日相关系数均 ≥ 0.6）")

    lines += [
        "",
        f"### 异常时段检测（共 {len(anomalies):,} 个时间点）",
        "",
        "![异常标注](task1_anomaly_annotated.png)",
        "",
        "类型分布：",
        "",
    ]
    if len(anomalies):
        for k, v in anomalies["anomaly_type"].value_counts().items():
            lines.append(f"- {k}: {v:,}")
    else:
        lines.append("- 未检测到显著异常模式")

    lines += [
        "",
        "## 任务二：湿度数据合理性",
        "",
        "### 描述性统计",
        "",
        "| 统计量 | 值 (%) |",
        "|--------|--------|",
    ]
    for k, label in [("min", "最小值"), ("max", "最大值"), ("mean", "均值"), ("median", "中位数"), ("std", "标准差"), ("q25", "25%分位"), ("q75", "75%分位")]:
        lines.append(f"| {label} | {rh_stats[k]:.2f} |")

    lines += [
        "",
        "### 异常值统计",
        "",
        "| 类型 | 数量 | 占比 |",
        "|------|------|------|",
        f"| 超出 [0,100] 无效值 | {rh_invalid:,} | {invalid_rh_pct:.2f}% |",
        f"| 极端高湿 (>90%) | {rh_high:,} | {100.0*rh_high/n:.2f}% |",
        f"| 极端低湿 (<10%) | {rh_low:,} | {100.0*rh_low/n:.2f}% |",
        "",
        f"### 黏滞时段（连续 ≥6 小时 RH>95% 或 <5%，共 {len(sticky)} 段）",
        "",
    ]
    if len(sticky):
        lines.append("| 起始 | 结束 | 时长(h) | 类型 | 平均RH |")
        lines.append("|------|------|---------|------|--------|")
        for _, r in sticky.iterrows():
            lines.append(f"| {r['start']} | {r['end']} | {r['duration_hours']:.1f} | {r['type']} | {r['mean_rh']:.1f} |")
    else:
        lines.append("无显著黏滞时段")

    lines += [
        "",
        "![湿度直方图](task2_rh_histogram.png)",
        "",
        "## 综合结论",
        "",
        f"**总体判定：{verdict}**",
        "",
        "### 依据",
        "",
        f"1. **功率-辐照背离**：全样本 r={overall_corr:.3f}（阈值 0.7），物理背离占比 {inv_ratio:.2f}%（阈值 5%）→ {'严重' if not power_ok else '可接受'}。",
        f"2. **湿度异常**：无效值占比 {invalid_rh_pct:.2f}%（阈值 1%），极端值占比 {extreme_rh_pct:.2f}%（阈值 30%）→ {'显著异常' if not rh_ok else '可接受'}。",
        "",
        "## 决策建议",
        "",
        f"**{action}**",
        "",
        "### 后续操作步骤",
        "",
    ]
    if power_ok and rh_ok:
        lines += [
            "1. 剔除相关系数 < 0.6 的小时/日时段或标记为不可用训练样本；",
            "2. 对物理背离样本（高辐照低功率）进行人工复核或限功率裁剪；",
            "3. 湿度极端值可做 Winsorize 或缺失化处理；",
            "4. 重新运行 EXP-P01 预处理并更新质量分数后再进入建模。",
        ]
    else:
        lines += [
            "1. **立即**对其他场站（Site 2/4/5）执行相同质量检查；",
            "2. 新数据集预筛选清单：",
            "   - 辐照度字段完整率 ≥ 95%，且无大面积连续缺失；",
            "   - 功率-辐照 Pearson r ≥ 0.85（白天段）；",
            "   - 物理背离样本占比 ≤ 3%；",
            "   - 湿度有效覆盖率 ≥ 98%，无效值 < 1%；",
            "   - RH 主要分布集中在 30%–70%；",
            "   - 采样频率稳定（15 min），时间戳无大量重复/乱序；",
            "   - 额定容量元数据明确，功率单位一致。",
            "3. 选定替代场站后，重复本报告流程并对比后再定训练集。",
        ]

    return "\n".join(lines)


def main():
    setup_chinese_font()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df, source = load_raw_site1()
    print(f"Loaded {len(df)} rows from {source}")

    daylight = df[df[IRRADIANCE_COL] > 0]
    overall_corr = daylight[IRRADIANCE_COL].corr(daylight[POWER_COL])
    print(f"Overall irradiance-power correlation: {overall_corr:.4f}")

    dev_stats = physical_deviation_stats(df)
    hourly, daily = hourly_daily_correlations(df)
    anomalies = detect_anomaly_windows(df)

    rh = df[RH_COL]
    rh_stats = rh_descriptive_stats(rh)
    rh_invalid = int(((rh < 0) | (rh > 100)).sum())
    rh_high = int((rh > 90).sum())
    rh_low = int((rh < 10).sum())
    sticky = pd.concat([
        find_sticky_periods(rh, df["timestamp"], high=True),
        find_sticky_periods(rh, df["timestamp"], high=False),
    ], ignore_index=True)

    plot_task1(df, OUT_DIR, overall_corr)
    plot_task2_rh_hist(rh, OUT_DIR)

    hourly.to_csv(OUT_DIR / "hourly_correlations.csv", index=False)
    daily.to_csv(OUT_DIR / "daily_correlations.csv", index=False)
    hourly[hourly["corr"] < 0.6].to_csv(OUT_DIR / "low_corr_hourly.csv", index=False)
    daily[daily["corr"] < 0.6].to_csv(OUT_DIR / "low_corr_daily.csv", index=False)
    anomalies.to_csv(OUT_DIR / "anomaly_timestamps.csv", index=False)
    sticky.to_csv(OUT_DIR / "rh_sticky_periods.csv", index=False)

    report = build_report(
        df, source, overall_corr, dev_stats, hourly, daily, anomalies,
        rh_stats, rh_invalid, rh_high, rh_low, sticky,
    )
    (OUT_DIR / "REPORT.md").write_text(report, encoding="utf-8")
    print(f"Report saved to {OUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
