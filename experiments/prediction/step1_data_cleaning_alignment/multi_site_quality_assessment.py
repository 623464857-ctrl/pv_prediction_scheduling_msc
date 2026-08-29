"""
Site 1-8 数据质量对比评估
湿度正常范围：30%–100%（更苛刻标准）
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STATIONS_DIR = PROJECT_ROOT / "data" / "prediction" / "step1_preprocessing" / "processed" / "stations"
OUT_ROOT = PROJECT_ROOT / "data" / "prediction" / "step1_preprocessing" / "reports" / "multi_site_quality"

IRRADIANCE_COL = "total_irradiance_wm2"
POWER_COL = "power_mw"
RH_COL = "relative_humidity_pct"
MIN_IRR_FOR_DEVIATION = 200.0
POWER_DEVIATION_FRAC = 0.05
RH_NORMAL_MIN = 30.0
RH_NORMAL_MAX = 100.0
SITES = list(range(1, 9))


def setup_chinese_font():
    for name in ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC"]:
        if name in {f.name for f in fm.fontManager.ttflist}:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def load_site(site_id: int) -> pd.DataFrame:
    path = STATIONS_DIR / f"Site_{site_id}_preprocessed.csv"
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    if "source_observed_flag" in df.columns:
        df = df[df["source_observed_flag"] == 1].copy()
    return df


def hourly_daily_correlations(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = df[(df[IRRADIANCE_COL] > 0) & (df[POWER_COL].notna())].copy()
    valid["date"] = valid["timestamp"].dt.date
    valid["hour_block"] = valid["timestamp"].dt.floor("h")

    hourly_rows = []
    for ts, g in valid.groupby("hour_block"):
        if len(g) < 3:
            continue
        hourly_rows.append({
            "period_start": ts,
            "n": len(g),
            "corr": g[IRRADIANCE_COL].corr(g[POWER_COL]),
        })
    hourly = pd.DataFrame(hourly_rows)

    daily_rows = []
    for dt, g in valid.groupby("date"):
        if len(g) < 8:
            continue
        daily_rows.append({
            "period_start": pd.Timestamp(dt),
            "n": len(g),
            "corr": g[IRRADIANCE_COL].corr(g[POWER_COL]),
        })
    daily = pd.DataFrame(daily_rows)
    return hourly, daily


def find_sticky_periods(
    series: pd.Series,
    timestamps: pd.Series,
    *,
    below: float | None = None,
    above: float | None = None,
    min_hours: float = 6.0,
) -> pd.DataFrame:
    ts = pd.to_datetime(timestamps)
    vals = series.values.astype(float)
    if below is not None:
        cond = vals < below
        label = f"低湿黏滞(<{below:.0f}%)"
    elif above is not None:
        cond = vals > above
        label = f"高湿黏滞(>{above:.0f}%)"
    else:
        raise ValueError("specify below or above")

    rows = []
    i, n = 0, len(vals)
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
                "type": label,
                "mean_rh": float(np.nanmean(vals[i:j])),
            })
        i = j
    return pd.DataFrame(rows)


def analyze_site(site_id: int) -> dict:
    df = load_site(site_id)
    capacity = float(df["capacity_mw"].iloc[0]) if "capacity_mw" in df.columns else np.nan
    n = len(df)

    daylight = df[df[IRRADIANCE_COL] > 0]
    overall_corr = float(daylight[IRRADIANCE_COL].corr(daylight[POWER_COL]))
    daylight_corr = float(
        df.loc[df[IRRADIANCE_COL] > 50, IRRADIANCE_COL].corr(df.loc[df[IRRADIANCE_COL] > 50, POWER_COL])
    )
    high_irr_corr = float(
        df.loc[df[IRRADIANCE_COL] > 200, IRRADIANCE_COL].corr(df.loc[df[IRRADIANCE_COL] > 200, POWER_COL])
    )

    threshold_pwr = capacity * POWER_DEVIATION_FRAC
    dev_mask = (df[IRRADIANCE_COL] > MIN_IRR_FOR_DEVIATION) & (df[POWER_COL] <= threshold_pwr)
    dev_count = int(dev_mask.sum())
    dev_ratio = 100.0 * dev_count / n if n else 0.0

    hourly, daily = hourly_daily_correlations(df)
    low_hourly = int((hourly["corr"] < 0.6).sum()) if len(hourly) else 0
    low_daily = int((daily["corr"] < 0.6).sum()) if len(daily) else 0
    daily_median_corr = float(daily["corr"].median()) if len(daily) else np.nan

    rh = df[RH_COL].astype(float)
    rh_invalid = int(((rh < 0) | (rh > RH_NORMAL_MAX)).sum())
    rh_abnormal_low = int((rh < RH_NORMAL_MIN).sum())
    rh_normal = int(((rh >= RH_NORMAL_MIN) & (rh <= RH_NORMAL_MAX)).sum())
    rh_abnormal_low_pct = 100.0 * rh_abnormal_low / n if n else 0.0
    rh_normal_pct = 100.0 * rh_normal / n if n else 0.0
    rh_invalid_pct = 100.0 * rh_invalid / n if n else 0.0

    sticky_low = find_sticky_periods(rh, df["timestamp"], below=RH_NORMAL_MIN)
    sticky_high = find_sticky_periods(rh, df["timestamp"], above=95.0)
    sticky_count = len(sticky_low) + len(sticky_high)

    power_ok = overall_corr >= 0.85 and dev_ratio <= 5.0 and (low_daily / max(len(daily), 1)) <= 0.10
    rh_ok = rh_invalid_pct <= 1.0 and rh_abnormal_low_pct <= 20.0 and rh_normal_pct >= 60.0 and sticky_count <= 50

    return {
        "site_id": site_id,
        "site_key": f"Site_{site_id}",
        "capacity_mw": capacity,
        "n_samples": n,
        "time_start": str(df["timestamp"].min()),
        "time_end": str(df["timestamp"].max()),
        "overall_corr": overall_corr,
        "daylight_corr": daylight_corr,
        "high_irr_corr": high_irr_corr,
        "daily_median_corr": daily_median_corr,
        "deviation_count": dev_count,
        "deviation_ratio_pct": dev_ratio,
        "low_corr_hourly": low_hourly,
        "low_corr_daily": low_daily,
        "low_corr_daily_pct": 100.0 * low_daily / max(len(daily), 1),
        "rh_min": float(rh.min()),
        "rh_max": float(rh.max()),
        "rh_mean": float(rh.mean()),
        "rh_median": float(rh.median()),
        "rh_std": float(rh.std()),
        "rh_invalid": rh_invalid,
        "rh_invalid_pct": rh_invalid_pct,
        "rh_abnormal_low": rh_abnormal_low,
        "rh_abnormal_low_pct": rh_abnormal_low_pct,
        "rh_normal": rh_normal,
        "rh_normal_pct": rh_normal_pct,
        "sticky_low_count": len(sticky_low),
        "sticky_high_count": len(sticky_high),
        "sticky_total": sticky_count,
        "power_ok": power_ok,
        "rh_ok": rh_ok,
        "both_ok": power_ok and rh_ok,
        "_df": df,
        "_rh": rh,
        "_hourly": hourly,
        "_daily": daily,
        "_sticky_low": sticky_low,
        "_sticky_high": sticky_high,
    }


def composite_score(m: dict) -> float:
    """0–100 综合质量分，越高越好。"""
    s_power = (
        min(m["overall_corr"] / 0.95, 1.0) * 35
        + max(0, 1 - m["deviation_ratio_pct"] / 5.0) * 25
        + max(0, 1 - m["low_corr_daily_pct"] / 15.0) * 10
    )
    s_rh = (
        min(m["rh_normal_pct"] / 80.0, 1.0) * 20
        + max(0, 1 - m["rh_abnormal_low_pct"] / 40.0) * 5
        + max(0, 1 - m["rh_invalid_pct"] / 1.0) * 3
        + max(0, 1 - m["sticky_total"] / 200.0) * 2
    )
    bonus = 5 if m["both_ok"] else 0
    return round(s_power + s_rh + bonus, 2)


def plot_site_charts(m: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    df, rh = m["_df"], m["_rh"]
    sid = m["site_id"]

    daylight = df[df[IRRADIANCE_COL] > 0]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].scatter(daylight[IRRADIANCE_COL], daylight[POWER_COL], s=2, alpha=0.2, c="steelblue")
    axes[0].set_xlabel("总辐射照度 (W/m²)")
    axes[0].set_ylabel("实际功率 (MW)")
    axes[0].set_title(f"Site {sid} 辐照度-功率 (r={m['overall_corr']:.3f})")
    axes[0].grid(True, alpha=0.3)

    sample = df.iloc[:: max(1, len(df) // 6000)]
    axes[1].plot(sample["timestamp"], sample[IRRADIANCE_COL] / 100, color="orange", alpha=0.7, label="辐照/100")
    axes[1].plot(sample["timestamp"], sample[POWER_COL], color="green", alpha=0.7, label="功率")
    axes[1].legend()
    axes[1].set_title(f"Site {sid} 时序对比")
    axes[1].grid(True, alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(out_dir / "task1_scatter_timeseries.png", dpi=120, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.arange(-2.5, 105, 5)
    ax.hist(rh.dropna(), bins=bins, edgecolor="black", alpha=0.75, color="teal")
    ax.axvline(RH_NORMAL_MIN, color="red", linestyle="--", label=f"正常下限 {RH_NORMAL_MIN:.0f}%")
    ax.axvline(RH_NORMAL_MAX, color="red", linestyle="--", label=f"正常上限 {RH_NORMAL_MAX:.0f}%")
    ax.set_xlabel("相对湿度 (%)")
    ax.set_ylabel("样本数")
    ax.set_title(f"Site {sid} 湿度分布 (正常: {RH_NORMAL_MIN:.0f}–{RH_NORMAL_MAX:.0f}%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "task2_rh_histogram.png", dpi=120, bbox_inches="tight")
    plt.close()


def plot_comparison(summary: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    sites = summary["site_key"].tolist()
    x = np.arange(len(sites))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].bar(x, summary["overall_corr"], color="steelblue")
    axes[0, 0].axhline(0.85, color="red", linestyle="--", alpha=0.7)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(sites, rotation=45)
    axes[0, 0].set_title("功率-辐照相关系数")
    axes[0, 0].set_ylim(0, 1)

    axes[0, 1].bar(x, summary["deviation_ratio_pct"], color="coral")
    axes[0, 1].axhline(5, color="red", linestyle="--", alpha=0.7)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(sites, rotation=45)
    axes[0, 1].set_title("物理背离占比 (%)")

    axes[1, 0].bar(x, summary["rh_normal_pct"], color="seagreen", label="正常(30-100%)")
    axes[1, 0].bar(x, summary["rh_abnormal_low_pct"], bottom=summary["rh_normal_pct"], color="salmon", label="异常低(<30%)")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(sites, rotation=45)
    axes[1, 0].set_title("湿度正常/异常占比")
    axes[1, 0].legend()

    colors = ["gold" if s == summary["composite_score"].max() else "gray" for s in summary["composite_score"]]
    axes[1, 1].bar(x, summary["composite_score"], color=colors)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(sites, rotation=45)
    axes[1, 1].set_title("综合质量得分")
    axes[1, 1].set_ylim(0, 100)

    plt.tight_layout()
    fig.savefig(out_dir / "sites_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


def build_site_report(m: dict) -> str:
    verdict = "合格" if m["both_ok"] else ("部分合格" if m["power_ok"] or m["rh_ok"] else "不合格")
    lines = [
        f"# {m['site_key']} 数据质量评估报告",
        "",
        "## 数据概览",
        "",
        f"| 项目 | 值 |",
        f"|------|-----|",
        f"| 样本总数 | {m['n_samples']:,} |",
        f"| 时间范围 | {m['time_start']} ~ {m['time_end']} |",
        f"| 额定容量 | {m['capacity_mw']} MW |",
        "",
        "## 任务一：辐照度-功率",
        "",
        f"| 指标 | 值 |",
        f"|------|-----|",
        f"| 全样本相关系数 r | {m['overall_corr']:.4f} |",
        f"| 白天段 r (辐照>50) | {m['daylight_corr']:.4f} |",
        f"| 高辐照段 r (辐照>200) | {m['high_irr_corr']:.4f} |",
        f"| 逐日相关系数中位数 | {m['daily_median_corr']:.4f} |",
        f"| 物理背离样本 | {m['deviation_count']} ({m['deviation_ratio_pct']:.2f}%) |",
        f"| 低相关日 (r<0.6) | {m['low_corr_daily']} 天 ({m['low_corr_daily_pct']:.1f}%) |",
        f"| 功率指标判定 | {'合格' if m['power_ok'] else '不合格'} |",
        "",
        "![散点与时序](task1_scatter_timeseries.png)",
        "",
        "## 任务二：湿度（正常范围 30%–100%）",
        "",
        f"| 指标 | 值 |",
        f"|------|-----|",
        f"| 最小值 | {m['rh_min']:.2f}% |",
        f"| 最大值 | {m['rh_max']:.2f}% |",
        f"| 均值 | {m['rh_mean']:.2f}% |",
        f"| 中位数 | {m['rh_median']:.2f}% |",
        f"| 正常湿度 (30–100%) | {m['rh_normal']:,} ({m['rh_normal_pct']:.2f}%) |",
        f"| 异常低湿 (<30%) | {m['rh_abnormal_low']:,} ({m['rh_abnormal_low_pct']:.2f}%) |",
        f"| 无效值 (超出0–100) | {m['rh_invalid']:,} ({m['rh_invalid_pct']:.2f}%) |",
        f"| 低湿黏滞段 (<30%, ≥6h) | {m['sticky_low_count']} |",
        f"| 高湿黏滞段 (>95%, ≥6h) | {m['sticky_high_count']} |",
        f"| 湿度指标判定 | {'合格' if m['rh_ok'] else '不合格'} |",
        "",
        "![湿度直方图](task2_rh_histogram.png)",
        "",
        f"## 综合结论：**{verdict}**（综合得分 {m['composite_score']:.1f}/100）",
        "",
    ]
    return "\n".join(lines)


def build_comparison_report(summary: pd.DataFrame, best: dict) -> str:
    lines = [
        "# Site 1–8 数据质量对比报告",
        "",
        f"**湿度正常标准（更苛刻）：{RH_NORMAL_MIN:.0f}% – {RH_NORMAL_MAX:.0f}%**",
        "",
        "## 综合排名",
        "",
        "| 排名 | 站点 | 容量(MW) | 综合得分 | 功率-辐照r | 背离% | 正常湿度% | 异常低湿% | 功率合格 | 湿度合格 | 双项合格 |",
        "|------|------|----------|----------|------------|-------|-----------|-----------|----------|----------|----------|",
    ]
    for rank, (_, r) in enumerate(summary.iterrows(), 1):
        lines.append(
            f"| {rank} | {r['site_key']} | {r['capacity_mw']:.0f} | **{r['composite_score']:.1f}** | "
            f"{r['overall_corr']:.3f} | {r['deviation_ratio_pct']:.2f} | {r['rh_normal_pct']:.1f} | "
            f"{r['rh_abnormal_low_pct']:.1f} | {'是' if r['power_ok'] else '否'} | "
            f"{'是' if r['rh_ok'] else '否'} | {'是' if r['both_ok'] else '否'} |"
        )

    lines += [
        "",
        f"## 结论：质量最高站点为 **{best['site_key']}**（{best['capacity_mw']:.0f} MW）",
        "",
        f"- 综合得分：**{best['composite_score']:.1f}/100**",
        f"- 功率-辐照 r = {best['overall_corr']:.4f}，物理背离 {best['deviation_ratio_pct']:.2f}%",
        f"- 正常湿度占比 {best['rh_normal_pct']:.1f}%，异常低湿 {best['rh_abnormal_low_pct']:.1f}%",
        "",
        "![对比图](sites_comparison.png)",
        "",
        "## 决策建议",
        "",
    ]

    both_ok_sites = summary[summary["both_ok"]]["site_key"].tolist()
    if both_ok_sites:
        lines.append(f"1. **优先选用**：{', '.join(both_ok_sites)}（功率与湿度双项合格）")
    else:
        lines.append("1. **无站点同时满足功率与湿度双项合格**；建议按综合得分优先，并对不合格项做特征剔除。")

    lines += [
        f"2. **综合质量最高**：{best['site_key']}，适合作为功率预测主训练集。",
        "3. 对异常低湿占比高的站点，建模时移除或替换湿度特征（如 WRF 再分析）。",
        "4. 剔除各站低相关日（r<0.6）后再进入 EXP-P01 后续流程。",
        "",
        "## 湿度判定标准（本次采用）",
        "",
        f"- 正常：{RH_NORMAL_MIN:.0f}% – {RH_NORMAL_MAX:.0f}%",
        f"- 异常低湿：< {RH_NORMAL_MIN:.0f}%",
        "- 无效：超出 [0, 100]",
        f"- 合格阈值：正常占比 ≥ 60%，异常低湿 ≤ 20%，无效值 ≤ 1%，黏滞段 ≤ 50",
        "",
    ]
    return "\n".join(lines)


def main():
    setup_chinese_font()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    results = []
    for site_id in SITES:
        print(f"Analyzing Site {site_id}...")
        m = analyze_site(site_id)
        m["composite_score"] = composite_score(m)

        site_dir = OUT_ROOT / f"site{site_id}"
        plot_site_charts(m, site_dir)
        m["_hourly"].to_csv(site_dir / "hourly_correlations.csv", index=False)
        m["_daily"].to_csv(site_dir / "daily_correlations.csv", index=False)
        m["_daily"][m["_daily"]["corr"] < 0.6].to_csv(site_dir / "low_corr_daily.csv", index=False)
        if len(m["_sticky_low"]):
            m["_sticky_low"].to_csv(site_dir / "rh_sticky_low.csv", index=False)
        if len(m["_sticky_high"]):
            m["_sticky_high"].to_csv(site_dir / "rh_sticky_high.csv", index=False)

        report = build_site_report(m)
        (site_dir / "REPORT.md").write_text(report, encoding="utf-8")

        row = {k: v for k, v in m.items() if not k.startswith("_")}
        results.append(row)
        print(f"  score={m['composite_score']:.1f}, r={m['overall_corr']:.3f}, rh_normal={m['rh_normal_pct']:.1f}%")

    summary = pd.DataFrame(results).sort_values("composite_score", ascending=False).reset_index(drop=True)
    summary["rank"] = summary.index + 1
    summary.to_csv(OUT_ROOT / "comparison_summary.csv", index=False)

    best = summary.iloc[0].to_dict()
    plot_comparison(summary, OUT_ROOT)
    comp_report = build_comparison_report(summary, best)
    (OUT_ROOT / "COMPARISON_REPORT.md").write_text(comp_report, encoding="utf-8")

    print(f"\nBest site: {best['site_key']} (score={best['composite_score']:.1f})")
    print(f"Report: {OUT_ROOT / 'COMPARISON_REPORT.md'}")


if __name__ == "__main__":
    main()
