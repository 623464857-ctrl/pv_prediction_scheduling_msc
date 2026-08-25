"""
Site1 数据预处理分析与 H1/H4/H16 预测曲线可视化脚本

生成内容：
1. Site1 预处理后的数据分析和可视化
2. H1/H4/H16 预测曲线对比图（每行一个H，一张图3张长图）

用法：
    python -m experiments.prediction.step4_optuna_hybrid.plot_site1_analysis
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import matplotlib.font_manager as fm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

# 设置中文字体
def setup_chinese_font():
    """设置支持中文的字体"""
    # 尝试多个可能的中文字体
    font_names = [
        'Microsoft YaHei',  # 微软雅黑
        'SimHei',           # 黑体
        'SimSun',            # 宋体
        'KaiTi',            # 楷体
        'FangSong',         # 仿宋
        'WenQuanYi Micro Hei',  # 文泉驿
        'Noto Sans CJK SC',     # Google Noto
    ]

    # 查找系统中可用的字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]

    for font_name in font_names:
        if font_name in available_fonts:
            plt.rcParams['font.sans-serif'] = [font_name]
            print(f"使用字体: {font_name}")
            break
    else:
        # 如果都没找到，使用第一个可用的
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        print("警告: 未找到中文字体，使用默认字体")

    # 设置负号显示
    plt.rcParams['axes.unicode_minus'] = False

setup_chinese_font()

# 路径定义
DATA_DIR = PROJECT_ROOT / "data" / "prediction" / "step1_preprocessing" / "processed" / "stations"
PREPROCESSED_DIR = PROJECT_ROOT / "data" / "prediction" / "step1_preprocessing" / "processed"
SAMPLES_DIR = PROJECT_ROOT / "data" / "prediction" / "step4_optuna_hybrid" / "samples"
PRED_DIR = PROJECT_ROOT / "data" / "prediction" / "step4_optuna_hybrid" / "predictions"
FIGURES_DIR = PROJECT_ROOT / "data" / "prediction" / "step4_optuna_hybrid" / "figures"


def load_site1_data():
    """加载Site1预处理后的完整数据"""
    csv_path = DATA_DIR / "Site_1_with_wrf.csv"
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_predictions(horizon: int):
    """加载指定horizon的CNN-BiLSTM预测结果"""
    pred_path = PRED_DIR / f"h{horizon}" / "cnn_bilstm_test.csv"
    ts_path = SAMPLES_DIR / f"h{horizon}" / "test_timestamps.csv"

    if not pred_path.exists():
        print(f"警告: 预测文件不存在 {pred_path}")
        return None

    df_pred = pd.read_csv(pred_path)
    df_pred["timestamp"] = pd.to_datetime(df_pred["timestamp"])
    return df_pred


def plot_site1_data_overview(df: pd.DataFrame, output_dir: Path):
    """绘制Site1数据概览图"""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(16, 12))

    # 2019年数据
    df_2019 = df[df["timestamp"].dt.year == 2019]
    axes[0].plot(df_2019["timestamp"], df_2019["power_pu"], alpha=0.7, linewidth=0.5)
    axes[0].set_title("Site 1 功率 (p.u.) - 2019年", fontsize=14)
    axes[0].set_xlabel("时间")
    axes[0].set_ylabel("功率 (p.u.)")
    axes[0].grid(True, alpha=0.3)

    # 2020年数据
    df_2020 = df[df["timestamp"].dt.year == 2020]
    axes[1].plot(df_2020["timestamp"], df_2020["power_pu"], alpha=0.7, linewidth=0.5, color="orange")
    axes[1].set_title("Site 1 功率 (p.u.) - 2020年", fontsize=14)
    axes[1].set_xlabel("时间")
    axes[1].set_ylabel("功率 (p.u.)")
    axes[1].grid(True, alpha=0.3)

    # 完整时间序列
    axes[2].plot(df["timestamp"], df["power_pu"], alpha=0.7, linewidth=0.3, color="green")
    axes[2].set_title("Site 1 功率 (p.u.) - 完整时间序列 (2019-2020)", fontsize=14)
    axes[2].set_xlabel("时间")
    axes[2].set_ylabel("功率 (p.u.)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "site1_power_overview.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已保存: {output_dir / 'site1_power_overview.png'}")


def plot_site1_monthly_pattern(df: pd.DataFrame, output_dir: Path):
    """绘制月度功率模式图"""
    output_dir.mkdir(parents=True, exist_ok=True)

    df["month"] = df["timestamp"].dt.month
    df["hour"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60

    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    axes = axes.flatten()

    month_names = ["一月", "二月", "三月", "四月", "五月", "六月",
                   "七月", "八月", "九月", "十月", "十一月", "十二月"]

    for month in range(1, 13):
        df_month = df[df["month"] == month]
        if len(df_month) == 0:
            continue

        hourly_power = df_month.groupby("hour")["power_pu"].mean()

        axes[month - 1].plot(hourly_power.index, hourly_power.values, "b-", linewidth=2)
        axes[month - 1].fill_between(hourly_power.index, hourly_power.values, alpha=0.3)
        axes[month - 1].set_title(month_names[month - 1], fontsize=12)
        axes[month - 1].set_xlabel("小时 (Hour)")
        axes[month - 1].set_ylabel("平均功率 (p.u.)")
        axes[month - 1].set_xlim(0, 24)
        axes[month - 1].set_ylim(0, 1)
        axes[month - 1].grid(True, alpha=0.3)

    plt.suptitle("Site 1 各月平均日功率曲线", fontsize=16, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "site1_monthly_pattern.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已保存: {output_dir / 'site1_monthly_pattern.png'}")


def plot_site1_data_quality(df: pd.DataFrame, output_dir: Path):
    """绘制数据质量分析图"""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 数据完整性统计
    flag_cols = [c for c in df.columns if "_flag" in c or "_imputed_flag" in c]
    flag_counts = df[flag_cols].sum()
    flag_counts = flag_counts[flag_counts > 0]

    # 中文标签映射
    flag_label_map = {
        "raw_missing_flag": "原始缺失",
        "raw_invalid_flag": "原始无效",
        "outlier_flag": "异常值",
        "imputed_flag": "已插值",
        "negative_clipped_flag": "负值截断",
    }

    if len(flag_counts) > 0:
        labels = [flag_label_map.get(c.replace("_imputed_", "_").replace("total_irradiance_", "").replace("direct_normal_irradiance_", "").replace("global_horizontal_irradiance_", "").replace("air_temperature_", "").replace("atmosphere_", "").replace("relative_humidity_", "").replace("power_mw_", ""), c.replace("_flag", "").replace("_", " ")) for c in flag_counts.index]
        axes[0, 0].barh(range(len(flag_counts)), flag_counts.values)
        axes[0, 0].set_yticks(range(len(flag_counts)))
        axes[0, 0].set_yticklabels(labels, fontsize=8)
        axes[0, 0].set_xlabel("记录数")
        axes[0, 0].set_title("数据质量标志统计")
    else:
        axes[0, 0].text(0.5, 0.5, "无数据质量问题", ha="center", va="center")
        axes[0, 0].set_title("数据质量标志统计")

    # 2. 数据质量分数分布
    if "data_quality_score" in df.columns:
        axes[0, 1].hist(df["data_quality_score"].dropna(), bins=50, edgecolor="black", alpha=0.7)
        axes[0, 1].set_xlabel("数据质量分数")
        axes[0, 1].set_ylabel("频数")
        axes[0, 1].set_title("数据质量分数分布")
        mean_score = df["data_quality_score"].mean()
        axes[0, 1].axvline(mean_score, color="red", linestyle="--", label=f"均值: {mean_score:.2f}")
        axes[0, 1].legend()

    # 3. 缺失值/插值统计
    imputed_counts = df[[c for c in df.columns if "imputed_flag" in c]].sum()
    imputed_counts = imputed_counts[imputed_counts > 0]

    if len(imputed_counts) > 0:
        labels = [c.replace("_imputed_flag", "") for c in imputed_counts.index]
        axes[1, 0].bar(range(len(imputed_counts)), imputed_counts.values, color="coral")
        axes[1, 0].set_xticks(range(len(imputed_counts)))
        axes[1, 0].set_xticklabels(labels, rotation=45, ha="right")
        axes[1, 0].set_ylabel("插值记录数")
        axes[1, 0].set_title("特征插值统计")
    else:
        axes[1, 0].text(0.5, 0.5, "无需插值", ha="center", va="center")
        axes[1, 0].set_title("特征插值统计")

    # 4. 各月份数据量
    monthly_counts = df.groupby(df["timestamp"].dt.to_period("M")).size()
    axes[1, 1].bar(range(len(monthly_counts)), monthly_counts.values, color="steelblue")
    axes[1, 1].set_xticks(range(len(monthly_counts)))
    axes[1, 1].set_xticklabels([str(p) for p in monthly_counts.index], rotation=45, ha="right")
    axes[1, 1].set_ylabel("数据点数")
    axes[1, 1].set_title("各月份数据量")
    mean_count = monthly_counts.mean()
    axes[1, 1].axhline(mean_count, color="red", linestyle="--", label=f"均值: {mean_count:.0f}")
    axes[1, 1].legend()

    plt.tight_layout()
    fig.savefig(output_dir / "site1_data_quality.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已保存: {output_dir / 'site1_data_quality.png'}")


def plot_site1_feature_correlation(df: pd.DataFrame, output_dir: Path):
    """绘制特征相关性热力图"""
    output_dir.mkdir(parents=True, exist_ok=True)

    key_features = [
        "power_pu", "total_irradiance_wm2", "direct_normal_irradiance_wm2",
        "global_horizontal_irradiance_wm2", "air_temperature_c",
        "relative_humidity_pct", "wrf_gti_wm2", "wrf_temperature_c",
        "wrf_clearness_index", "wrf_cloud_cover_ratio"
    ]

    key_features = [f for f in key_features if f in df.columns]

    # 中文标签映射
    feature_label_map = {
        "power_pu": "实际功率",
        "total_irradiance_wm2": "总辐射照度",
        "direct_normal_irradiance_wm2": "直接法向辐射",
        "global_horizontal_irradiance_wm2": "水平辐射照度",
        "air_temperature_c": "环境温度",
        "relative_humidity_pct": "相对湿度",
        "wrf_gti_wm2": "WRF斜面辐射",
        "wrf_temperature_c": "WRF温度",
        "wrf_clearness_index": "WRF清晰度指数",
        "wrf_cloud_cover_ratio": "WRF云量比例",
    }

    if len(key_features) > 2:
        corr_matrix = df[key_features].corr()

        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

        ax.set_xticks(range(len(key_features)))
        ax.set_yticks(range(len(key_features)))
        ax.set_xticklabels([feature_label_map.get(f, f) for f in key_features], rotation=45, ha="right")
        ax.set_yticklabels([feature_label_map.get(f, f) for f in key_features])

        for i in range(len(key_features)):
            for j in range(len(key_features)):
                text = ax.text(j, i, f"{corr_matrix.values[i, j]:.2f}",
                             ha="center", va="center", fontsize=9,
                             color="white" if abs(corr_matrix.values[i, j]) > 0.5 else "black")

        plt.colorbar(im, ax=ax, label="相关系数")
        ax.set_title("Site 1 关键特征相关性热力图", fontsize=14)

        plt.tight_layout()
        fig.savefig(output_dir / "site1_feature_correlation.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"已保存: {output_dir / 'site1_feature_correlation.png'}")


def plot_site1_wrf_features(df: pd.DataFrame, output_dir: Path):
    """绘制WRF气象特征对比图"""
    output_dir.mkdir(parents=True, exist_ok=True)

    key_wrf = ["wrf_gti_wm2", "wrf_tsi_wm2", "wrf_temperature_c", "wrf_clearness_index", "wrf_cloud_cover_ratio"]

    # 中文标签映射
    wrf_label_map = {
        "wrf_gti_wm2": "WRF斜面总辐射 (W/m²)",
        "wrf_tsi_wm2": "WRF天文辐射 (W/m²)",
        "wrf_temperature_c": "WRF温度 (°C)",
        "wrf_clearness_index": "WRF清晰度指数",
        "wrf_cloud_cover_ratio": "WRF云量比例",
    }

    df_sample = df[(df["timestamp"].dt.year == 2020) & (df["timestamp"].dt.month.isin([6, 7, 8]))].copy()
    df_sample = df_sample.set_index("timestamp")

    fig, axes = plt.subplots(len(key_wrf), 1, figsize=(16, 12))

    for i, col in enumerate(key_wrf):
        if col in df_sample.columns:
            axes[i].plot(df_sample.index, df_sample[col], alpha=0.7, linewidth=0.5)
            axes[i].set_title(wrf_label_map.get(col, col), fontsize=10)
            axes[i].set_ylabel("值")
            axes[i].grid(True, alpha=0.3)
            axes[i].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
            axes[i].xaxis.set_major_locator(mdates.DayLocator(interval=7))

    plt.suptitle("Site 1 WRF气象特征 (2020年6-8月样本)", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "site1_wrf_features.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已保存: {output_dir / 'site1_wrf_features.png'}")


def plot_horizon_prediction_curves(horizons: list, output_dir: Path):
    """绘制H1/H4/H16预测曲线对比图"""
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_data = {}
    for h in horizons:
        pred_path = PRED_DIR / f"h{h}" / "cnn_bilstm_test.csv"
        if pred_path.exists():
            df = pd.read_csv(pred_path)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            pred_data[h] = df
        else:
            print(f"警告: 预测文件不存在 {pred_path}")

    if not pred_data:
        print("错误: 没有找到任何预测数据")
        return

    start_date = "2020-10-05 08:00:00"
    end_date = "2020-10-12 18:00:00"

    data_by_horizon = {}
    for h, df in pred_data.items():
        mask = (df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)
        df_subset = df[mask].copy()

        if h == 1:
            df_subset = df_subset.sort_values("timestamp").head(500)
        elif h == 4:
            df_subset["step"] = df_subset.groupby("timestamp").cumcount()
            df_subset = df_subset[df_subset["step"] == h - 1]
            df_subset = df_subset.sort_values("timestamp").head(500)
        elif h == 16:
            df_subset["step"] = df_subset.groupby("timestamp").cumcount()
            df_subset = df_subset[df_subset["step"] == h - 1]
            df_subset = df_subset.sort_values("timestamp").head(500)

        data_by_horizon[h] = df_subset

    fig, axes = plt.subplots(3, 1, figsize=(20, 18))

    horizon_colors = {1: "red", 4: "blue", 16: "green"}
    horizon_labels = {1: "H1 (15分钟预测)", 4: "H4 (1小时预测)", 16: "H16 (4小时预测)"}

    for idx, h in enumerate(horizons):
        if h not in data_by_horizon or len(data_by_horizon[h]) == 0:
            axes[idx].text(0.5, 0.5, f"H{h} 无数据", ha="center", va="center", transform=axes[idx].transAxes)
            continue

        df = data_by_horizon[h]

        axes[idx].plot(df["timestamp"], df["y_true"], "k-", linewidth=1.5, label="真实值", alpha=0.9)
        axes[idx].plot(df["timestamp"], df["y_pred"], "--", color=horizon_colors.get(h, "red"),
                       linewidth=1.5, label="CNN-BiLSTM预测", alpha=0.8)
        axes[idx].fill_between(df["timestamp"],
                               df["y_true"],
                               df["y_pred"],
                               alpha=0.2, color=horizon_colors.get(h, "red"),
                               label="预测误差")

        axes[idx].set_title(f"预测步 {horizon_labels[h]}", fontsize=14, fontweight="bold")
        axes[idx].set_xlabel("时间 (精确到分钟)", fontsize=11)
        axes[idx].set_ylabel("功率 (p.u.)", fontsize=11)
        axes[idx].legend(loc="upper right", fontsize=10)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_ylim(-0.05, 1.1)

        axes[idx].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        axes[idx].xaxis.set_major_locator(mdates.DayLocator())
        plt.setp(axes[idx].xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.suptitle("Site 1 CNN-BiLSTM 预测曲线对比\n验证集时间范围: 2020年10月5日-12日",
                 fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    output_path = output_dir / "site1_h1_h4_h16_prediction_curves.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已保存: {output_path}")


def plot_horizon_monthly_comparison(horizons: list, output_dir: Path):
    """为9、10、11、12月各选取同一段时间，绘制3个H的预测对比长图"""
    output_dir.mkdir(parents=True, exist_ok=True)

    months = [
        ("09", "九月"),
        ("10", "十月"),
        ("11", "十一月"),
        ("12", "十二月")
    ]

    day_ranges = {
        "09": ("2020-09-15 08:00:00", "2020-09-22 18:00:00"),
        "10": ("2020-10-15 08:00:00", "2020-10-22 18:00:00"),
        "11": ("2020-11-15 08:00:00", "2020-11-22 18:00:00"),
        "12": ("2020-12-15 08:00:00", "2020-12-22 18:00:00")
    }

    horizon_colors = {1: "red", 4: "blue", 16: "green"}
    horizon_labels = {1: "H1 (15分钟预测)", 4: "H4 (1小时预测)", 16: "H16 (4小时预测)"}

    for month_key, month_label in months:
        start_date, end_date = day_ranges[month_key]

        # 创建3行1列的长图
        fig, axes = plt.subplots(3, 1, figsize=(20, 18))

        for h_idx, h in enumerate(horizons):
            pred_path = PRED_DIR / f"h{h}" / "cnn_bilstm_test.csv"

            if not pred_path.exists():
                axes[h_idx].text(0.5, 0.5, f"H{h} 无数据", ha="center", va="center",
                                transform=axes[h_idx].transAxes)
                continue

            df = pd.read_csv(pred_path)
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            mask = (df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)
            df_subset = df[mask].copy()

            if len(df_subset) == 0:
                axes[h_idx].text(0.5, 0.5, f"H{h} 无{month_label}数据", ha="center", va="center",
                                transform=axes[h_idx].transAxes)
                continue

            if h > 1:
                df_subset["step"] = df_subset.groupby("timestamp").cumcount()
                df_subset = df_subset[df_subset["step"] == h - 1]

            df_subset = df_subset.sort_values("timestamp").reset_index(drop=True)

            # 限制数据点数量以保持可读性
            if len(df_subset) > 600:
                df_subset = df_subset.iloc[::2].reset_index(drop=True)

            # 真实值
            axes[h_idx].plot(df_subset["timestamp"], df_subset["y_true"], "k-",
                           linewidth=1.5, label="真实值", alpha=0.9)
            # 预测值
            axes[h_idx].plot(df_subset["timestamp"], df_subset["y_pred"], "--",
                           color=horizon_colors[h], linewidth=1.5,
                           label="CNN-BiLSTM预测", alpha=0.8)
            # 填充误差
            axes[h_idx].fill_between(df_subset["timestamp"],
                           df_subset["y_true"],
                           df_subset["y_pred"],
                           alpha=0.2, color=horizon_colors[h],
                           label="预测误差")

            axes[h_idx].set_title(f"预测步 {horizon_labels[h]}", fontsize=14, fontweight="bold")
            axes[h_idx].set_xlabel("时间 (精确到分钟)", fontsize=11)
            axes[h_idx].set_ylabel("功率 (p.u.)", fontsize=11)
            axes[h_idx].legend(loc="upper right", fontsize=10)
            axes[h_idx].grid(True, alpha=0.3)
            axes[h_idx].set_ylim(-0.05, 1.1)

            axes[h_idx].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
            axes[h_idx].xaxis.set_major_locator(mdates.DayLocator())
            plt.setp(axes[h_idx].xaxis.get_majorticklabels(), rotation=45, ha="right")

        plt.suptitle(f"Site 1 CNN-BiLSTM 预测曲线 - {month_label} 2020年\n"
                    f"时间范围: {start_date[:10]} 至 {end_date[:10]}",
                    fontsize=16, fontweight="bold", y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.98])

        output_path = output_dir / f"site1_predictions_{month_key}_month.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"已保存: {output_path}")


def plot_all_horizons_single_figure(horizons: list, output_dir: Path):
    """创建一张总图：3行，每行一个H，展示9-12月的数据"""
    output_dir.mkdir(parents=True, exist_ok=True)

    months = [
        ("09", "九月", "2020-09-15 08:00:00", "2020-09-22 18:00:00"),
        ("10", "十月", "2020-10-15 08:00:00", "2020-10-22 18:00:00"),
        ("11", "十一月", "2020-11-15 08:00:00", "2020-11-22 18:00:00"),
        ("12", "十二月", "2020-12-15 08:00:00", "2020-12-22 18:00:00")
    ]

    horizon_colors = {1: "#E41A1C", 4: "#377EB8", 16: "#4DAF4A"}

    fig, axes = plt.subplots(3, 4, figsize=(24, 15), sharey="row")
    plt.subplots_adjust(hspace=0.4, wspace=0.15)

    for h_idx, h in enumerate(horizons):
        pred_path = PRED_DIR / f"h{h}" / "cnn_bilstm_test.csv"

        if not pred_path.exists():
            for m_idx in range(4):
                axes[h_idx, m_idx].text(0.5, 0.5, f"H{h} 无数据", ha="center", va="center")
            continue

        df = pd.read_csv(pred_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        for m_idx, (month_key, month_label, start_date, end_date) in enumerate(months):
            mask = (df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)
            df_subset = df[mask].copy()

            if len(df_subset) == 0:
                axes[h_idx, m_idx].text(0.5, 0.5, "无数据", ha="center", va="center")
                continue

            if h > 1:
                df_subset["step"] = df_subset.groupby("timestamp").cumcount()
                df_subset = df_subset[df_subset["step"] == h - 1]

            df_subset = df_subset.sort_values("timestamp").reset_index(drop=True)

            if len(df_subset) > 400:
                df_subset = df_subset.iloc[::2].reset_index(drop=True)

            ax = axes[h_idx, m_idx]

            ax.plot(df_subset["timestamp"], df_subset["y_true"], "k-",
                   linewidth=1.2, label="真实值", alpha=0.9)
            ax.plot(df_subset["timestamp"], df_subset["y_pred"], "--",
                   color=horizon_colors[h], linewidth=1.2,
                   label=f"H{h}", alpha=0.8)
            ax.fill_between(df_subset["timestamp"],
                           df_subset["y_true"],
                           df_subset["y_pred"],
                           alpha=0.15, color=horizon_colors[h])

            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.05, 1.1)

            if m_idx == 0:
                ax.set_ylabel("功率 (p.u.)", fontsize=10)

            if h_idx == 2:
                ax.set_xlabel("时间", fontsize=9)
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)
            else:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)

            if h_idx == 0:
                ax.set_title(month_label, fontsize=12, fontweight="bold")

    for h_idx, h in enumerate(horizons):
        axes[h_idx, 0].annotate(f"H{h}",
                               xy=(-0.15, 0.5), xycoords="axes fraction",
                               fontsize=14, fontweight="bold",
                               rotation=90, va="center", ha="center")

    plt.suptitle("Site 1 CNN-BiLSTM 预测曲线对比 (H1/H4/H16)\n"
                "真实值(黑色实线) vs 预测值(彩色虚线)",
                fontsize=16, fontweight="bold", y=0.98)

    output_path = output_dir / "site1_h1_h4_h16_all_months.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已保存: {output_path}")


def plot_quarterly_prediction_curves(horizons: list, output_dir: Path):
    """按季度生成预测曲线图，每张图包含3行(H1/H4/H16)，横跨3个月"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 定义4个季度，每个季度横跨3个月
    # 注：测试集时间范围为 2020-09-13 至 2020-12-31
    quarters = [
        {
            "name": "summer_autumn",
            "title": "夏秋过渡期 (7-9月)",
            "months": [
                ("07", "七月", "2020-07-15 08:00:00", "2020-07-22 18:00:00"),
                ("08", "八月", "2020-08-15 08:00:00", "2020-08-22 18:00:00"),
                ("09", "九月", "2020-09-15 08:00:00", "2020-09-22 18:00:00"),
            ]
        },
        {
            "name": "autumn",
            "title": "秋季 (9-11月)",
            "months": [
                ("09", "九月", "2020-09-15 08:00:00", "2020-09-22 18:00:00"),
                ("10", "十月", "2020-10-15 08:00:00", "2020-10-22 18:00:00"),
                ("11", "十一月", "2020-11-15 08:00:00", "2020-11-22 18:00:00"),
            ]
        },
        {
            "name": "early_winter",
            "title": "初冬 (11月-12月)",
            "months": [
                ("11", "十一月", "2020-11-15 08:00:00", "2020-11-22 18:00:00"),
                ("12", "十二月", "2020-12-15 08:00:00", "2020-12-22 18:00:00"),
                ("01", "一月", "2021-01-15 08:00:00", "2021-01-22 18:00:00"),
            ]
        },
        {
            "name": "full_test",
            "title": "完整测试集 (9-12月)",
            "months": [
                ("09", "九月", "2020-09-15 08:00:00", "2020-09-22 18:00:00"),
                ("10", "十月", "2020-10-15 08:00:00", "2020-10-22 18:00:00"),
                ("11", "十一月", "2020-11-15 08:00:00", "2020-11-22 18:00:00"),
            ]
        },
    ]

    horizon_colors = {1: "#E41A1C", 4: "#377EB8", 16: "#4DAF4A"}
    horizon_labels = {1: "H1 (15分钟)", 4: "H4 (1小时)", 16: "H16 (4小时)"}

    for quarter in quarters:
        fig, axes = plt.subplots(3, 3, figsize=(24, 15), sharey="row")
        plt.subplots_adjust(hspace=0.35, wspace=0.15)

        for h_idx, h in enumerate(horizons):
            pred_path = PRED_DIR / f"h{h}" / "cnn_bilstm_test.csv"

            if not pred_path.exists():
                for m_idx in range(3):
                    axes[h_idx, m_idx].text(0.5, 0.5, f"H{h} 无数据", ha="center", va="center")
                continue

            df = pd.read_csv(pred_path)
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            for m_idx, (month_key, month_label, start_date, end_date) in enumerate(quarter["months"]):
                mask = (df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)
                df_subset = df[mask].copy()

                if len(df_subset) == 0:
                    axes[h_idx, m_idx].text(0.5, 0.5, "无数据", ha="center", va="center")
                    continue

                if h > 1:
                    df_subset["step"] = df_subset.groupby("timestamp").cumcount()
                    df_subset = df_subset[df_subset["step"] == h - 1]

                df_subset = df_subset.sort_values("timestamp").reset_index(drop=True)

                if len(df_subset) > 400:
                    df_subset = df_subset.iloc[::2].reset_index(drop=True)

                ax = axes[h_idx, m_idx]

                ax.plot(df_subset["timestamp"], df_subset["y_true"], "k-",
                       linewidth=1.2, label="真实值", alpha=0.9)
                ax.plot(df_subset["timestamp"], df_subset["y_pred"], "--",
                       color=horizon_colors[h], linewidth=1.2,
                       label=f"H{h}", alpha=0.8)
                ax.fill_between(df_subset["timestamp"],
                               df_subset["y_true"],
                               df_subset["y_pred"],
                               alpha=0.15, color=horizon_colors[h])

                ax.grid(True, alpha=0.3)
                ax.set_ylim(-0.05, 1.1)

                if m_idx == 0:
                    ax.set_ylabel("功率 (p.u.)", fontsize=10)

                if h_idx == 2:
                    ax.set_xlabel("时间", fontsize=9)
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
                    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)
                else:
                    ax.set_xlabel("")
                    ax.tick_params(labelbottom=False)

                if h_idx == 0:
                    ax.set_title(f"{month_label}\n{start_date[:10]} 至 {end_date[:10]}",
                                fontsize=11, fontweight="bold")

        # 左侧标注H
        for h_idx, h in enumerate(horizons):
            axes[h_idx, 0].annotate(horizon_labels[h],
                                   xy=(-0.25, 0.5), xycoords="axes fraction",
                                   fontsize=13, fontweight="bold",
                                   rotation=90, va="center", ha="center")

        plt.suptitle(f"Site 1 CNN-BiLSTM 预测曲线 - {quarter['title']}\n"
                    f"真实值(黑色实线) vs 预测值(彩色虚线)",
                    fontsize=16, fontweight="bold", y=0.98)

        output_path = output_dir / f"site1_quarterly_predictions_{quarter['name'].lower()}.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Site1数据预处理分析与预测曲线可视化")
    parser.add_argument("--analysis", action="store_true", help="仅生成Site1数据分析图")
    parser.add_argument("--predictions", action="store_true", help="仅生成预测曲线图")
    parser.add_argument("--all", action="store_true", help="生成所有图表")
    parser.add_argument("--quarterly", action="store_true", help="生成季度预测图")
    args = parser.parse_args()

    generate_analysis = args.all or (not args.predictions)
    generate_predictions = args.all or args.predictions

    figures_output = FIGURES_DIR / "site1_analysis"
    figures_output.mkdir(parents=True, exist_ok=True)

    if generate_analysis:
        print("=" * 60)
        print("生成 Site1 数据预处理分析图表...")
        print("=" * 60)

        df = load_site1_data()
        print(f"加载数据: {len(df)} 条记录")
        print(f"时间范围: {df['timestamp'].min()} 至 {df['timestamp'].max()}")

        plot_site1_data_overview(df, figures_output)
        plot_site1_monthly_pattern(df, figures_output)
        plot_site1_data_quality(df, figures_output)
        plot_site1_feature_correlation(df, figures_output)
        plot_site1_wrf_features(df, figures_output)

    if generate_predictions:
        print("\n" + "=" * 60)
        print("生成 H1/H4/H16 预测曲线对比图...")
        print("=" * 60)

        horizons = [1, 4, 16]

        plot_horizon_prediction_curves(horizons, figures_output)
        plot_horizon_monthly_comparison(horizons, figures_output)
        plot_all_horizons_single_figure(horizons, figures_output)

    if args.quarterly:
        print("\n" + "=" * 60)
        print("生成季度预测曲线图 (每季度一张)... ")
        print("=" * 60)

        horizons = [1, 4, 16]
        plot_quarterly_prediction_curves(horizons, figures_output)

    print("\n" + "=" * 60)
    print("所有图表已生成!")
    print(f"输出目录: {figures_output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
