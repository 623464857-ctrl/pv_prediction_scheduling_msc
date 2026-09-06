"""
明月湖数据集 - 数据质量诊断报告 (v2)
=====================================
装机容量: 281.6 kW
DNI 字段已移除（保留 GHI 和 DHI）
白天功率为0 = 真实停机，不修正
"""

import pandas as pd
import numpy as np
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from pathlib import Path

# ===================== 配置 =====================
RAW_DIR = Path(r"c:\Users\MoYu\Desktop\pv_prediction_scheduling_msc_new\data\raw")
DATA_DIR = Path(r"c:\Users\MoYu\Desktop\pv_prediction_scheduling_msc_new\data\data-analysis")
POWER_FILE = RAW_DIR / "明月湖光伏发电.csv"
WEATHER_FILE = RAW_DIR / "明月湖6-8月天气数据.csv"
OUTPUT_DATA = DATA_DIR / "data" / "明月湖_merged_raw.csv"
OUTPUT_REPORT = DATA_DIR / "reports" / "data_quality_report.txt"

CAPACITY_KW = 281.6  # 装机容量 kW

# ===================== 1. 加载数据 =====================
print("=" * 60)
print("1. 加载数据")
print("=" * 60)

power = pd.read_csv(POWER_FILE)
weather = pd.read_csv(WEATHER_FILE)

print(f"[功率] 行数: {len(power)}, 列数: {len(power.columns)}")
print(f"[天气] 行数: {len(weather)}, 列数: {len(weather.columns)}")

# ===================== 2. 合并 =====================
print("\n" + "=" * 60)
print("2. 合并数据集（时间对齐）")
print("=" * 60)

power["时间"] = pd.to_datetime(power["时间"])
weather["timestamp_local"] = pd.to_datetime(weather["timestamp_local"])

merged = pd.merge(
    power,
    weather,
    left_on="时间",
    right_on="timestamp_local",
    how="outer"
)

print(f"[合并] 行数: {len(merged)}, 时间范围: {merged['时间'].min()} ~ {merged['时间'].max()}")

# ===================== 3. 时间连续性 =====================
print("\n" + "=" * 60)
print("3. 时间连续性检查 (15分钟间隔)")
print("=" * 60)

time_col = merged["时间"].dropna().sort_values()
time_diffs = time_col.diff().dropna()
expected_delta = pd.Timedelta(minutes=15)
non_standard_gaps = time_diffs[time_diffs != expected_delta]
full_range = pd.date_range(start=time_col.min(), end=time_col.max(), freq="15min")
missing_timestamps = full_range.difference(time_col)
dup_times = merged["时间"][merged["时间"].duplicated(keep=False)]

print(f"理论行数: {len(full_range)}, 实际: {len(time_col)}, 缺失: {len(missing_timestamps)}")
print(f"非标准间隔: {len(non_standard_gaps)}, 重复时间戳: {len(dup_times)}")
if len(non_standard_gaps) > 0:
    print(f"  非标准间隔详情: {non_standard_gaps.value_counts().to_dict()}")

# ===================== 4. 缺失值 =====================
print("\n" + "=" * 60)
print("4. 缺失值检查")
print("=" * 60)

numeric_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
missing_summary = merged[numeric_cols].isnull().sum()
missing_pct = (missing_summary / len(merged) * 100).round(2)
missing_report = pd.DataFrame({"缺失数量": missing_summary, "缺失比例(%)": missing_pct})
missing_report = missing_report[missing_report["缺失数量"] > 0]

if len(missing_report) == 0:
    print("所有数值列均无缺失值")
else:
    print(missing_report.to_string())

# ===================== 5. 功率数据质量检查 =====================
print("\n" + "=" * 60)
print("5. 功率数据质量检查")
print("=" * 60)

power_col = "光伏发电功率"

# 5a. 负值
neg_power = merged[merged[power_col] < 0]
print(f"负功率记录: {len(neg_power)}")

# 5b. 超出装机容量
over_cap = merged[merged[power_col] > CAPACITY_KW]
print(f"超容量记录 (>{CAPACITY_KW} kW): {len(over_cap)}")
if len(over_cap) > 0:
    print(f"  超容量最大值: {over_cap[power_col].max():.2f} kW")
    print(f"  超容量比例: {len(over_cap)/len(merged)*100:.2f}%")

# 5c. 夜间功率>0（夜间微量，Clip为0）
night_power = merged[(merged["ghi"] == 0) & (merged[power_col] > 0)]
print(f"\n夜间(ghi=0)功率>0的记录: {len(night_power)}")
if len(night_power) > 0:
    print(f"  夜间最大功率: {night_power[power_col].max():.2f} kW")
    print(f"  夜间功率分布: mean={night_power[power_col].mean():.4f}, std={night_power[power_col].std():.4f}")

# 5d. 白天功率=0（真实停机，不修正）
daytime = merged[(merged["ghi"] > 10) & (merged["solar_alt"] > 0)]
zero_power_day = daytime[daytime[power_col] == 0]
print(f"\n白天(辐照>10 & alt>0)功率=0记录: {len(zero_power_day)}")
print(f"  占比(白天总记录): {len(zero_power_day)/len(daytime)*100:.2f}%")
print(f"  (白天功率为0 = 真实停机，保持原样)")

# 5e. 功率统计
print(f"\n功率统计:")
print(merged[power_col].describe().round(4).to_string())

# ===================== 6. 辐照度检查 =====================
print("\n" + "=" * 60)
print("6. 辐照度数据检查 (GHI / DHI)")
print("=" * 60)

# 6a. DHI > GHI
dhi_gt_ghi = merged[merged["dhi"] > merged["ghi"]]
print(f"DHI > GHI 的记录: {len(dhi_gt_ghi)}")
if len(dhi_gt_ghi) > 0:
    print(f"  占比: {len(dhi_gt_ghi)/len(merged)*100:.2f}%")
    print(f"  超出量: max={(dhi_gt_ghi['dhi'] - dhi_gt_ghi['ghi']).max():.1f} W/m²")
    print(f"  示例:\n{dhi_gt_ghi[['时间', 'ghi', 'dhi', 'solar_alt']].head(5).to_string()}")

# 6b. GHI 极值
ghi_extreme = merged[(merged["ghi"] > 1100) & merged["ghi"].notnull()]
print(f"\nGHI > 1100 W/m² 极值记录: {len(ghi_extreme)}")
if len(ghi_extreme) > 0:
    print(f"  最大值: {ghi_extreme['ghi'].max():.1f} W/m²")

# 6c. GHI 物理一致性（用 GHI = DHI + GHI_direct，重排：GHI_direct = GHI - DHI，应 ≥ 0）
ghi_direct = merged["ghi"] - merged["dhi"]
negative_direct = ghi_direct[ghi_direct < -10]  # 容许10误差
print(f"\nGHI直接辐照分量 (GHI - DHI < -10) 的记录: {len(negative_direct)}")
if len(negative_direct) > 0:
    print(f"  最小值: {negative_direct.min():.1f} W/m²")

# 6d. DHI 极值
dhi_extreme = merged[(merged["dhi"] > 300) & merged["dhi"].notnull()]
print(f"\nDHI > 300 W/m² 记录: {len(dhi_extreme)}")
if len(dhi_extreme) > 0:
    print(f"  最大值: {dhi_extreme['dhi'].max():.1f} W/m²")

# 6e. 辐照统计
print(f"\nGHI / DHI 统计:")
print(merged[["ghi", "dhi"]].describe().round(2).to_string())

# ===================== 7. 辐照-功率一致性 =====================
print("\n" + "=" * 60)
print("7. 辐照-功率 物理一致性检查")
print("=" * 60)

# 白天高辐照低功率（除开停机时刻，即功率>0但明显偏低）
day_normal = merged[(merged["solar_alt"] > 20) & (merged["ghi"] > 500) & (merged[power_col] > 0)]
print(f"白天(alt>20, ghi>500, power>0)有效发电记录: {len(day_normal)}")
if len(day_normal) > 0:
    # 计算转换效率：power_kW / (ghi * area_estimate)
    # 粗估：281.6kW / 1000 W/m2 = 281.6 m2 面板面积（合理）
    area_est = CAPACITY_KW / 1000.0  # m2，假设1000 W/m2标准辐照下的满发面积
    efficiency = day_normal[power_col] / (day_normal["ghi"] * area_est) * 100
    print(f"  转换效率分布(%): mean={efficiency.mean():.1f}, median={efficiency.median():.1f}, max={efficiency.max():.1f}")
    low_eff = efficiency[efficiency < 5]
    print(f"  效率<5%的记录: {len(low_eff)}")

# ===================== 8. 气象字段统计 =====================
print("\n" + "=" * 60)
print("8. 气象字段统计")
print("=" * 60)

key_cols = ["temp", "rh", "pres", "wind_spd", "clouds", "ghi", "dhi", "solar_alt"]
print(merged[key_cols].describe().round(2).to_string())

# 时段分布
merged["hour"] = merged["时间"].dt.hour
day_hours = merged[(merged["hour"] >= 6) & (merged["hour"] <= 18)]
night_hours = merged[(merged["hour"] < 6) | (merged["hour"] > 18)]
print(f"\n白天(6-18时): {len(day_hours)} 条 ({len(day_hours)/len(merged)*100:.1f}%)")
print(f"夜间(<6时或>18时): {len(night_hours)} 条 ({len(night_hours)/len(merged)*100:.1f}%)")

print(f"\n天气类型分布:")
weather_dist = merged["weather_desc"].value_counts()
for desc, count in weather_dist.items():
    print(f"  {desc}: {count} ({count/len(merged)*100:.1f}%)")

print(f"\n云量分布:")
cloud_bins = pd.cut(merged["clouds"], bins=[0, 10, 30, 60, 80, 100],
                     labels=["晴(0-10%)", "少云(10-30%)", "多云(30-60%)", "阴(60-80%)", "浓阴(80-100%)"], right=True)
print(cloud_bins.value_counts().sort_index().to_string())

# ===================== 9. 保存合并结果 =====================
print("\n" + "=" * 60)
print("9. 保存合并数据集")
print("=" * 60)

# 移除冗余列和 DNI，整理列顺序
cols_to_drop = ["timestamp_local", "timestamp_utc", "ts", "dni"]
cols_to_drop = [c for c in cols_to_drop if c in merged.columns]
merged_clean = merged.drop(columns=cols_to_drop)

merged_clean = merged_clean.sort_values("时间").reset_index(drop=True)

# 重命名列
merged_clean = merged_clean.rename(columns={
    "时间": "timestamp",
    "光伏发电功率": "power_kw",
    "temp": "temperature_c",
    "app_temp": "apparent_temperature_c",
    "rh": "relative_humidity_pct",
    "dewpt": "dew_point_c",
    "pres": "pressure_hpa",
    "wind_spd": "wind_speed_ms",
    "wind_dir": "wind_dir_deg",
    "wind_gust_spd": "wind_gust_ms",
    "vis": "visibility_km",
    "clouds": "cloud_cover_pct",
    "solar_alt": "solar_altitude_deg",
    "solar_az": "solar_azimuth_deg",
    "solar_rad": "solar_radiation_wm2",
    "precip_rate": "precip_rate_mmhr",
    "snow_rate": "snow_rate_mmhr",
    "pod": "part_of_day",
    "weather_code": "weather_code",
    "weather_desc": "weather_description",
    "weather_icon": "weather_icon",
    "hour": "hour_of_day"
})

merged_clean.to_csv(OUTPUT_DATA, index=False, encoding="utf-8-sig")
print(f"已保存数据: {OUTPUT_DATA}")
print(f"行数: {len(merged_clean)}, 列数: {len(merged_clean.columns)}")
print(f"列名: {list(merged_clean.columns)}")

# ===================== 10. 问题汇总 =====================
print("\n" + "=" * 60)
print("10. 数据问题汇总")
print("=" * 60)

issues = []

if len(non_standard_gaps) > 0:
    issues.append(f"[时间] 非标准间隔: {len(non_standard_gaps)} 处")
if len(missing_timestamps) > 0:
    issues.append(f"[时间] 缺失时间点: {len(missing_timestamps)} 个")
if len(dup_times) > 0:
    issues.append(f"[时间] 重复时间戳: {len(dup_times)} 个")
if len(neg_power) > 0:
    issues.append(f"[功率] 负功率记录: {len(neg_power)} 条")
if len(over_cap) > 0:
    issues.append(f"[功率] 超装机容量({CAPACITY_KW}kW): {len(over_cap)} 条 ({len(over_cap)/len(merged)*100:.1f}%)")
if len(night_power) > 0:
    issues.append(f"[功率] 夜间功率>0: {len(night_power)} 条 (建议Clip为0)")
if len(dhi_gt_ghi) > 0:
    issues.append(f"[辐照] DHI>GHI: {len(dhi_gt_ghi)} 条 ({len(dhi_gt_ghi)/len(merged)*100:.1f}%)")

if issues:
    for issue in issues:
        print(issue)
else:
    print("初步检查未发现明显数据问题")

print("\n诊断完成!")

# ===================== 11. 生成文本报告文件 =====================
import datetime

lines = []
lines.append("=" * 60)
lines.append("明月湖数据集 - 数据质量诊断报告")
lines.append(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
lines.append(f"数据时间范围: {merged['时间'].min()} ~ {merged['时间'].max()}")
lines.append("=" * 60)
lines.append("")
lines.append("【一、数据概览】")
lines.append(f"  原始功率文件: 明月湖光伏发电.csv ({len(power)} 行 x {len(power.columns)} 列)")
lines.append(f"  原始天气文件: 明月湖6-8月天气数据.csv ({len(weather)} 行 x {len(weather.columns)} 列)")
lines.append(f"  合并后文件:   明月湖_merged_raw.csv ({len(merged_clean)} 行 x {len(merged_clean.columns)} 列)")
lines.append(f"  时间分辨率:   15 分钟")
lines.append(f"  装机容量:     {CAPACITY_KW} kW")
lines.append("")
lines.append("【二、时间连续性】")
lines.append(f"  理论行数: {len(full_range)}, 实际: {len(time_col)}, 缺失: {len(missing_timestamps)}")
lines.append(f"  非标准间隔: {len(non_standard_gaps)}, 重复时间戳: {len(dup_times)}")
lines.append(f"  结论: {'通过' if len(non_standard_gaps)==0 and len(missing_timestamps)==0 and len(dup_times)==0 else '存在问题'}")
lines.append("")
lines.append("【三、缺失值】")
lines.append(f"  数值列缺失: {'无' if len(missing_report)==0 else str(len(missing_report))+' 列有缺失'}")
lines.append("")
lines.append("【四、功率数据】")
lines.append(f"  负功率记录: {len(neg_power)}")
lines.append(f"  超容量(>{CAPACITY_KW}kW): {len(over_cap)} 条")
lines.append(f"  夜间功率>0: {len(night_power)} 条 (建议Clip为0)")
lines.append(f"  白天功率=0(停机): {len(zero_power_day)} 条 (保持原样)")
lines.append(f"  功率范围: {merged[power_col].min():.4f} ~ {merged[power_col].max():.4f} kW")
lines.append(f"  功率均值: {merged[power_col].mean():.4f} kW, 中位数: {merged[power_col].median():.4f} kW")
lines.append("")
lines.append("【五、辐照度数据(GHI/DHI)】")
lines.append(f"  DHI > GHI 记录: {len(dhi_gt_ghi)} 条 ({len(dhi_gt_ghi)/len(merged)*100:.2f}%)")
lines.append(f"  GHI > 1100 W/m2 极值: {len(ghi_extreme)} 条")
lines.append(f"  GHI 直接分量(GHI-DHI)<-10: {len(negative_direct)} 条")
lines.append(f"  GHI 范围: {merged['ghi'].min():.1f} ~ {merged['ghi'].max():.1f} W/m2")
lines.append(f"  DHI 范围: {merged['dhi'].min():.1f} ~ {merged['dhi'].max():.1f} W/m2")
lines.append("")
lines.append("【六、辐照-功率一致性】")
lines.append(f"  白天正常发电记录: {len(day_normal)} 条")
if len(day_normal) > 0:
    lines.append(f"  转换效率均值: {efficiency.mean():.1f}%, 中位数: {efficiency.median():.1f}%")
    lines.append(f"  效率<5%的记录: {len(low_eff)} 条")
lines.append("")
lines.append("【七、气象特征统计】")
lines.append(f"  温度: {merged['temp'].min():.1f} ~ {merged['temp'].max():.1f} °C, 均值: {merged['temp'].mean():.1f} °C")
lines.append(f"  湿度: {merged['rh'].min():.0f} ~ {merged['rh'].max():.0f} %, 均值: {merged['rh'].mean():.1f} %")
lines.append(f"  气压: {merged['pres'].min():.0f} ~ {merged['pres'].max():.0f} hPa, 均值: {merged['pres'].mean():.1f} hPa")
lines.append(f"  风速: {merged['wind_spd'].min():.1f} ~ {merged['wind_spd'].max():.1f} m/s, 均值: {merged['wind_spd'].mean():.1f} m/s")
lines.append(f"  云量: {merged['clouds'].min():.0f} ~ {merged['clouds'].max():.0f} %, 均值: {merged['clouds'].mean():.1f} %")
lines.append(f"  白天时段(6-18时): {len(day_hours)} 条 ({len(day_hours)/len(merged)*100:.1f}%)")
lines.append(f"  夜间时段: {len(night_hours)} 条 ({len(night_hours)/len(merged)*100:.1f}%)")
lines.append("")
lines.append("【八、天气类型分布】")
for desc, count in weather_dist.items():
    lines.append(f"  {desc}: {count} 条 ({count/len(merged)*100:.1f}%)")
lines.append("")
lines.append("【九、待处理问题】")
lines.append(f"  1. 夜间功率>0 ({len(night_power)}条): 建议Clip为0")
lines.append(f"  2. DHI>GHI ({len(dhi_gt_ghi)}条): 超出量极小(最大6 W/m2), 可忽略或Clip DHI<=GHI")
lines.append("")
lines.append("【十、合并数据集字段说明】")
field_desc = {
    "timestamp": "时间戳",
    "power_kw": "光伏发电功率 (kW)",
    "temperature_c": "温度 (°C)",
    "apparent_temperature_c": "体感温度 (°C)",
    "relative_humidity_pct": "相对湿度 (%)",
    "dew_point_c": "露点温度 (°C)",
    "pressure_hpa": "气压 (hPa)",
    "wind_speed_ms": "风速 (m/s)",
    "wind_dir_deg": "风向 (度)",
    "wind_gust_ms": "阵风风速 (m/s)",
    "visibility_km": "能见度 (km)",
    "cloud_cover_pct": "云量 (%)",
    "solar_altitude_deg": "太阳高度角 (度)",
    "solar_azimuth_deg": "太阳方位角 (度)",
    "ghi": "水平面总辐射 (W/m2)",
    "dhi": "水平面散射辐射 (W/m2)",
    "solar_radiation_wm2": "太阳辐射强度 (W/m2)",
    "uv": "紫外线指数",
    "precip_rate_mmhr": "降水率 (mm/h)",
    "snow_rate_mmhr": "降雪率 (mm/h)",
    "part_of_day": "时段(n=夜间/d=白天)",
    "weather_code": "天气代码",
    "weather_description": "天气描述",
    "weather_icon": "天气图标",
    "hour_of_day": "小时(0-23)",
}
for k, v in field_desc.items():
    lines.append(f"  {k}: {v}")

report_content = "\n".join(lines)
with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
    f.write(report_content)
print(f"报告已保存: {OUTPUT_REPORT}")
