"""
Weatherbit.io API - 全量天气参数数据获取
获取指定时间范围内的所有可用天气参数，输出为 CSV

API 文档: https://www.weatherbit.io/api/history-subhourly
"""

import requests
import pandas as pd
import time
from pathlib import Path
from datetime import datetime, timedelta

# ==================== 配置区 ====================

API_KEY = "6ab7f37c60e94aa8af1277feb03b1b96"
LAT = 22.816312
LON = 108.285408

# 数据时间范围
START_DATE = "2026-06-09"
END_DATE   = "2026-09-01"

# 输出配置
OUTPUT_DIR  = Path(r"C:\Users\MoYu\Desktop\pv_prediction_scheduling_msc_new\data\raw")
OUTPUT_FILE = OUTPUT_DIR / "明月湖6-8月天气数据.csv"

# 请求间隔（秒），避免触发限流
REQUEST_DELAY = 3

# 最多重试次数
MAX_RETRIES = 3

# ==================== 工具函数 ====================

def get_month_ranges(start_date: str, end_date: str) -> list:
    """
    生成按月分段的请求日期范围列表。
    Weatherbit 每次请求不超过约 31 天，按月分段最稳妥。
    """
    ranges = []
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    while current <= end:
        month_start = current.replace(day=1)
        if month_start.month == 12:
            month_end = month_start.replace(
                year=month_start.year + 1, month=1, day=1
            ) - timedelta(days=1)
        else:
            month_end = month_start.replace(
                month=month_start.month + 1, day=1
            ) - timedelta(days=1)

        req_start = current.strftime("%Y-%m-%d")
        req_end   = min(month_end, end).strftime("%Y-%m-%d")
        ranges.append((req_start, req_end))

        # 移到下个月1号
        if month_start.month == 12:
            current = month_start.replace(year=month_start.year + 1, month=1)
        else:
            current = month_start.replace(month=month_start.month + 1)

    return ranges


def fetch_weatherbit_batch(start_date: str, end_date: str) -> list:
    """
    单批次请求 Weatherbit subhourly API，返回 records 列表。
    """
    url = (
        "https://api.weatherbit.io/v2.0/history/subhourly"
        f"?start_date={start_date}&end_date={end_date}"
        f"&lat={LAT}&lon={LON}"
        f"&key={API_KEY}"
    )

    for attempt in range(MAX_RETRIES):
        try:
            print(f"  [请求] {start_date} ~ {end_date} (尝试 {attempt + 1}/{MAX_RETRIES}) ...")
            resp = requests.get(url, timeout=120)

            if resp.status_code == 200:
                data = resp.json()
                records = data.get("data", [])
                city = data.get("city_name", "Unknown")
                print(f"  [成功] {city}，获取 {len(records)} 条记录")
                return records

            elif resp.status_code == 429:
                print(f"  [限流] 等待 90 秒后重试 ...")
                time.sleep(90)

            elif resp.status_code == 400:
                print(f"  [错误] 400: 日期范围可能超出 API 支持范围")
                return []

            else:
                print(f"  [错误] HTTP {resp.status_code}: {resp.text[:200]}")
                time.sleep(10)

        except requests.exceptions.Timeout:
            print(f"  [超时] 等待 15 秒 ...")
            time.sleep(15)

        except requests.exceptions.RequestException as e:
            print(f"  [网络错误] {e}，等待 5 秒 ...")
            time.sleep(5)

    print(f"  [失败] {start_date} ~ {end_date} 请求失败")
    return []


def flatten_record(record: dict) -> dict:
    """
    将 Weatherbit 单条 subhourly 记录展平为单层 dict，
    包含所有可用字段。
    """
    weather = record.get("weather", {})
    return {
        # ── 时间 ──────────────────────────────────────────────
        "timestamp_local": record.get("timestamp_local", ""),
        "timestamp_utc":   record.get("timestamp_utc",   ""),
        "ts":              record.get("ts",              None),   # Unix 秒

        # ── 温度 / 体感温度 ────────────────────────────────────
        "temp":     record.get("temp",     None),   # 气温 (°C)
        "app_temp": record.get("app_temp", None),   # 体感温度 (°C)

        # ── 湿度 / 露点 / 水汽压 ────────────────────────────────
        "rh":   record.get("rh",   None),   # 相对湿度 (%)
        "dewpt": record.get("dewpt", None), # 露点温度 (°C)
        "pres": record.get("pres", None),   # 气压 (hPa/mbar)

        # ── 风 ─────────────────────────────────────────────────
        "wind_spd":      record.get("wind_spd",      None),  # 风速 (m/s)
        "wind_dir":      record.get("wind_dir",      None),  # 风向 (°)
        "wind_gust_spd": record.get("wind_gust_spd", None),  # 阵风风速 (m/s)

        # ── 能见度 ─────────────────────────────────────────────
        "vis": record.get("vis", None),   # 能见度 (km)

        # ── 云量 ────────────────────────────────────────────────
        "clouds": record.get("clouds", None),  # 总云量 (%)

        # ── 太阳位置 ────────────────────────────────────────────
        "solar_alt":  record.get("elev_angle", None),  # 太阳高度角 (°)
        "solar_az":   record.get("azimuth",   None),  # 太阳方位角 (°)

        # ── 辐射 / 辐照度 ───────────────────────────────────────
        "ghi":         record.get("ghi",         None),  # 水平面总辐照度 (W/m²)
        "dni":         record.get("dni",         None),  # 直接法向辐照度 (W/m²)
        "dhi":         record.get("dhi",         None),  # 水平面散射辐照度 (W/m²)
        "solar_rad":  record.get("solar_rad",  None),  # 太阳辐射 (W/m²)

        # ── 紫外线 ─────────────────────────────────────────────
        "uv": record.get("uv", None),   # UV 指数

        # ── 降水 / 降雪 ────────────────────────────────────────
        "precip_rate":  record.get("precip_rate",  None),  # 降水率 (mm/hr)
        "snow_rate":    record.get("snow_rate",    None),  # 降雪率 (mm/hr)

        # ── 天气状态 ────────────────────────────────────────────
        "pod":       record.get("pod",       ""),  # part of day: d=昼, n=夜
        "weather_code": weather.get("code",        None),  # Weatherbit 天气码
        "weather_desc": weather.get("description",  ""),  # 天气描述（英文）
        "weather_icon": weather.get("icon",        ""),  # 天气图标代码
    }


def save_checkpoint(records: list, filepath: Path):
    """保存中间检查点，防止意外中断导致数据丢失。"""
    df = pd.DataFrame([flatten_record(r) for r in records])
    df.to_csv(filepath.with_suffix(".checkpoint.csv"), index=False, encoding="utf-8")


# ==================== 主程序 ====================

def main():
    print("=" * 70)
    print("Weatherbit.io 全量天气参数获取")
    print("=" * 70)
    print(f"位置: ({LAT}, {LON})")
    print(f"时间范围: {START_DATE} ~ {END_DATE}")
    print(f"输出文件: {OUTPUT_FILE}")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    date_ranges = get_month_ranges(START_DATE, END_DATE)
    print(f"将分 {len(date_ranges)} 批请求数据:\n")
    for i, (s, e) in enumerate(date_ranges):
        print(f"  批次 {i+1:02d}: {s} ~ {e}")
    print()

    all_records = []
    CHECKPOINT_INTERVAL = 3  # 每 N 批保存一次检查点

    for i, (start, end) in enumerate(date_ranges):
        print(f"[批次 {i+1:02d}/{len(date_ranges)}] 请求 {start} ~ {end}")

        records = fetch_weatherbit_batch(start, end)

        if records:
            all_records.extend(records)

            # 预览前两条
            for r in records[:2]:
                desc = r.get("weather", {}).get("description", "N/A")
                print(
                    f"    {r['timestamp_local']} | {r['temp']}°C | "
                    f"{r['rh']}% | {r['clouds']}% clouds | {desc}"
                )
            if len(records) > 4:
                print(f"    ... 共 {len(records)} 条 ...")
                for r in records[-2:]:
                    desc = r.get("weather", {}).get("description", "N/A")
                    print(
                        f"    {r['timestamp_local']} | {r['temp']}°C | "
                        f"{r['rh']}% | {r['clouds']}% clouds | {desc}"
                    )
        else:
            print(f"  [警告] 该批次无数据")

        # 定期保存检查点
        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(all_records, OUTPUT_FILE)
            print(f"  [检查点] 已保存 ~{len(all_records)} 条记录")

        print()

        if i < len(date_ranges) - 1:
            print(f"  等待 {REQUEST_DELAY} 秒 ...")
            time.sleep(REQUEST_DELAY)

    # ── 最终处理 ────────────────────────────────────────────────────────────
    if all_records:
        df = pd.DataFrame([flatten_record(r) for r in all_records])

        # 按本地时间排序
        df = df.sort_values("timestamp_local").reset_index(drop=True)

        # 去重（保留首次出现）
        before = len(df)
        df = df.drop_duplicates(subset=["timestamp_local"], keep="first")
        after = len(df)
        if before != after:
            print(f"[去重] 去除 {before - after} 条重复记录")

        # 保存完整数据
        df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

        print("=" * 70)
        print("数据获取完成！")
        print("=" * 70)
        print(f"总记录数: {len(df):,}")
        print(f"时间范围: {df['timestamp_local'].iloc[0]} ~ {df['timestamp_local'].iloc[-1]}")
        print(f"输出文件: {OUTPUT_FILE}")
        print()
        print(f"共 {len(df.columns)} 个字段:")
        for col in df.columns:
            null_pct = df[col].isna().mean() * 100
            print(f"  {col:<25s}  空值率 {null_pct:5.1f}%")
        print()
        print("【数据预览（前5行）】")
        print(df.head().to_string(index=False))
    else:
        print("未获取到任何数据，请检查 API 密钥和日期范围。")


if __name__ == "__main__":
    main()
