"""
亳州场站天气数据获取脚本 - 统一入口
通过 Weatherbit API 获取天气数据

使用新坐标: 东经115°47'55.003", 北纬33°47'39.624"
- 115°47'55.003"E = 115.7986°
- 33°47'39.624"N = 33.7943°N

使用方式:
    python fetch_bozhou.py                          # 默认：获取完整数据
    python fetch_bozhou.py --mode full              # 获取完整数据
    python fetch_bozhou.py --mode missing           # 补充缺失数据
    python fetch_bozhou.py --mode boundary          # 获取月末月初边界数据
"""

import requests
import pandas as pd
import time
from pathlib import Path
from datetime import datetime, timedelta
from calendar import monthrange
import argparse

# ==================== 配置区 ====================

API_KEY = "6ab7f37c60e94aa8af1277feb03b1b96"

# 亳州场站坐标
LAT = 33.7943   # 北纬 33°47'39.624"N
LON = 115.7986  # 东经 115°47'55.003"E

SITE_NAME = "亳州"

# 输出目录
OUTPUT_DIR = Path(r"C:\Users\MoYu\Desktop\pv_prediction_scheduling_msc_new\data\raw\亳州")

# 请求间隔（秒）
REQUEST_DELAY = 3

# 月初/月末边界天数
BOUNDARY_DAYS = 3


# ==================== 工具函数 ====================

def fetch_weatherbit_batch(lat: float, lon: float, start_date: str, end_date: str, max_retries: int = 3) -> list:
    """
    单批次请求 Weatherbit subhourly API。
    
    Args:
        lat, lon: 坐标
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        max_retries: 最大重试次数
    
    Returns:
        records 列表
    """
    url = (
        "https://api.weatherbit.io/v2.0/history/subhourly"
        f"?start_date={start_date}&end_date={end_date}"
        f"&lat={lat}&lon={lon}"
        f"&key={API_KEY}"
    )

    for attempt in range(max_retries):
        try:
            print(f"  [请求] {start_date} ~ {end_date} (尝试 {attempt + 1}/{max_retries}) ...")
            resp = requests.get(url, timeout=120)

            if resp.status_code == 200:
                data = resp.json()
                records = data.get("data", [])
                city = data.get("city_name", "Unknown")
                timezone = data.get("timezone", "Unknown")
                print(f"  [成功] {city} ({timezone})，获取 {len(records)} 条记录")
                return records

            elif resp.status_code == 429:
                wait_time = 90 * (attempt + 1)
                print(f"  [限流] 等待 {wait_time} 秒后重试 ...")
                time.sleep(wait_time)

            elif resp.status_code == 400:
                print(f"  [错误] 400: 日期范围可能超出 API 支持范围")
                return []

            else:
                print(f"  [错误] HTTP {resp.status_code}: {resp.text[:200]}")
                if attempt < max_retries - 1:
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
    """将 Weatherbit 单条记录展平为单层 dict"""
    weather = record.get("weather", {})
    return {
        # 时间
        "timestamp_local": record.get("timestamp_local", ""),
        "timestamp_utc": record.get("timestamp_utc", ""),
        "ts": record.get("ts", None),
        
        # 温度
        "temp": record.get("temp", None),
        "app_temp": record.get("app_temp", None),
        
        # 湿度/气压
        "rh": record.get("rh", None),
        "dewpt": record.get("dewpt", None),
        "pres": record.get("pres", None),
        
        # 风
        "wind_spd": record.get("wind_spd", None),
        "wind_dir": record.get("wind_dir", None),
        "wind_gust_spd": record.get("wind_gust_spd", None),
        
        # 其他
        "vis": record.get("vis", None),
        "clouds": record.get("clouds", None),
        
        # 太阳位置
        "solar_alt": record.get("elev_angle", None),
        "solar_az": record.get("azimuth", None),
        
        # 辐射
        "ghi": record.get("ghi", None),
        "dni": record.get("dni", None),
        "dhi": record.get("dhi", None),
        "solar_rad": record.get("solar_rad", None),
        
        # 紫外线
        "uv": record.get("uv", None),
        
        # 降水
        "precip_rate": record.get("precip_rate", None),
        "snow_rate": record.get("snow_rate", None),
        
        # 天气状态
        "pod": record.get("pod", ""),
        "weather_code": weather.get("code", None),
        "weather_desc": weather.get("description", ""),
        "weather_icon": weather.get("icon", ""),
        
        # 站点信息
        "site": SITE_NAME,
        "lat": LAT,
        "lon": LON,
    }


def save_checkpoint(records: list, filepath: Path):
    """保存中间检查点"""
    df = pd.DataFrame([flatten_record(r) for r in records])
    df.to_csv(filepath.with_suffix(".checkpoint.csv"), index=False, encoding="utf-8")
    print(f"  [检查点] 已保存 {len(records)} 条记录 → {filepath.with_suffix('.checkpoint.csv').name}")


def get_month_ranges_with_full_coverage(start_date: str, end_date: str, boundary_days: int = 3) -> list:
    """
    生成按月分段的请求日期范围列表。
    
    关键处理:
    - 月初: 请求起始日期 = 上月末日（确保获取本月1日00:00数据）
    - 月末: 请求结束日期 = 下月1-2日（确保获取月末最后几小时数据）
    - 补充: 对于完整月份，需要单独请求中间部分
    """
    ranges = []
    target_start = datetime.strptime(start_date, "%Y-%m-%d")
    target_end = datetime.strptime(end_date, "%Y-%m-%d")
    
    print(f"生成数据范围 (目标: {start_date} ~ {end_date}, 边界天数: {boundary_days})")
    print()

    current = datetime(target_start.year, target_start.month, 1)
    
    while current <= target_end:
        year = current.year
        month = current.month
        
        month_start = datetime(year, month, 1)
        _, last_day = monthrange(year, month)
        month_end = datetime(year, month, last_day)
        
        if month == 1:
            prev_month_end = datetime(year - 1, 12, 31)
        else:
            _, last_prev = monthrange(year, month - 1)
            prev_month_end = datetime(year, month - 1, last_prev)
        
        if month == 12:
            next_month_start = datetime(year + 1, 1, 1)
        else:
            next_month_start = datetime(year, month + 1, 1)
        
        # 月初数据范围
        req_start = prev_month_end - timedelta(days=boundary_days - 1)
        req_end = month_start + timedelta(days=boundary_days - 1)
        
        if req_start <= req_end:
            ranges.append({
                "start": req_start.strftime("%Y-%m-%d"),
                "end": req_end.strftime("%Y-%m-%d"),
                "desc": f"{month}月月初"
            })
        
        # 月末数据范围
        req_start_end = month_end - timedelta(days=boundary_days - 1)
        req_end_end = next_month_start + timedelta(days=boundary_days - 1)
        
        if req_start_end <= req_end_end:
            ranges.append({
                "start": req_start_end.strftime("%Y-%m-%d"),
                "end": req_end_end.strftime("%Y-%m-%d"),
                "desc": f"{month}月月末"
            })
        
        # 中间月份数据
        mid_start = month_start + timedelta(days=boundary_days)
        mid_end = month_end - timedelta(days=boundary_days)
        
        if mid_start <= mid_end and (mid_end - mid_start).days >= 7:
            ranges.append({
                "start": mid_start.strftime("%Y-%m-%d"),
                "end": mid_end.strftime("%Y-%m-%d"),
                "desc": f"{month}月中"
            })
        
        current = next_month_start

    # 去重并按日期排序
    seen = set()
    unique_ranges = []
    for r in ranges:
        key = (r["start"], r["end"])
        if key not in seen:
            seen.add(key)
            unique_ranges.append(r)
    
    return sorted(unique_ranges, key=lambda x: x["start"])


def get_missing_periods() -> list:
    """
    获取因 boundary_days=3 导致的数据缺口时间段。
    每月3-4日和27-29日之间存在数据缺口。
    """
    return [
        # 月初缺口 (每月3日~5日)
        ("2026-03-03", "2026-03-05", "3月初"),
        ("2026-04-03", "2026-04-05", "4月初"),
        ("2026-05-03", "2026-05-05", "5月初"),
        ("2026-06-03", "2026-06-05", "6月初"),
        ("2026-07-03", "2026-07-05", "7月初"),
        ("2026-08-03", "2026-08-05", "8月初"),
        # 月末缺口 (每月27-29日)
        ("2026-03-28", "2026-03-30", "3月末"),
        ("2026-04-27", "2026-04-29", "4月末"),
        ("2026-05-28", "2026-05-30", "5月末"),
        ("2026-06-27", "2026-06-29", "6月末"),
        ("2026-07-28", "2026-07-30", "7月末"),
        ("2026-08-28", "2026-08-30", "8月末"),
    ]


def get_month_boundary_periods() -> list:
    """
    获取月末月初的完整时间段。
    包含月末3天和月初3天，确保白天数据完整。
    """
    return [
        ("2026-03-28", "2026-04-04", "3月末~4月初"),
        ("2026-04-27", "2026-05-04", "4月末~5月初"),
        ("2026-05-28", "2026-06-04", "5月末~6月初"),
        ("2026-06-27", "2026-07-04", "6月末~7月初"),
        ("2026-07-28", "2026-08-04", "7月末~8月初"),
        ("2026-08-29", "2026-09-02", "8月末"),
    ]


def fetch_full_data(args):
    """模式1: 获取完整数据"""
    print("=" * 70)
    print("模式: 获取完整数据")
    print("=" * 70)
    print(f"时间范围: {args.start} ~ {args.end}")
    print(f"边界天数: {args.boundary} 天")
    print()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / args.output

    # 生成请求范围
    date_ranges = get_month_ranges_with_full_coverage(args.start, args.end, args.boundary)
    
    print(f"将分 {len(date_ranges)} 批请求数据:\n")
    for i, r in enumerate(date_ranges):
        print(f"  批次 {i+1:02d}: {r['start']} ~ {r['end']} [{r['desc']}]")
    print()

    # 检查检查点
    all_records = []
    if args.checkpoint and checkpoint_file.exists():
        df_check = pd.read_csv(checkpoint_file)
        all_records = df_check.to_dict("records")
        print(f"[恢复] 从检查点加载 {len(all_records)} 条记录\n")

    checkpoint_file = OUTPUT_DIR / (args.output + ".checkpoint.csv")

    # 开始请求
    print("=" * 70)
    print("开始获取数据...")
    print("=" * 70)
    
    for i, r in enumerate(date_ranges):
        print(f"\n[批次 {i+1:02d}/{len(date_ranges)}] {r['desc']}")
        print(f"  请求: {r['start']} ~ {r['end']}")

        records = fetch_weatherbit_batch(LAT, LON, r["start"], r["end"])

        if records:
            all_records.extend(records)
            print(f"  当前累计: {len(all_records)} 条")
            
            sample = records[0]
            desc = sample.get("weather", {}).get("description", "N/A")
            print(f"  首条: {sample['timestamp_local']} | {sample['temp']}°C | {sample['rh']}% | {desc}")
        else:
            print(f"  [警告] 该批次无数据")

        # 保存检查点
        if args.checkpoint and (i + 1) % 2 == 0:
            save_checkpoint(all_records, checkpoint_file)

        print()
        if i < len(date_ranges) - 1:
            print(f"  等待 {args.delay} 秒 ...")
            time.sleep(args.delay)

    # 最终处理
    if all_records:
        df = pd.DataFrame([flatten_record(r) for r in all_records])
        df = df.sort_values("timestamp_local").reset_index(drop=True)

        before = len(df)
        df = df.drop_duplicates(subset=["timestamp_local"], keep="first")
        after = len(df)
        if before != after:
            print(f"[去重] 去除 {before - after} 条重复记录")

        df.to_csv(output_file, index=False, encoding="utf-8")
        
        # 删除检查点
        if args.checkpoint and checkpoint_file.exists():
            checkpoint_file.unlink()
            print(f"[清理] 删除检查点文件")

        print("=" * 70)
        print("数据获取完成！")
        print("=" * 70)
        print(f"总记录数: {len(df):,}")
        print(f"时间范围: {df['timestamp_local'].iloc[0]} ~ {df['timestamp_local'].iloc[-1]}")
        print(f"输出文件: {output_file}")
    else:
        print("未获取到任何数据，请检查 API 密钥和日期范围。")


def fetch_missing_data(args):
    """模式2: 补充缺失数据"""
    print("=" * 70)
    print("模式: 补充缺失数据")
    print("=" * 70)
    print("补充因 boundary_days=3 导致的数据缺口\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    original_file = OUTPUT_DIR / args.output

    missing_ranges = get_missing_periods()
    all_missing_records = []

    for i, (start, end, desc) in enumerate(missing_ranges):
        print(f"\n[{i+1}/{len(missing_ranges)}] {desc}: {start} ~ {end}")
        records = fetch_weatherbit_batch(LAT, LON, start, end)
        if records:
            all_missing_records.extend(records)
            print(f"  获取 {len(records)} 条记录")
        time.sleep(args.delay)

    if all_missing_records:
        # 读取原始数据
        if original_file.exists():
            df_original = pd.read_csv(original_file)
            print(f"\n原始数据: {len(df_original)} 条")
        else:
            print(f"\n警告: 原始文件不存在 {original_file}")
            df_original = pd.DataFrame()

        # 展平新数据
        df_new = pd.DataFrame([flatten_record(r) for r in all_missing_records])
        
        print(f"新增数据: {len(df_new)} 条")

        # 合并数据
        if len(df_original) > 0:
            df_combined = pd.concat([df_original, df_new], ignore_index=True)
        else:
            df_combined = df_new
        
        # 按时间排序
        df_combined = df_combined.sort_values("timestamp_local").reset_index(drop=True)
        
        # 去重
        before = len(df_combined)
        df_combined = df_combined.drop_duplicates(subset=["timestamp_local"], keep="first")
        after = len(df_combined)
        print(f"合并后: {after} 条 (去除 {before - after} 条重复)")

        # 保存
        df_combined.to_csv(original_file, index=False, encoding="utf-8")
        
        print(f"\n数据已保存: {original_file}")
        print(f"总记录数: {len(df_combined)}")
    else:
        print("未获取到任何数据。")


def fetch_boundary_data(args):
    """模式3: 获取月末月初边界数据"""
    print("=" * 70)
    print("模式: 获取月末月初边界数据")
    print("=" * 70)
    print("获取月末3天和月初3天的完整数据（包括白天）\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "亳州_月末月初数据.csv"

    boundary_ranges = get_month_boundary_periods()
    all_records = []

    for i, (start, end, desc) in enumerate(boundary_ranges):
        print(f"\n[{i+1}/{len(boundary_ranges)}] {desc}: {start} ~ {end}")
        records = fetch_weatherbit_batch(LAT, LON, start, end)
        if records:
            for r in records:
                flat = flatten_record(r)
                all_records.append(flat)
            print(f"  获取 {len(records)} 条记录")
        print(f"  当前累计: {len(all_records)} 条")
        time.sleep(args.delay)

    if all_records:
        df = pd.DataFrame(all_records)
        df = df.sort_values("timestamp_local").reset_index(drop=True)
        
        before = len(df)
        df = df.drop_duplicates(subset=["timestamp_local"], keep="first")
        after = len(df)
        if before != after:
            print(f"[去重] 去除 {before - after} 条重复记录")

        df.to_csv(output_file, index=False, encoding="utf-8")
        
        print("\n" + "=" * 70)
        print("数据获取完成！")
        print("=" * 70)
        print(f"总记录数: {len(df)}")
        print(f"时间范围: {df['timestamp_local'].iloc[0]} ~ {df['timestamp_local'].iloc[-1]}")
        print(f"输出文件: {output_file}")
    else:
        print("未获取到任何数据。")


# ==================== 主程序 ====================

def main():
    parser = argparse.ArgumentParser(
        description="亳州场站天气数据获取 - 统一入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python fetch_bozhou.py --mode full --start 2026-03-01 --end 2026-09-21
  python fetch_bozhou.py --mode missing
  python fetch_bozhou.py --mode boundary
        """
    )
    
    parser.add_argument("--mode", choices=["full", "missing", "boundary"], default="full",
                        help="获取模式: full=完整数据, missing=补充缺失, boundary=月末月初")
    parser.add_argument("--start", default="2026-03-01", help="开始日期 (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-09-21", help="结束日期 (YYYY-MM-DD)")
    parser.add_argument("--boundary", type=int, default=3, help="月初月末边界天数")
    parser.add_argument("--output", default="亳州_天气数据.csv", help="输出文件名")
    parser.add_argument("--delay", type=int, default=3, help="请求间隔(秒)")
    parser.add_argument("--checkpoint", action="store_true", help="启用检查点保存")
    
    args = parser.parse_args()

    print()
    print("=" * 70)
    print("亳州场站天气数据获取")
    print("=" * 70)
    print(f"场站: {SITE_NAME}")
    print(f"坐标: ({LAT}°N, {LON}°E)")
    print(f"API: Weatherbit.io")
    print()

    if args.mode == "full":
        fetch_full_data(args)
    elif args.mode == "missing":
        fetch_missing_data(args)
    elif args.mode == "boundary":
        fetch_boundary_data(args)


if __name__ == "__main__":
    main()
