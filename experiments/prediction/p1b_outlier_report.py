"""
输出 Hampel 异常检测结果清单 CSV

当前配置：仅对 power_kw 做异常检测，
temperature_c / relative_humidity_pct / cloud_cover_pct / ghi_wm2 不做 Hampel（原样保留）。

用法:
    python experiments/prediction/p1b_outlier_report.py \
        --before data/processed/Bozhou_1_before_hampel.csv \
        --detected data/processed/Bozhou_1_before_interpolation_two_stage_sigma3_165006.csv \
        --final data/processed/Bozhou_1_preprocessed_two_stage_sigma3.csv
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'

WEATHER_FEATURES = ['temperature_c', 'relative_humidity_pct', 'cloud_cover_pct', 'ghi_wm2']


def find_latest(pattern):
    """按文件名中提取的时间戳找最新的匹配文件"""
    files = sorted(PROCESSED_DIR.glob(pattern))
    if not files:
        return None
    def ts_of(f):
        parts = f.stem.split('_')
        for p in reversed(parts):
            if p.isdigit() and len(p) == 6:
                return p
        return ''
    return sorted(files, key=ts_of)[-1]


def main():
    parser = argparse.ArgumentParser(description='生成 Hampel 异常检测结果清单')
    parser.add_argument('--before', type=str, default=None,
                        help='异常检测前的基线CSV (默认: data/processed/Bozhou_1_before_hampel.csv)')
    parser.add_argument('--detected', type=str, default=None,
                        help='异常检测后、插值前的CSV (默认: 自动查找最新)')
    parser.add_argument('--final', type=str, default=None,
                        help='完整预处理后的CSV (默认: 自动查找最新)')
    parser.add_argument('--out', type=str, default=None,
                        help='输出CSV路径 (默认: data/processed/Bozhou_1_outlier_report.csv)')
    args = parser.parse_args()

    before_path = Path(args.before) if args.before else PROCESSED_DIR / 'Bozhou_1_before_hampel.csv'
    detected_path = Path(args.detected) if args.detected else find_latest('Bozhou_1_before_interpolation_*.csv')
    final_path = Path(args.final) if args.final else find_latest('Bozhou_1_preprocessed_*.csv')
    out_path = Path(args.out) if args.out else PROCESSED_DIR / 'Bozhou_1_outlier_report.csv'

    if not before_path.exists():
        print(f'[错误] 找不到异常检测前基线: {before_path}')
        return 1
    if detected_path is None or not detected_path.exists():
        print('[错误] 找不到异常检测后CSV，请用 --detected 指定')
        return 1

    print(f'基线文件   : {before_path.name}')
    print(f'检测后文件 : {detected_path.name}')
    print(f'最终文件   : {final_path.name if final_path else "(无)"}')

    before = pd.read_csv(before_path, parse_dates=['timestamp'])
    detected = pd.read_csv(detected_path, parse_dates=['timestamp'])

    # 以 timestamp 对齐
    b = before.set_index('timestamp')
    d = detected.set_index('timestamp')
    common = b.index.intersection(d.index)

    report = pd.DataFrame(index=common)

    # 功率：原始值 + 清洗后值 + 是否异常
    report['power_kw_raw'] = b.loc[common, 'power_kw']
    report['power_kw_cleaned'] = d.loc[common, 'power_kw']
    report['is_power_outlier'] = (d.loc[common, 'power_kw'].isna()
                                  & b.loc[common, 'power_kw'].notna()).astype(int)

    # 天气特征：原样保留，不做检测
    for col in WEATHER_FEATURES:
        if col in b.columns:
            report[f'{col}_raw'] = b.loc[common, col]
            report[f'{col}_detected'] = d.loc[common, col]
            report[f'{col}_changed'] = (b.loc[common, col].values != d.loc[common, col].values).astype(int)

    report['hour'] = common.hour
    report['is_night'] = (report['hour'] >= 20) | (report['hour'] < 5)

    # 是否被后续插值回填
    if final_path is not None and final_path.exists():
        fin = pd.read_csv(final_path, parse_dates=['timestamp']).set_index('timestamp')
        if 'power_kw' in fin.columns:
            fin_common = fin.index.intersection(common)
            filled = pd.Series(False, index=common)
            filled.loc[fin_common] = (report.loc[fin_common, 'is_power_outlier'] == 1) & \
                                     fin.loc[fin_common, 'power_kw'].notna()
            report['refilled_by_interpolation'] = filled.astype(int)

    report = report.reset_index()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_path, index=False)

    n_out = int(report['is_power_outlier'].sum())
    print('\n' + '=' * 70)
    print(f'异常检测结果清单已保存: {out_path}')
    print('=' * 70)
    print(f'  总行数        : {len(report):,}')
    print(f'  功率异常点    : {n_out:,}')
    for col in WEATHER_FEATURES:
        ch = f'{col}_changed'
        if ch in report.columns:
            print(f'  {col:20s} 被改动点: {int(report[ch].sum()):,} (应为 0)')
    if 'refilled_by_interpolation' in report.columns:
        print(f'  其中被插值回填: {int(report["refilled_by_interpolation"].sum()):,}')

    if n_out:
        print('\n异常点时刻分布:')
        dist = report[report['is_power_outlier'] == 1]['hour'].value_counts().sort_index()
        for h, n in dist.items():
            tag = '夜' if h >= 20 or h < 5 else '昼'
            print(f'  {h:02d}:00  {n:3d}  ({tag})')

        print('\n前10个异常点:')
        cols = ['timestamp', 'power_kw_raw', 'power_kw_cleaned',
                'cloud_cover_pct_raw', 'ghi_wm2_raw']
        cols = [c for c in cols if c in report.columns]
        print(report[report['is_power_outlier'] == 1][cols].head(10).to_string(index=False))

    return 0


if __name__ == '__main__':
    sys.exit(main())
