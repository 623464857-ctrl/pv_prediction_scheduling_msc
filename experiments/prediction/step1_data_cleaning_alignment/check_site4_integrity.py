"""Quick integrity check for Site_4 preprocessed CSV."""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PATH = PROJECT_ROOT / "data/prediction/step1_preprocessing/processed/stations/Site_4_preprocessed.csv"


def main():
    df = pd.read_csv(PATH, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    cap = float(df["capacity_mw"].iloc[0])

    print("=" * 60)
    print("Site_4_preprocessed.csv 完整性 & 夜间功率检查")
    print("=" * 60)
    print(f"行数: {len(df):,}")
    print(f"时间范围: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
    print(f"额定容量: {cap} MW")

    full_idx = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="15min")
    print("\n【1. 时间轴完整性】")
    print(f"期望 15min 步数: {len(full_idx):,}")
    print(f"实际行数: {len(df):,}")
    print(f"差值: {len(full_idx) - len(df):,}")
    print(f"重复 timestamp: {df['timestamp'].duplicated().sum()}")

    actual_set = set(df["timestamp"])
    missing = [t for t in full_idx if t not in actual_set]
    print(f"缺失时间点数量: {len(missing)}")
    if missing:
        print("缺失示例:", missing[:5])

    diffs = df["timestamp"].diff().dropna()
    bad = diffs[diffs != pd.Timedelta("15min")]
    print(f"非 15min 间隔跳变: {len(bad)}")
    for idx in bad.index[:5]:
        prev = df.loc[idx - 1, "timestamp"]
        curr = df.loc[idx, "timestamp"]
        print(f"  {prev} -> {curr}  delta={bad.loc[idx]}")

    print(f"row_inserted_by_reindex=1: {int(df['row_inserted_by_reindex'].sum()):,} "
          f"({100 * df['row_inserted_by_reindex'].mean():.2f}%)")
    print(f"source_observed_flag=0: {int((df['source_observed_flag'] == 0).sum()):,}")

    core = [
        "total_irradiance_wm2", "direct_normal_irradiance_wm2", "global_horizontal_irradiance_wm2",
        "air_temperature_c", "atmosphere_hpa", "relative_humidity_pct", "power_mw",
    ]
    print("\n【2. 核心字段 NaN】")
    any_nan = False
    for col in core:
        nan_n = int(df[col].isna().sum())
        if nan_n:
            any_nan = True
            print(f"  {col}: {nan_n} NaN")
    if not any_nan:
        print("  无 NaN（7 个核心列均已填满）")

    print("\n【3. 夜间功率检查】")
    defs = {
        "irr<=5 (P01 夜间置零条件)": df["total_irradiance_wm2"] <= 5,
        "irr<=20 (daylight_flag 边界)": df["total_irradiance_wm2"] <= 20,
        "hour<=5 或 hour>=20": (df["hour"] <= 5) | (df["hour"] >= 20),
    }
    for name, night in defs.items():
        sub = df[night]
        nz = sub[sub["power_mw"] > 0]
        print(f"\n  定义: {name}")
        print(f"    夜间行数: {len(sub):,}")
        print(f"    power_mw>0: {len(nz):,} ({100 * len(nz) / max(len(sub), 1):.3f}%)")
        if len(nz):
            print(f"    功率: {nz['power_mw'].min():.4f} ~ {nz['power_mw'].max():.4f} MW")
            print(f"    辐照: {nz['total_irradiance_wm2'].min():.2f} ~ {nz['total_irradiance_wm2'].max():.2f}")
            top = nz.nlargest(5, "power_mw")[
                ["timestamp", "total_irradiance_wm2", "power_mw", "hour", "daylight_flag",
                 "power_mw_imputed_flag", "power_mw_outlier_flag"]
            ]
            print("    功率最大 5 条:")
            for _, r in top.iterrows():
                print(
                    f"      {r['timestamp']}  irr={r['total_irradiance_wm2']:.1f}  "
                    f"pwr={r['power_mw']:.3f}  h={int(r['hour'])}  "
                    f"imp={int(r['power_mw_imputed_flag'])}  out={int(r['power_mw_outlier_flag'])}"
                )

    strict = df[(df["total_irradiance_wm2"] <= 5) & (df["power_mw"] > 0.01)]
    print(f"\n  严格异常 (irr<=5 & power>0.01 MW): {len(strict)} 条")

    z = df[(df["total_irradiance_wm2"] == 0) & (df["power_mw"] > 0)]
    print(f"  irr=0 且 power>0: {len(z)} 条")

    # 夜间正功率是否来自插补/回填
    night_pos = df[(df["total_irradiance_wm2"] <= 5) & (df["power_mw"] > 0)]
    if len(night_pos):
        print(f"\n  夜间正功率来源:")
        print(f"    imputed_flag=1: {int((night_pos['power_mw_imputed_flag']==1).sum()):,}")
        print(f"    outlier后被回填: 见 imputed/outlier 列")

    print("\n【4. 质量评分】")
    print(f"  data_quality_score 均值: {df['data_quality_score'].mean():.4f}")
    print(f"  score<1: {(df['data_quality_score'] < 1).sum():,}")
    print(f"  imputed_feature_count>0: {(df['imputed_feature_count'] > 0).sum():,}")


if __name__ == "__main__":
    main()
