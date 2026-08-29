"""
修复已对齐预处理 CSV 中的辐照-功率物理不一致问题。
适用于 raw CSV 不可用时，直接修正 processed/stations/*_preprocessed.csv。

修复规则（与 run_exp_p01_preprocessing.py 3.7b / 3.10 一致）：
  1. irr < 20 且 power > 5% 容量 → 置 NaN 后填 0
  2. irr <= 5 → 强制 power = 0
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STATIONS_DIR = PROJECT_ROOT / "data" / "prediction" / "step1_preprocessing" / "processed" / "stations"


def robust_scale(series: pd.Series) -> pd.Series:
    med = float(series.median())
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = float(q3 - q1)
    scale = iqr if iqr > 1e-8 else (float(series.std()) if float(series.std()) > 1e-8 else 1.0)
    return (series - med) / scale


def repair_site(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    cap = float(out["capacity_mw"].iloc[0])
    irr = out["total_irradiance_wm2"].fillna(0)
    pwr = out["power_mw"]

    inconsistent = (irr < 20) & (pwr > 0.05 * cap)
    low_irr = irr <= 5
    to_zero = low_irr & (out["power_mw"].fillna(0) != 0)

    n_inconsistent = int(inconsistent.sum())
    n_low_irr_zero = int(to_zero.sum())

    if n_inconsistent:
        out.loc[inconsistent, "power_mw_outlier_flag"] = 1
        out.loc[inconsistent, "power_mw"] = 0.0
        out.loc[inconsistent, "power_mw_imputed_flag"] = 1

    if n_low_irr_zero:
        out.loc[to_zero, "power_mw"] = 0.0
        out.loc[to_zero, "power_mw_imputed_flag"] = 1

    out["power_pu"] = out["power_mw"] / cap
    out["power_ramp_15m_mw"] = out["power_mw"].diff()
    out["power_ramp_15m_pu"] = out["power_pu"].diff()

    core = [
        "total_irradiance_wm2", "direct_normal_irradiance_wm2", "global_horizontal_irradiance_wm2",
        "air_temperature_c", "atmosphere_hpa", "relative_humidity_pct", "power_mw",
    ]
    imp_cols = [f"{c}_imputed_flag" for c in core]
    out["imputed_feature_count"] = out[imp_cols].sum(axis=1).astype(int)
    n_feat = len(core)
    out["data_quality_score"] = (1 - out["imputed_feature_count"] / n_feat).clip(lower=0)

    for feat in ["power_mw", "power_pu", "power_ramp_15m_mw", "power_ramp_15m_pu"]:
        col = f"{feat}_robust_scaled"
        if feat in out.columns:
            out[col] = robust_scale(out[feat])

    return out, {
        "n_inconsistent_fixed": n_inconsistent,
        "n_low_irr_zeroed": n_low_irr_zero,
    }


def main() -> None:
    paths = sorted(STATIONS_DIR.glob("Site_*_preprocessed.csv"))
    if not paths:
        print("未找到预处理 CSV", file=sys.stderr)
        sys.exit(1)

    for path in paths:
        df = pd.read_csv(path, parse_dates=["timestamp"])
        fixed, stats = repair_site(df)
        fixed.to_csv(path, index=False)
        print(
            f"{path.name}: 不一致修正 {stats['n_inconsistent_fixed']:,} 行, "
            f"低辐照置零 {stats['n_low_irr_zeroed']:,} 行"
        )


if __name__ == "__main__":
    main()
