"""
调度 S1: 光储充运行数据整理

输入: data/schedule_data/*.xlsx  (5 分钟, 光伏功率 / 储能功率 / 充电功率 / SOC, 单位 W 和 %)
输出: data/processed/schedule_15min.csv
        time      15 分钟时段起点
        pv_kw     光伏功率 (负值置 0)
        ev_kw     充电桩功率 (缺失记为 0, 见下)
        ess_kw    储能功率 (正 = 放电, 负 = 充电), 仅供参考
        soc       储能 SOC (%), 仅供参考

充电功率缺失的处理: 每段缺失之前的读数都是 0, 且缺失集中在夜间, 说明采集系统在没有车充电时不上报,
所以缺失按 0 kW 处理。

python -m experiments.scheduling.s1_prepare_data
"""

import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.common import SCHEDULE_CSV, SCHEDULE_RAW_DIR
from experiments.scheduling.config import EV_P_MAX

COLUMNS = ["time", "station", "pv_w", "ess_w", "ev_w", "soc", "ess_ch_kwh_hourly", "ess_dis_kwh_hourly"]


def load_raw() -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # openpyxl 的样式警告
        frames = [pd.read_excel(f) for f in sorted(SCHEDULE_RAW_DIR.glob("*.xlsx"))]
    df = pd.concat(frames, ignore_index=True)
    df.columns = COLUMNS
    df["time"] = pd.to_datetime(df["time"])
    return df.drop_duplicates("time").set_index("time").sort_index()


def main():
    raw = load_raw()
    df = pd.DataFrame({
        "pv_kw": raw["pv_w"].clip(lower=0) / 1000,
        "ev_kw": raw["ev_w"].fillna(0).clip(upper=EV_P_MAX * 1000) / 1000,
        "ess_kw": raw["ess_w"] / 1000,
        "soc": raw["soc"],
    })
    out = df.resample("15min").mean()
    out = out[out.index < "2026-09-01"]
    SCHEDULE_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.round(4).to_csv(SCHEDULE_CSV, index_label="time")

    print(f"{out.index.min()} ~ {out.index.max()}, {len(out)} 个时段 -> {SCHEDULE_CSV}")
    print(f"光伏: 最大 {out.pv_kw.max():.0f} kW | 充电桩: 平均 {out.ev_kw.mean():.1f} kW, "
          f"最大 {out.ev_kw.max():.0f} kW, 日均电量 {out.ev_kw.sum() * 0.25 / (len(out) / 96):.0f} kWh")


if __name__ == "__main__":
    main()
