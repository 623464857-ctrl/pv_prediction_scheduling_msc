"""
调度 S2: 逐日滚动调度 + 实际结算, 对比各策略

输入:
  data/processed/Bozhou_1_cleaned.csv                 实际光伏 (与预测模型同一数据源)
  data/processed/schedule_15min.csv                   充电桩负荷 (S1 输出)
  result/prediction/predictions/h1_test.csv           CNN-BiLSTM 测试集预测 (预测 P6 输出)
  result/prediction/predictions/h1_val.csv            验证集预测, 用于估计误差分布 (非对称修正)
输出 (result/scheduling/):
  steps_<策略>.csv     每 15 分钟的计划、实际功率和费用
  summary.csv          各策略汇总 (成本、偏差电量、消纳率等)
  figures/*.png

调度时段: 预测测试集覆盖的完整天 (2026-08-05 ~ 2026-08-31)。

python -m experiments.scheduling.s2_run_dispatch
python -m experiments.scheduling.s2_run_dispatch --days 2 --strategies no_control mpc_model   # 快速试跑
"""

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.common import CAPACITY_KW, CLEANED_CSV, PREDICTIONS_DIR, SCHED_RESULT_DIR, SCHEDULE_CSV
from experiments.scheduling import config as C
from experiments.scheduling.settlement import settle
from experiments.scheduling.strategies import STRATEGIES, ErrorQuantiles, build_day, plan_mpc, plan_rule

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
FIG_DIR = SCHED_RESULT_DIR / "figures"


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------
def load_inputs(n_days: int | None):
    cleaned = pd.read_csv(CLEANED_CSV, parse_dates=["timestamp"], index_col="timestamp")
    pv_actual = cleaned["power_kw"].clip(lower=0)
    test = pd.read_csv(PREDICTIONS_DIR / "h1_test.csv", parse_dates=["timestamp"])
    test = test[test["step"] == 1].set_index("timestamp")
    pv_model = (test["y_pred"] * CAPACITY_KW).clip(lower=0)
    val = pd.read_csv(PREDICTIONS_DIR / "h1_val.csv")
    errq = ErrorQuantiles((val["y_pred"] * CAPACITY_KW).to_numpy(), (val["y_true"] * CAPACITY_KW).to_numpy())
    ev = pd.read_csv(SCHEDULE_CSV, parse_dates=["time"], index_col="time")["ev_kw"]

    # 只取有完整 96 个预测值、且前一天有实际光伏的日期
    dates = [d for d, n in pv_model.groupby(pv_model.index.normalize()).size().items()
             if n == 96 and (d - pd.Timedelta(days=1)) in pv_actual.index]
    dates = dates[:n_days] if n_days else dates
    days = [build_day(d, pv_actual, pv_model, ev) for d in dates]
    return days, errq


# ---------------------------------------------------------------------------
# 逐时段仿真
# ---------------------------------------------------------------------------
def simulate(args) -> pd.DataFrame:
    strategy, days, errq, position = args
    soc, rows = C.SOC_INIT, []
    pbar = tqdm(total=len(days) * 96, desc=f"{STRATEGIES[strategy]:<22}", position=position,
                leave=True, dynamic_ncols=True, mininterval=1.0)
    for day in days:
        served = 0.0  # 当天已完成的可平移充电电量
        for t in range(96):
            if strategy.startswith("mpc"):
                plan = plan_mpc(strategy, t, day, soc, served, errq)
            else:
                plan = plan_rule(strategy, t, day, soc)

            # 执行计划: 更新 SOC 和可平移充电进度, 再按实际光伏结算
            soc = soc + (C.ESS_ETA_CH * plan["ch"] - plan["dis"] / C.ESS_ETA_DIS) * C.DT / C.ESS_CAPACITY
            soc = min(max(soc, C.SOC_MIN), C.SOC_MAX)  # 去掉求解器的数值误差
            served += plan["sl"] * C.DT
            # 求解器结果有 1e-8 级的数值误差, 累计后可能让"已完成 > 已到达", 导致下一步模型无解
            served = min(max(served, day["flex_deadline"][t]), day["flex_arrived"][t])
            backlog = max(day["flex_arrived"][t] - served, 0.0)
            ev_fixed = (1 - C.FLEX_SHARE) * day["ev"][t]
            res = settle(plan, day["pv_actual"][t], ev_fixed, day["price"][t], backlog)

            rows.append({
                "time": day["time"][t], "price_buy": day["price"][t],
                "pv_actual_kw": day["pv_actual"][t], "pv_plan_kw": plan["pv_plan"],
                "ev_original_kw": day["ev"][t], "ev_served_kw": ev_fixed + plan["sl"] - plan["il"],
                "ess_ch_kw": plan["ch"], "ess_dis_kw": plan["dis"], "soc": soc,
                "il_kw": plan["il"], "flex_backlog_kwh": backlog, **res,
            })
            pbar.update(1)
    pbar.close()
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 汇总与画图
# ---------------------------------------------------------------------------
def summarize(steps: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in steps.items():
        pv_gen = df["pv_actual_kw"].sum() * C.DT
        local_pv = (df["pv_used_kw"] - df["sell_kw"]).clip(lower=0).sum() * C.DT
        err = df["pv_plan_kw"] - df["pv_actual_kw"]
        rows.append({
            "策略": STRATEGIES[name], "key": name,
            "总成本_元": df["cost_total"].sum(),
            "购电费_元": df["cost_buy"].sum(),
            "售电收入_元": -df["cost_sell"].sum(),
            "偏差考核_高估_元": df["cost_dev_over"].sum(),
            "偏差考核_低估_元": df["cost_dev_under"].sum(),
            "储能成本_元": df["cost_ess"].sum(),
            "柔性负荷补偿_元": df["cost_flex"].sum(),
            "弃光惩罚_元": df["cost_curt"].sum(),
            "缺电偏差_kWh": df["dev_over_kwh"].sum(),
            "多电偏差_kWh": df["dev_under_kwh"].sum(),
            "购电量_kWh": df["buy_kw"].sum() * C.DT,
            "上网电量_kWh": df["sell_kw"].sum() * C.DT,
            "光伏就地消纳率_%": local_pv / pv_gen * 100,
            "储能充电量_kWh": df["ess_ch_kw"].sum() * C.DT,
            "推迟充电_kWh·h": df["flex_backlog_kwh"].sum() * C.DT,
            "计划光伏RMSE_kW": float(np.sqrt(np.mean(err ** 2))),
            "计划光伏平均偏差_kW": float(err.mean()),
        })
    out = pd.DataFrame(rows)
    if "no_control" in out["key"].values:
        base = out.loc[out["key"] == "no_control", "总成本_元"].iloc[0]
        out["比无优化节省_%"] = (base - out["总成本_元"]) / abs(base) * 100
    return out


def plot_costs(summary: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    names = summary["策略"]
    axes[0].barh(names, summary["总成本_元"], color="#2E86AB")
    axes[0].set_title("27 天总运行成本 (元, 负数 = 净收益, 越低越好)")
    for i, v in enumerate(summary["总成本_元"]):
        axes[0].text(v, i, f" {v:,.0f}", va="center")
    axes[1].barh(names, summary["缺电偏差_kWh"], color="#C73E1D", label="缺电偏差 (光伏高估)")
    axes[1].barh(names, summary["多电偏差_kWh"], left=summary["缺电偏差_kWh"], color="#F18F01", label="多电偏差 (光伏低估)")
    axes[1].set_title("偏差电量 (kWh)")
    axes[1].legend(loc="lower right")
    for ax in axes:
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_day(steps: dict[str, pd.DataFrame], key: str, date: pd.Timestamp, out: Path) -> None:
    df = steps[key]
    d = df[df["time"].dt.normalize() == date]
    t = d["time"]
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    axes[0].plot(t, d["pv_actual_kw"], "k-", lw=1.4, label="实际光伏")
    axes[0].plot(t, d["pv_plan_kw"], "--", color="#2E86AB", label="调度使用的光伏 (预测/修正后)")
    axes[0].set_ylabel("kW")
    axes[1].plot(t, d["ev_original_kw"], color="gray", label="原始充电负荷")
    axes[1].plot(t, d["ev_served_kw"], color="#A23B72", label="调度后充电负荷")
    axes[1].set_ylabel("kW")
    axes[2].bar(t, d["ess_ch_kw"], width=0.008, color="#3B8E3B", label="储能充电")
    axes[2].bar(t, -d["ess_dis_kw"], width=0.008, color="#C73E1D", label="储能放电")
    ax2 = axes[2].twinx()
    ax2.plot(t, d["soc"] * 100, color="black", lw=1, label="SOC")
    ax2.set_ylabel("SOC (%)")
    axes[2].set_ylabel("kW")
    axes[3].plot(t, d["buy_kw"] - d["sell_kw"], color="#2E86AB", label="实际净购电 (负=上网)")
    axes[3].plot(t, d["declared_kw"], "--", color="#F18F01", label="申报净购电")
    ax3 = axes[3].twinx()
    ax3.step(t, d["price_buy"], where="post", color="gray", lw=0.8, label="购电价")
    ax3.set_ylabel("元/kWh")
    axes[3].set_ylabel("kW")
    for ax in axes:
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle(f"{STRATEGIES[key]}  {date.date()}")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="滚动调度与实际结算")
    parser.add_argument("--strategies", nargs="+", default=list(STRATEGIES), choices=list(STRATEGIES))
    parser.add_argument("--days", type=int, default=None, help="只跑前 N 天 (调试用)")
    args = parser.parse_args()

    days, errq = load_inputs(args.days)
    print(f"调度 {len(days)} 天: {days[0]['time'][0].date()} ~ {days[-1]['time'][0].date()}, "
          f"策略 {len(args.strategies)} 个 (并行运行)")
    tasks = [(s, days, errq, i) for i, s in enumerate(args.strategies)]
    with Pool(processes=len(tasks)) as pool:
        results = pool.map(simulate, tasks)
    steps = dict(zip(args.strategies, results))

    SCHED_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in steps.items():
        df.to_csv(SCHED_RESULT_DIR / f"steps_{name}.csv", index=False)
    summary = summarize(steps)
    summary.round(3).to_csv(SCHED_RESULT_DIR / "summary.csv", index=False, encoding="utf-8-sig")

    plot_costs(summary, FIG_DIR / "cost_comparison.png")
    # 典型日: 光伏波动最大 (预测最难) 的一天
    any_df = next(iter(steps.values()))
    volatility = any_df.groupby(any_df["time"].dt.normalize())["pv_actual_kw"].apply(lambda s: s.diff().abs().sum())
    for key in ("mpc_model_asym", "mpc_model"):
        if key in steps:
            plot_day(steps, key, volatility.idxmax(), FIG_DIR / f"typical_day_{key}.png")
            break

    cols = ["策略", "总成本_元", "比无优化节省_%", "偏差考核_高估_元", "偏差考核_低估_元",
            "缺电偏差_kWh", "多电偏差_kWh", "光伏就地消纳率_%", "计划光伏RMSE_kW"]
    print("\n" + summary[[c for c in cols if c in summary]].round(2).to_string(index=False))
    print(f"\n结果: {SCHED_RESULT_DIR}")


if __name__ == "__main__":
    main()
