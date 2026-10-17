"""
对比的调度策略, 以及每 15 分钟如何生成"当前时段的计划"。

  no_control      无优化: 储能不动作, 充电不推迟
  rule_tou        规则: 谷段满功率充电、峰段满功率放电 (模仿该站现在的运行方式)
  mpc_persist     滚动优化 + 持续性预测 (下一时刻光伏 = 当前时刻光伏)
  mpc_model       滚动优化 + CNN-BiLSTM 预测
  mpc_model_asym  滚动优化 + CNN-BiLSTM 预测 + 非对称风险修正 (开题创新点 2)
  mpc_oracle      滚动优化 + 完美预测 (知道未来真实光伏, 理论最优, 只作参照)

滚动优化 (MPC): 每个时段 t 都重新求解一次 [t, 当天结束] 的调度模型, 只执行第 t 个时段的结果。
  - 时段 t 的光伏: 用超短期预测 (提前 15 分钟, 即 H=1 模型)
  - t 之后的光伏: 超短期模型无法提供, 用"昨天同一时刻的实际光伏"作为日前估计

非对称风险修正: 光伏高估 (缺电) 罚得比低估 (多电) 重, 所以按偏低的分位数做计划。
  这是一个"报童问题": 设高估、低估每 kWh 的代价分别为 c_over、c_under, 最优计划值是预测分布的
  q = c_under / (c_over + c_under) 分位数。预测误差分布从验证集残差中估计 (按预测值大小分档)。
"""

import numpy as np
import pandas as pd

from experiments.scheduling import config as C
from experiments.scheduling.dispatch_model import DispatchInputs, solve_dispatch

STRATEGIES = {
    "no_control": "无优化",
    "rule_tou": "峰谷规则",
    "mpc_persist": "滚动优化+持续性预测",
    "mpc_model": "滚动优化+CNN-BiLSTM",
    "mpc_model_asym": "滚动优化+CNN-BiLSTM+非对称修正",
    "mpc_oracle": "滚动优化+完美预测(参照)",
}

SHIFT_STEPS = int(C.SHIFT_MAX_H / C.DT)


class ErrorQuantiles:
    """按预测值大小分档的预测误差分位数 (误差 = 实际 − 预测, kW)。"""

    EDGES = [10, 60, 150, 250]  # kW, 分成 5 档: 夜间/很小, 小, 中, 大, 很大

    def __init__(self, y_pred_kw: np.ndarray, y_true_kw: np.ndarray):
        bins = np.digitize(y_pred_kw, self.EDGES)
        self.residuals = [np.sort((y_true_kw - y_pred_kw)[bins == b]) for b in range(len(self.EDGES) + 1)]

    def adjust(self, forecast_kw: float, q: float) -> float:
        res = self.residuals[int(np.digitize(forecast_kw, self.EDGES))]
        return max(forecast_kw + float(np.quantile(res, q)), 0.0)


def risk_quantile(price_buy: float) -> float:
    c_over = C.OVER_PENALTY * price_buy
    c_under = C.UNDER_PENALTY * C.PRICE_SELL
    return c_under / (c_over + c_under)


def flex_arrays(ev: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """当天可平移充电的累计到达量, 以及各时段末必须完成的累计量 (kWh)。"""
    arrived = np.cumsum(C.FLEX_SHARE * ev * C.DT)
    deadline = np.concatenate([np.zeros(SHIFT_STEPS), arrived[:-SHIFT_STEPS]])
    return arrived, deadline


def plan_rule(strategy: str, t: int, day: dict, soc: float) -> dict:
    """非优化策略的当前时段计划。申报功率用持续性预测计算。"""
    ev = day["ev"][t]
    ch = dis = 0.0
    if strategy == "rule_tou":
        if day["price"][t] == C.PRICE_VALLEY:
            ch = min(C.ESS_P_MAX, (C.SOC_MAX - soc) * C.ESS_CAPACITY / (C.ESS_ETA_CH * C.DT))
        elif day["price"][t] == C.PRICE_PEAK:
            dis = min(C.ESS_P_MAX, (soc - C.SOC_MIN) * C.ESS_CAPACITY * C.ESS_ETA_DIS / C.DT)
    net = ev + ch - dis - day["pv_persist"][t]  # 充电不推迟, 全部负荷当时满足
    return {"buy": max(net, 0.0), "sell": max(-net, 0.0), "ch": max(ch, 0.0), "dis": max(dis, 0.0),
            "sl": C.FLEX_SHARE * ev, "il": 0.0, "pv_plan": day["pv_persist"][t]}


def plan_mpc(strategy: str, t: int, day: dict, soc: float, served: float, errq: ErrorQuantiles) -> dict:
    """滚动优化: 求解 [t, 当天结束] 的调度模型, 返回第 t 个时段的计划。"""
    if strategy == "mpc_oracle":
        pv = day["pv_actual"][t:].copy()
    else:
        pv = day["pv_dayahead"][t:].copy()
        if strategy == "mpc_persist":
            pv[0] = day["pv_persist"][t]
        elif strategy == "mpc_model":
            pv[0] = day["pv_model"][t]
        elif strategy == "mpc_model_asym":
            pv[0] = errq.adjust(day["pv_model"][t], risk_quantile(day["price"][t]))

    ev = day["ev"][t:]
    plan = solve_dispatch(DispatchInputs(
        pv=pv, ev_fixed=(1 - C.FLEX_SHARE) * ev, ev_total=ev,
        flex_arrived_cum=day["flex_arrived"][t:], flex_deadline_cum=day["flex_deadline"][t:],
        flex_served_before=served, price_buy=day["price"][t:], soc0=soc,
    ))
    first = {k: max(float(plan[k][0]), 0.0) for k in ("buy", "sell", "ch", "dis", "sl", "il")}  # 去掉 -1e-9 这类数值误差
    return first | {"pv_plan": float(pv[0])}


def build_day(date: pd.Timestamp, pv_actual: pd.Series, pv_model: pd.Series, ev: pd.Series) -> dict:
    """整理一天 96 个时段的输入数组。"""
    idx = pd.date_range(date, periods=96, freq="15min")
    prev_idx = idx - pd.Timedelta(days=1)
    day = {
        "time": idx,
        "pv_actual": pv_actual.reindex(idx).to_numpy(),
        "pv_model": pv_model.reindex(idx).to_numpy(),
        "pv_persist": pv_actual.reindex(idx - pd.Timedelta(minutes=15)).to_numpy(),
        "pv_dayahead": pv_actual.reindex(prev_idx).to_numpy(),
        "ev": ev.reindex(idx).fillna(0).to_numpy(),
        "price": np.array([C.buy_price(t.hour + t.minute / 60) for t in idx]),
    }
    day["flex_arrived"], day["flex_deadline"] = flex_arrays(day["ev"])
    return day
