"""
源网荷储经济调度模型 (混合整数线性规划, MILP)

对应开题报告 1.3.2 节:
    目标 (1-5): min Σ_t [ C_grid(t) + C_ess(t) + C_curt(t) + C_FL(t) ]
        C_grid (1-6) = λ_buy·P_buy − λ_sell·P_sell
        C_ess  (1-7) = λ_ess·(P_ch·η_ch + P_dis/η_dis)
        C_curt (1-8) = λ_curt·P_curt             (P_curt = 预测可发功率 − 实际使用的光伏功率)
        C_FL   (1-9) = λ_SL·(推迟的充电电量·小时) + λ_IL·P_IL
    约束:
        (1-10) 功率平衡   PV − P_curt + P_dis + P_buy + P_IL = 固定负荷 + P_SL + P_ch + P_sell
        (1-11) 光伏出力   0 ≤ P_curt ≤ PV_预测
        (1-12) 储能       充放电功率上限、充放电互斥 (0-1 变量)、SOC 递推与上下限
        (1-13) 柔性负荷   可平移充电: 不能提前、最多推迟 SHIFT_MAX_H 小时、当天充完; 可削减负荷有上限
        (1-14) 联络线     购/售电功率上限, 购售电互斥 (0-1 变量), 只允许光伏电量上网
                          (谷段电价 0.30 低于上网电价 0.3844, 不加这两条模型会"从电网买电再卖回电网"套利)

说明: 服务区只有一个并网点, 这里用单节点功率平衡, 没有建配电网潮流 (开题中的二阶锥潮流需要网络参数)。

用法: plan = solve_dispatch(inputs)   # inputs 见 DispatchInputs
"""

from dataclasses import dataclass

import numpy as np
import pulp

from experiments.scheduling import config as C


@dataclass
class DispatchInputs:
    """从当前时段 t 到当天结束 (共 N 个时段) 的已知信息, 数组长度均为 N。"""
    pv: np.ndarray            # 光伏可发功率 (预测值), kW
    ev_fixed: np.ndarray      # 不可调整的充电负荷, kW
    ev_total: np.ndarray      # 充电总负荷 (可削减负荷按它的比例计算), kW
    flex_arrived_cum: np.ndarray  # 当天 0 点起到各时段末, 累计到达的可平移充电电量, kWh (长度 N)
    flex_deadline_cum: np.ndarray # 各时段末必须已经完成的累计可平移电量 (= SHIFT_MAX_H 小时前的累计到达量), kWh
    flex_served_before: float     # 当前时段之前已经完成的可平移电量, kWh
    price_buy: np.ndarray     # 购电价, 元/kWh
    soc0: float               # 当前时段开始时的 SOC (0-1)


def solve_dispatch(inp: DispatchInputs) -> dict:
    N = len(inp.pv)
    T = range(N)
    m = pulp.LpProblem("dispatch", pulp.LpMinimize)

    def var(name, ub=None, cat="Continuous"):
        return [pulp.LpVariable(f"{name}_{t}", lowBound=0, upBound=ub, cat=cat) for t in T]

    buy, sell = var("buy", C.GRID_P_MAX), var("sell", C.GRID_P_MAX)            # (1-14)
    ch, dis = var("ch", C.ESS_P_MAX), var("dis", C.ESS_P_MAX)
    u_ch, u_dis = var("u_ch", 1, "Binary"), var("u_dis", 1, "Binary")
    u_buy = var("u_buy", 1, "Binary")  # 1 = 购电, 0 = 售电
    soc = var("soc", C.SOC_MAX)
    curt = [pulp.LpVariable(f"curt_{t}", 0, max(float(inp.pv[t]), 0)) for t in T]  # (1-11)
    sl = var("sl", C.SL_P_MAX)
    il = [pulp.LpVariable(f"il_{t}", 0, C.IL_SHARE * float(inp.ev_total[t])) for t in T]
    backlog = var("backlog")  # 已到达但尚未完成的可平移充电电量 kWh

    # 目标函数 (1-5) ~ (1-9)
    m += pulp.lpSum(
        (inp.price_buy[t] * buy[t] - C.PRICE_SELL * sell[t]                       # (1-6)
         + C.ESS_COST * (C.ESS_ETA_CH * ch[t] + dis[t] * (1 / C.ESS_ETA_DIS))           # (1-7)
         + C.CURTAIL_PENALTY * curt[t]                                            # (1-8)
         + C.SL_COMPENSATION * backlog[t] + C.IL_COMPENSATION * il[t]) * C.DT     # (1-9)
        for t in T
    )

    served = inp.flex_served_before
    for t in T:
        # (1-10) 功率平衡
        m += (inp.pv[t] - curt[t] + dis[t] + buy[t] + il[t]
              == inp.ev_fixed[t] + sl[t] + ch[t] + sell[t]), f"balance_{t}"

        # (1-14) 联络线: 购售电互斥, 上网电量不超过光伏实际使用量
        m += buy[t] <= C.GRID_P_MAX * u_buy[t]
        m += sell[t] <= C.GRID_P_MAX * (1 - u_buy[t])
        m += sell[t] <= inp.pv[t] - curt[t]

        # (1-12) 储能: 充放电互斥、功率上限、SOC 递推
        m += ch[t] <= C.ESS_P_MAX * u_ch[t]
        m += dis[t] <= C.ESS_P_MAX * u_dis[t]
        m += u_ch[t] + u_dis[t] <= 1
        prev = inp.soc0 if t == 0 else soc[t - 1]
        m += soc[t] == prev + (C.ESS_ETA_CH * ch[t] - dis[t] * (1 / C.ESS_ETA_DIS)) * C.DT / C.ESS_CAPACITY
        m += soc[t] >= C.SOC_MIN

        # (1-13) 可平移充电: 累计完成量 ≤ 累计到达量 (不能提前), ≥ 截止量 (最多推迟 SHIFT_MAX_H 小时)
        served = served + sl[t] * C.DT
        m += served <= inp.flex_arrived_cum[t]
        m += served >= inp.flex_deadline_cum[t]
        m += backlog[t] == inp.flex_arrived_cum[t] - served

    m += soc[N - 1] >= C.SOC_END_MIN               # 当天结束时保留电量
    m += served == inp.flex_arrived_cum[N - 1]     # 当天的充电需求当天完成

    status = m.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"调度模型求解失败: {pulp.LpStatus[status]}")

    val = lambda xs: np.array([x.value() or 0.0 for x in xs])
    return {"buy": val(buy), "sell": val(sell), "ch": val(ch), "dis": val(dis), "soc": val(soc),
            "curt": val(curt), "sl": val(sl), "il": val(il), "objective": pulp.value(m.objective)}
