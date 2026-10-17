"""
实时执行与结算: 计划是按"预测光伏"做的, 这里用"实际光伏"算出真正发生的功率和费用。

执行规则 (每 15 分钟):
  - 储能、可平移充电、可削减负荷严格按计划执行
  - 实际光伏全部使用; 与预测的差额由电网自动补足/吸收
  - 上网功率超过联络线上限时, 超出部分弃光

费用:
  电费      = 购电价 x 实际购电量 − 上网电价 x 实际上网电量
  偏差考核  = 实际净购电 > 申报 (光伏高估, 缺电) : 购电价 x OVER_PENALTY  x 偏差电量
            + 实际净购电 < 申报 (光伏低估, 多电) : 上网电价 x UNDER_PENALTY x 偏差电量
  储能成本、弃光惩罚、柔性负荷补偿 同调度模型 (式 1-7 ~ 1-9)
"""

from experiments.scheduling import config as C


def settle(plan: dict, pv_actual: float, ev_fixed: float, price_buy: float, backlog: float) -> dict:
    """
    plan:     当前时段的计划 (buy, sell, ch, dis, sl, il, 单位 kW)
    backlog:  执行本时段计划之后, 仍在等待的可平移充电电量 (kWh)
    """
    declared = plan["buy"] - plan["sell"]  # 申报的净购电功率 (负数表示上网)
    net = ev_fixed + plan["sl"] + plan["ch"] - plan["dis"] - plan["il"] - pv_actual  # 实际净购电
    curt = 0.0
    if net < -C.GRID_P_MAX:          # 上网超限 -> 弃光
        curt, net = -C.GRID_P_MAX - net, -C.GRID_P_MAX
    buy, sell = max(net, 0.0), max(-net, 0.0)
    over, under = max(net - declared, 0.0), max(declared - net, 0.0)

    cost = {
        "cost_buy": price_buy * buy * C.DT,
        "cost_sell": -C.PRICE_SELL * sell * C.DT,
        "cost_dev_over": C.OVER_PENALTY * price_buy * over * C.DT,
        "cost_dev_under": C.UNDER_PENALTY * C.PRICE_SELL * under * C.DT,
        "cost_ess": C.ESS_COST * (C.ESS_ETA_CH * plan["ch"] + plan["dis"] / C.ESS_ETA_DIS) * C.DT,
        "cost_curt": C.CURTAIL_PENALTY * curt * C.DT,
        "cost_flex": (C.SL_COMPENSATION * backlog + C.IL_COMPENSATION * plan["il"]) * C.DT,
    }
    cost["cost_total"] = sum(cost.values())
    return {
        "buy_kw": buy, "sell_kw": sell, "declared_kw": declared,
        "dev_over_kwh": over * C.DT, "dev_under_kwh": under * C.DT,
        "curt_kw": curt, "pv_used_kw": pv_actual - curt,
        **cost,
    }
