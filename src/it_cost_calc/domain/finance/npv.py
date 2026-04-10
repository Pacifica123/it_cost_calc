from __future__ import annotations


def calculate_npv(investment: float, discount_rate: float, cash_flows: list[float]) -> float:
    npv = -investment
    for period, cash_flow in enumerate(cash_flows, start=1):
        npv += cash_flow / ((1 + discount_rate) ** period)
    return npv
