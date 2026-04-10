"""Builds a UI-friendly NPV report without mixing plotting code and formulas."""

from __future__ import annotations

from it_cost_calc.domain.finance.npv import calculate_npv


class NPVReportService:
    def build_report(
        self, investment: float, discount_rate: float, cash_flows: list[float]
    ) -> dict:
        rows: list[dict] = [
            {
                "year": 0,
                "investment": investment,
                "cash_flow": -investment,
                "discount_factor": None,
                "present_value": -investment,
                "accumulated_npv": -investment,
            }
        ]

        accumulated_npv = -investment
        accumulated_points = [accumulated_npv]

        for year, cash_flow in enumerate(cash_flows, start=1):
            discount_factor = 1 / ((1 + discount_rate) ** year)
            present_value = cash_flow * discount_factor
            accumulated_npv += present_value
            accumulated_points.append(accumulated_npv)
            rows.append(
                {
                    "year": year,
                    "investment": 0.0,
                    "cash_flow": cash_flow,
                    "discount_factor": discount_factor,
                    "present_value": present_value,
                    "accumulated_npv": accumulated_npv,
                }
            )

        return {
            "npv": calculate_npv(investment, discount_rate, cash_flows),
            "rows": rows,
            "accumulated_points": accumulated_points,
        }
