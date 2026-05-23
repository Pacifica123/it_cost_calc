"""Builds a UI-friendly NPV report without mixing plotting code and formulas."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from application.services.tco_model_service import TCOModelService
from domain.finance.npv import calculate_npv


class NPVReportService:
    def __init__(self, tco_model_service: TCOModelService | None = None):
        self.tco_model_service = tco_model_service or TCOModelService()

    def build_report(
        self,
        investment: float,
        discount_rate: float,
        cash_flows: list[float],
        *,
        financial_basis: Mapping[str, Any] | None = None,
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

        report = {
            "npv": calculate_npv(investment, discount_rate, cash_flows),
            "rows": rows,
            "accumulated_points": accumulated_points,
        }
        if financial_basis is not None:
            report["financial_basis"] = deepcopy(dict(financial_basis))
            report["warnings"] = list(financial_basis.get("warnings", []))
        return report

    def build_report_from_tco(
        self,
        tco_summary: Mapping[str, Any],
        *,
        discount_rate: float,
        horizon_years: int = TCOModelService.DEFAULT_NPV_HORIZON_YEARS,
        annual_effect: float = 0.0,
    ) -> dict:
        financial_basis = self.tco_model_service.build_npv_inputs(
            tco_summary,
            horizon_years=horizon_years,
            annual_effect=annual_effect,
            discount_rate=discount_rate,
        )
        return self.build_report(
            investment=financial_basis["investment"],
            discount_rate=discount_rate,
            cash_flows=list(financial_basis["cash_flows"]),
            financial_basis=financial_basis,
        )

    def build_financial_basis_from_totals(
        self,
        totals: Mapping[str, Any],
        *,
        horizon_years: int = TCOModelService.DEFAULT_NPV_HORIZON_YEARS,
        annual_effect: float = 0.0,
        discount_rate: float | None = None,
    ) -> dict[str, Any]:
        return self.tco_model_service.build_npv_inputs_from_totals(
            totals,
            horizon_years=horizon_years,
            annual_effect=annual_effect,
            discount_rate=discount_rate,
        )
