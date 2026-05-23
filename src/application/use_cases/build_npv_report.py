from __future__ import annotations

from typing import Any, Mapping

from application.services.npv_report_service import NPVReportService


class BuildNpvReportUseCase:
    def __init__(self, report_service: NPVReportService):
        self.report_service = report_service

    def execute(
        self,
        investment: float,
        discount_rate: float,
        cash_flows: list[float],
        *,
        financial_basis: Mapping[str, Any] | None = None,
    ) -> dict:
        return self.report_service.build_report(
            investment,
            discount_rate,
            cash_flows,
            financial_basis=financial_basis,
        )

    def execute_from_tco(
        self,
        tco_summary: Mapping[str, Any],
        *,
        discount_rate: float,
        horizon_years: int = 5,
        annual_effect: float = 0.0,
    ) -> dict:
        return self.report_service.build_report_from_tco(
            tco_summary,
            discount_rate=discount_rate,
            horizon_years=horizon_years,
            annual_effect=annual_effect,
        )

    def prepare_from_totals(
        self,
        totals: Mapping[str, Any],
        *,
        horizon_years: int = 5,
        annual_effect: float = 0.0,
        discount_rate: float | None = None,
    ) -> dict[str, Any]:
        return self.report_service.build_financial_basis_from_totals(
            totals,
            horizon_years=horizon_years,
            annual_effect=annual_effect,
            discount_rate=discount_rate,
        )
