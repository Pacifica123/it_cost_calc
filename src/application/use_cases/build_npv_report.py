from __future__ import annotations

from application.services.npv_report_service import NPVReportService


class BuildNpvReportUseCase:
    def __init__(self, report_service: NPVReportService):
        self.report_service = report_service

    def execute(self, investment: float, discount_rate: float, cash_flows: list[float]) -> dict:
        return self.report_service.build_report(investment, discount_rate, cash_flows)
