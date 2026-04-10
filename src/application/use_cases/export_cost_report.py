from __future__ import annotations

import logging
from pathlib import Path

from application.services.cost_aggregation_service import CostAggregationService
from application.use_cases.prepare_cost_summary import PrepareCostSummaryUseCase

logger = logging.getLogger(__name__)


class ExportCostReportUseCase:
    def __init__(
        self,
        prepare_cost_summary_use_case: PrepareCostSummaryUseCase,
        cost_aggregation_service: CostAggregationService,
    ):
        self.prepare_cost_summary_use_case = prepare_cost_summary_use_case
        self.cost_aggregation_service = cost_aggregation_service

    def execute(self, filename: str | Path, electricity_cost: float = 0.0) -> dict:
        logger.info("Подготовка экспортного отчёта: %s", filename)
        totals = self.prepare_cost_summary_use_case.execute(electricity_cost=electricity_cost)
        self.cost_aggregation_service.export_to_csv(filename)
        logger.info("Экспортный отчёт завершён: %s", filename)
        return totals
