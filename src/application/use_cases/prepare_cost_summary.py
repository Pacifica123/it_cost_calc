from __future__ import annotations

from application.services.cost_aggregation_service import CostAggregationService
from application.services.equipment_service import EquipmentService


class PrepareCostSummaryUseCase:
    def __init__(
        self, cost_aggregation_service: CostAggregationService, equipment_service: EquipmentService
    ):
        self.cost_aggregation_service = cost_aggregation_service
        self.equipment_service = equipment_service

    def execute(self, electricity_cost: float = 0.0) -> dict:
        self.cost_aggregation_service.update_capital_costs(self.equipment_service)
        self.cost_aggregation_service.update_operational_costs(self.equipment_service)
        self.cost_aggregation_service.update_electricity_costs(electricity_cost)
        return self.cost_aggregation_service.calculate_totals()
