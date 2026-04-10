from __future__ import annotations

from application.services.electricity_cost_service import ElectricityCostService
from domain import ElectricityProfile


class CalculateElectricityCostsUseCase:
    def __init__(self, electricity_cost_service: ElectricityCostService):
        self.electricity_cost_service = electricity_cost_service

    def execute(
        self,
        equipment_rows: list[ElectricityProfile | dict],
        hours_per_day: float,
        working_days: float,
        cost_per_kwh: float,
        round_the_clock_names: set[str] | None = None,
    ) -> dict:
        return self.electricity_cost_service.calculate(
            equipment_rows=equipment_rows,
            hours_per_day=hours_per_day,
            working_days=working_days,
            cost_per_kwh=cost_per_kwh,
            round_the_clock_names=round_the_clock_names,
        )
