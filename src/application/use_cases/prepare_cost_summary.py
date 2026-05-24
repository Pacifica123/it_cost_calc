from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from application.services.cost_aggregation_service import CostAggregationService
from application.services.equipment_service import EquipmentService
from application.services.solution_component_runtime_service import SolutionComponentRuntimeService


class PrepareCostSummaryUseCase:
    def __init__(
        self,
        cost_aggregation_service: CostAggregationService,
        equipment_service: EquipmentService,
        solution_component_runtime_service: SolutionComponentRuntimeService | None = None,
    ):
        self.cost_aggregation_service = cost_aggregation_service
        self.equipment_service = equipment_service
        self.solution_component_runtime_service = solution_component_runtime_service

    def execute(
        self,
        electricity_cost: float = 0.0,
        *,
        electricity_profile: Mapping[str, Any] | None = None,
    ) -> dict:
        self.cost_aggregation_service.update_capital_costs(self.equipment_service)
        self.cost_aggregation_service.update_operational_costs(self.equipment_service)
        self.cost_aggregation_service.update_electricity_costs(electricity_cost)
        totals = self.cost_aggregation_service.calculate_totals()
        return self._with_solution_component_financials(
            totals,
            electricity_profile=electricity_profile,
        )

    def _with_solution_component_financials(
        self,
        totals: Mapping[str, Any],
        *,
        electricity_profile: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        payload = {str(key): deepcopy(value) for key, value in dict(totals).items()}
        if self.solution_component_runtime_service is None:
            return payload

        chain = self.solution_component_runtime_service.build_financial_chain(
            electricity_profile=electricity_profile,
        )
        payload["solution_component_financial_chain"] = chain
        if int(chain.get("strict_financial_component_count", 0)) <= 0:
            return payload

        tco = chain.get("tco") if isinstance(chain.get("tco"), Mapping) else {}
        solution_capex = self._number(tco.get("total_capex"), default=0.0)
        solution_one_time = self._number(tco.get("one_time_costs"), default=0.0)
        solution_electricity = self._number(tco.get("electricity_monthly_cost"), default=0.0)
        solution_monthly_opex = max(
            self._number(tco.get("monthly_opex"), default=0.0) - solution_electricity,
            0.0,
        )

        payload["total_capital"] = self._number(payload.get("total_capital"), default=0.0) + solution_capex
        payload["total_operational_one_time"] = (
            self._number(payload.get("total_operational_one_time"), default=0.0)
            + solution_one_time
        )
        payload["total_operational_monthly"] = (
            self._number(payload.get("total_operational_monthly"), default=0.0)
            + solution_monthly_opex
        )
        payload["electricity_costs"] = (
            self._number(payload.get("electricity_costs"), default=0.0)
            + solution_electricity
        )
        payload["solution_component_financial_totals"] = {
            "total_capex": solution_capex,
            "one_time_costs": solution_one_time,
            "monthly_opex_without_electricity": solution_monthly_opex,
            "electricity_monthly_cost": solution_electricity,
            "total_ownership_cost": self._number(tco.get("total_ownership_cost"), default=0.0),
        }
        payload["tco"] = self.cost_aggregation_service.tco_model_service.from_cost_totals(payload)
        return payload

    def _number(self, value: Any, *, default: float) -> float:
        try:
            if value is None or value == "":
                return float(default)
            return float(value)
        except (TypeError, ValueError):
            return float(default)
