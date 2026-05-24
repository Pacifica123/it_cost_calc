"""TCO model and NPV bridge for candidate configurations.

The service introduced at stage 5 of the conceptual-cohesion roadmap connects
CAPEX, OPEX, electricity and NPV without moving financial assumptions into UI
widgets.  It is intentionally tolerant to legacy dictionaries: old runtime rows,
AHP devices and GA selected items can all be summarized into the same cost
model.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

from domain import (
    AnalysisScope,
    CandidateConfiguration,
    ComponentType,
    CostModel,
    ensure_candidate_configuration,
    to_plain_data,
)
from shared.constants import ANALYSIS_SCOPE_SOFTWARE

_CAPITAL_CATEGORIES = {"server", "client", "network", "licenses"}
_IMPLEMENTATION_CATEGORIES = {"migration", "testing"}
_SUBSCRIPTION_CATEGORIES = {"subscription_licenses", "server_rental"}
_SUPPORT_CATEGORIES = {"backup", "labor_costs", "server_administration"}
_IMPLEMENTATION_TYPES = {ComponentType.IMPLEMENTATION_SERVICE.value}
_SUBSCRIPTION_TYPES = {
    ComponentType.SOFTWARE_SUBSCRIPTION.value,
    ComponentType.SOFTWARE_SERVICE.value,
}
_SUPPORT_TYPES = {ComponentType.SUPPORT_SERVICE.value, ComponentType.BACKUP_SERVICE.value}
_ENERGY_COMPONENT_TYPES = {
    ComponentType.SERVER.value,
    ComponentType.WORKSTATION.value,
    ComponentType.PERIPHERAL.value,
    ComponentType.NETWORK_DEVICE.value,
}
_SOFTWARE_COMPONENT_TYPES = {
    ComponentType.SOFTWARE_LICENSE.value,
    ComponentType.SOFTWARE_SUBSCRIPTION.value,
    ComponentType.SOFTWARE_SERVICE.value,
}


class TCOModelService:
    """Builds total-cost-of-ownership summaries and NPV input data."""

    DEFAULT_HORIZON_MONTHS = 12
    DEFAULT_NPV_HORIZON_YEARS = 5

    def build_component_cost_model(
        self,
        component: Mapping[str, Any],
        *,
        electricity_profile: Mapping[str, Any] | None = None,
    ) -> CostModel:
        payload = dict(component)
        category = str(
            payload.get("source_category")
            or payload.get("category")
            or payload.get("expense_category")
            or ""
        )
        component_type = str(payload.get("component_type") or "")

        purchase_cost = self._explicit_or_fallback_purchase_cost(payload, category=category)
        implementation_cost = self._number(payload.get("implementation_cost"), default=0.0)
        testing_cost = self._number(payload.get("testing_cost"), default=0.0)
        migration_cost = self._number(payload.get("migration_cost"), default=0.0)

        one_time_cost = self._number(payload.get("one_time_cost"), default=0.0)
        if one_time_cost and not any((implementation_cost, testing_cost, migration_cost)):
            if category == "testing":
                testing_cost = one_time_cost
            elif category == "migration":
                migration_cost = one_time_cost
            else:
                implementation_cost = one_time_cost

        monthly_cost = self._number(payload.get("monthly_cost"), default=0.0)
        annual_monthly_equivalent = self._number(
            payload.get("annual_cost_monthly_equivalent"),
            default=self._number(payload.get("annual_cost"), default=0.0) / 12.0,
        )
        recurring_monthly = monthly_cost + annual_monthly_equivalent

        subscription_cost = 0.0
        support_cost = 0.0
        if category in _SUBSCRIPTION_CATEGORIES or component_type in _SUBSCRIPTION_TYPES:
            subscription_cost = recurring_monthly
        elif category in _SUPPORT_CATEGORIES or component_type in _SUPPORT_TYPES:
            support_cost = recurring_monthly
        elif str(payload.get("scope") or "") == ANALYSIS_SCOPE_SOFTWARE:
            subscription_cost = recurring_monthly
        else:
            support_cost = recurring_monthly

        electricity_cost = 0.0
        if self._energy_applicable(payload):
            electricity_cost = self._explicit_electricity_cost(payload)
            if electricity_cost == 0.0 and electricity_profile is not None:
                electricity_cost = self._derived_monthly_electricity_cost(
                    payload,
                    electricity_profile=electricity_profile,
                )

        return CostModel(
            purchase_cost=purchase_cost,
            implementation_cost=implementation_cost,
            testing_cost=testing_cost,
            migration_cost=migration_cost,
            subscription_cost=subscription_cost,
            support_cost=support_cost,
            electricity_cost=electricity_cost,
        )

    def build_candidate_tco(
        self,
        candidate: CandidateConfiguration | Mapping[str, Any],
        *,
        horizon_months: int | float = DEFAULT_HORIZON_MONTHS,
        electricity_profile: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        model = ensure_candidate_configuration(candidate)
        component_models = [
            self.build_component_cost_model(component, electricity_profile=electricity_profile)
            for component in model.components
        ]
        aggregated = CostModel()
        for component_model in component_models:
            aggregated = aggregated.plus(component_model)

        return self._summary_from_cost_model(
            aggregated,
            horizon_months=horizon_months,
            component_count=len(model.components),
            component_cost_models=component_models,
        )

    def attach_to_candidate(
        self,
        candidate: CandidateConfiguration | Mapping[str, Any],
        *,
        horizon_months: int | float = DEFAULT_HORIZON_MONTHS,
        electricity_profile: Mapping[str, Any] | None = None,
    ) -> CandidateConfiguration:
        model = ensure_candidate_configuration(candidate)
        tco = self.build_candidate_tco(
            model,
            horizon_months=horizon_months,
            electricity_profile=electricity_profile,
        )
        totals = deepcopy(model.totals)
        totals["tco"] = tco
        totals.setdefault("total_ownership_cost", tco["total_ownership_cost"])
        totals.setdefault("initial_investment", tco["initial_investment"])
        totals.setdefault("monthly_opex", tco["monthly_opex"])
        totals.setdefault("annual_opex", tco["annual_opex"])
        metadata = deepcopy(model.metadata)
        metadata.setdefault("tco_horizon_months", int(float(horizon_months)))
        return CandidateConfiguration(
            id=model.id,
            name=model.name,
            scope=model.scope,
            components=deepcopy(model.components),
            totals=totals,
            metrics=deepcopy(model.metrics),
            source=model.source,
            metadata=metadata,
        )

    def from_cost_totals(
        self,
        totals: Mapping[str, Any],
        *,
        horizon_months: int | float = DEFAULT_HORIZON_MONTHS,
    ) -> dict[str, Any]:
        """Build a TCO summary from the current CAPEX/OPEX/electricity totals."""
        model = CostModel(
            purchase_cost=self._number(totals.get("total_capital"), default=0.0),
            implementation_cost=self._number(
                totals.get("total_operational_one_time"), default=0.0
            ),
            support_cost=self._number(totals.get("total_operational_monthly"), default=0.0),
            electricity_cost=self._number(totals.get("electricity_costs"), default=0.0),
        )
        return self._summary_from_cost_model(model, horizon_months=horizon_months)

    def build_npv_inputs(
        self,
        tco_summary: Mapping[str, Any],
        *,
        horizon_years: int = DEFAULT_NPV_HORIZON_YEARS,
        annual_effect: float = 0.0,
        discount_rate: float | None = None,
    ) -> dict[str, Any]:
        """Convert a TCO summary into NPV-ready investment and cash flows."""
        effective_horizon = max(int(horizon_years), 1)
        initial_investment = self._number(tco_summary.get("initial_investment"), default=0.0)
        annual_opex = self._number(tco_summary.get("annual_opex"), default=0.0)
        annual_effect_value = float(annual_effect)
        cash_flows = [annual_effect_value - annual_opex for _ in range(effective_horizon)]

        warnings: list[str] = []
        if annual_effect_value == 0.0:
            warnings.append(
                "Ожидаемый эффект или экономия не заданы: NPV отражает только оттоки TCO."
            )
        if annual_effect_value < annual_opex:
            warnings.append(
                "Годовой эффект меньше регулярных расходов: денежные потоки остаются отрицательными."
            )

        result = {
            "investment": initial_investment,
            "cash_flows": cash_flows,
            "horizon_years": effective_horizon,
            "annual_effect": annual_effect_value,
            "annual_opex": annual_opex,
            "cost_sources": {
                "capex": self._number(tco_summary.get("total_capex"), default=0.0),
                "one_time_costs": self._number(tco_summary.get("one_time_costs"), default=0.0),
                "monthly_opex": self._number(tco_summary.get("monthly_opex"), default=0.0),
                "annual_opex": annual_opex,
                "electricity_monthly_cost": self._number(
                    tco_summary.get("electricity_monthly_cost"), default=0.0
                ),
            },
            "warnings": warnings,
            "explanation": (
                "CAPEX и разовые внедренческие расходы используются как начальные инвестиции; "
                "OPEX и электроэнергия преобразуются в ежегодные регулярные оттоки; "
                "ожидаемый эффект задаётся отдельно и может быть изменён пользователем."
            ),
        }
        if discount_rate is not None:
            result["discount_rate"] = float(discount_rate)
        return result

    def build_npv_inputs_from_totals(
        self,
        totals: Mapping[str, Any],
        *,
        horizon_years: int = DEFAULT_NPV_HORIZON_YEARS,
        annual_effect: float = 0.0,
        discount_rate: float | None = None,
    ) -> dict[str, Any]:
        tco_summary = totals.get("tco") if isinstance(totals.get("tco"), Mapping) else None
        if tco_summary is None:
            tco_summary = self.from_cost_totals(totals, horizon_months=horizon_years * 12)
        else:
            tco_summary = deepcopy(dict(tco_summary))
        inputs = self.build_npv_inputs(
            tco_summary,
            horizon_years=horizon_years,
            annual_effect=annual_effect,
            discount_rate=discount_rate,
        )
        inputs["tco"] = tco_summary
        return inputs

    def _summary_from_cost_model(
        self,
        model: CostModel,
        *,
        horizon_months: int | float,
        component_count: int = 0,
        component_cost_models: Sequence[CostModel] | None = None,
    ) -> dict[str, Any]:
        effective_horizon = int(float(horizon_months))
        summary = {
            "horizon_months": effective_horizon,
            "component_count": component_count,
            "purchase_cost": float(model.purchase_cost),
            "implementation_cost": float(model.implementation_cost),
            "testing_cost": float(model.testing_cost),
            "migration_cost": float(model.migration_cost),
            "subscription_monthly_cost": float(model.subscription_cost),
            "support_monthly_cost": float(model.support_cost),
            "electricity_monthly_cost": float(model.electricity_cost),
            "total_capex": float(model.purchase_cost),
            "one_time_costs": float(
                model.implementation_cost + model.testing_cost + model.migration_cost
            ),
            "initial_investment": float(model.one_time_costs),
            "monthly_opex": float(model.monthly_costs),
            "annual_opex": float(model.annual_costs),
            "electricity_cost_period": float(model.electricity_cost) * effective_horizon,
            "total_ownership_cost": float(model.total_for_months(effective_horizon)),
        }
        if component_cost_models is not None:
            summary["component_cost_models"] = [
                to_plain_data(component_model) for component_model in component_cost_models
            ]
        return summary

    def _explicit_or_fallback_purchase_cost(
        self,
        payload: Mapping[str, Any],
        *,
        category: str,
    ) -> float:
        explicit = self._number(payload.get("purchase_cost"), default=0.0)
        if explicit:
            return explicit
        return self._purchase_cost(payload, category=category)

    def _purchase_cost(self, payload: Mapping[str, Any], *, category: str) -> float:
        total = self._number(
            payload.get("total_cost", payload.get("capital_cost")),
            default=0.0,
        )
        if total:
            return total
        if category in _CAPITAL_CATEGORIES or "price" in payload or "cost" in payload:
            quantity = self._number(payload.get("quantity"), default=1.0)
            price = self._number(
                payload.get("price", payload.get("unit_price", payload.get("cost"))),
                default=0.0,
            )
            return quantity * price
        return 0.0

    def _explicit_electricity_cost(self, payload: Mapping[str, Any]) -> float:
        for key in ("electricity_cost", "energy_cost", "monthly_electricity_cost"):
            value = self._number(payload.get(key), default=0.0)
            if value:
                return value
        return 0.0

    def _energy_applicable(self, payload: Mapping[str, Any]) -> bool:
        energy_block = payload.get("energy") if isinstance(payload.get("energy"), Mapping) else {}
        if "applicable" in energy_block:
            return bool(energy_block.get("applicable"))
        scope = str(payload.get("scope") or "")
        component_type = str(payload.get("component_type") or "")
        if component_type in _SOFTWARE_COMPONENT_TYPES or scope == ANALYSIS_SCOPE_SOFTWARE:
            return False
        if component_type in _ENERGY_COMPONENT_TYPES:
            return True
        if scope == "mixed":
            return bool(payload.get("max_power") or payload.get("energy_applicable"))
        return scope == "technical"

    def _derived_monthly_electricity_cost(
        self,
        payload: Mapping[str, Any],
        *,
        electricity_profile: Mapping[str, Any],
    ) -> float:
        scope = str(payload.get("scope") or "")
        if scope == ANALYSIS_SCOPE_SOFTWARE or not self._energy_applicable(payload):
            return 0.0
        max_power = self._number(payload.get("max_power"), default=0.0)
        if max_power <= 0:
            return 0.0
        quantity = self._number(payload.get("quantity"), default=1.0)
        hours_per_day = self._number(electricity_profile.get("hours_per_day"), default=0.0)
        working_days = self._number(
            electricity_profile.get("working_days"),
            default=self._number(electricity_profile.get("working_days_per_month"), default=0.0),
        )
        cost_per_kwh = self._number(electricity_profile.get("cost_per_kwh"), default=0.0)
        return (max_power / 1000.0) * quantity * hours_per_day * working_days * cost_per_kwh

    def _number(self, value: Any, *, default: float) -> float:
        try:
            if value is None or value == "":
                return float(default)
            return float(value)
        except (TypeError, ValueError):
            return float(default)


__all__ = ["TCOModelService"]
