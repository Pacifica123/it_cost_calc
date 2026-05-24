"""Financial integration for SolutionComponent rows.

The service connects the advanced component editor to the existing financial
chain without creating a parallel calculator.  It accepts normalized or raw
``SolutionComponent`` rows, keeps drafts out of strict totals, builds TCO through
``TCOModelService`` and prepares the same financial basis that the NPV tab uses.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

from application.services.solution_component_normalization_service import (
    SolutionComponentNormalizationService,
)
from application.services.tco_model_service import TCOModelService
from domain import SolutionComponent, to_plain_data


class SolutionComponentFinancialIntegrationService:
    """Build cost, TCO and NPV inputs from strict SolutionComponent rows."""

    DEFAULT_HORIZON_MONTHS = 36
    DEFAULT_HORIZON_YEARS = 5

    def __init__(
        self,
        *,
        normalization_service: SolutionComponentNormalizationService | None = None,
        tco_model_service: TCOModelService | None = None,
    ):
        self.tco_model_service = tco_model_service or TCOModelService()
        self.normalization_service = normalization_service or SolutionComponentNormalizationService(
            tco_model_service=self.tco_model_service
        )

    def build_financial_chain(
        self,
        components: Sequence[SolutionComponent | Mapping[str, Any]],
        *,
        candidate_id: str = "solution_components_financial",
        candidate_name: str = "Финансовая база компонентов редактора",
        horizon_months: int | float = DEFAULT_HORIZON_MONTHS,
        horizon_years: int = DEFAULT_HORIZON_YEARS,
        annual_effect: float = 0.0,
        discount_rate: float | None = None,
        electricity_profile: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return the editor component financial chain.

        The returned payload is intentionally report-friendly: it includes the
        strict component count, excluded drafts, per-component cost models, the
        aggregated TCO and NPV-ready financial basis.
        """

        normalized = self.normalization_service.normalize_many(components)
        strict_components = [
            component
            for component in normalized
            if component.candidate_eligible and component.tco_eligible
        ]
        excluded_components = [
            component
            for component in normalized
            if component not in strict_components
        ]
        candidate = self.normalization_service.to_candidate_configuration(
            strict_components,
            candidate_id=candidate_id,
            name=candidate_name,
            horizon_months=horizon_months,
            electricity_profile=electricity_profile,
        )
        candidate.metadata.update(
            {
                "financial_chain": "cost_to_tco_to_npv",
                "input_component_count": len(normalized),
                "strict_financial_component_count": len(strict_components),
                "excluded_component_ids": [component.id for component in excluded_components],
            }
        )
        tco = deepcopy(candidate.totals.get("tco", {}))
        npv_inputs = self.tco_model_service.build_npv_inputs(
            tco,
            horizon_years=horizon_years,
            annual_effect=annual_effect,
            discount_rate=discount_rate,
        )
        component_financial_rows = self._component_financial_rows(
            strict_components,
            electricity_profile=electricity_profile,
        )
        warnings = self._warnings(
            normalized=normalized,
            strict_components=strict_components,
            npv_inputs=npv_inputs,
            tco=tco,
        )
        return {
            "schema_version": 1,
            "chain": "solution_component_cost_to_tco_to_npv",
            "horizon_months": int(float(horizon_months)),
            "horizon_years": max(int(horizon_years), 1),
            "annual_effect": float(annual_effect),
            "discount_rate": None if discount_rate is None else float(discount_rate),
            "component_count": len(normalized),
            "strict_financial_component_count": len(strict_components),
            "excluded_component_count": len(excluded_components),
            "excluded_component_ids": [component.id for component in excluded_components],
            "candidate": candidate.to_dict(),
            "tco": tco,
            "npv_inputs": npv_inputs,
            "component_financial_rows": component_financial_rows,
            "warnings": warnings,
        }

    def _component_financial_rows(
        self,
        components: Sequence[SolutionComponent],
        *,
        electricity_profile: Mapping[str, Any] | None,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for component in components:
            payload = self.normalization_service.to_component_payload(component)
            cost_model = self.tco_model_service.build_component_cost_model(
                payload,
                electricity_profile=electricity_profile,
            )
            energy = payload.get("energy") if isinstance(payload.get("energy"), Mapping) else {}
            rows.append(
                {
                    "id": component.id,
                    "name": component.name,
                    "scope": component.scope.value if component.scope else None,
                    "component_type": component.component_type.value if component.component_type else None,
                    "cost_model": to_plain_data(cost_model),
                    "energy_applicable": bool(energy.get("applicable", False)),
                    "energy_source": energy.get("source", "not_applicable"),
                    "warnings": list(component.validation_warnings),
                }
            )
        return rows

    def _warnings(
        self,
        *,
        normalized: Sequence[SolutionComponent],
        strict_components: Sequence[SolutionComponent],
        npv_inputs: Mapping[str, Any],
        tco: Mapping[str, Any],
    ) -> list[str]:
        warnings: list[str] = []
        if normalized and not strict_components:
            warnings.append(
                "Нет строгих компонентов редактора с финансовыми данными: TCO/NPV не получают вклад SolutionComponent."
            )
        for component in normalized:
            prefix = f"{component.name or component.id}: "
            for warning in component.validation_warnings:
                warnings.append(prefix + str(warning))
            for error in component.blocking_errors:
                warnings.append(prefix + str(error))
        for warning in npv_inputs.get("warnings", []) or []:
            warnings.append(str(warning))
        if self._number(tco.get("initial_investment"), default=0.0) > 0 and self._number(
            npv_inputs.get("annual_effect"), default=0.0
        ) <= 0:
            warnings.append(
                "NPV получил расходы компонентов, но не получил отдельный ожидаемый эффект или экономию."
            )
        return list(dict.fromkeys(warnings))

    def _number(self, value: Any, *, default: float) -> float:
        try:
            if value is None or value == "":
                return float(default)
            return float(value)
        except (TypeError, ValueError):
            return float(default)


__all__ = ["SolutionComponentFinancialIntegrationService"]
