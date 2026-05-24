"""Normalization and adapters for user-defined solution components.

This service is the C2 implementation step for the solution-component editor
roadmap.  It is intentionally UI-free: callers can create draft dictionaries or
``SolutionComponent`` objects, normalize them, and then convert only ready
components into existing analytical shapes.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Mapping, Sequence

from application.services.analysis_scope_profile_service import AnalysisScopeProfileService
from application.services.tco_model_service import TCOModelService
from domain import (
    AnalysisScope,
    CandidateConfiguration,
    CandidateConfigurationSource,
    ComponentType,
    CostModel,
    SolutionComponent,
    SolutionComponentNormalizationState,
    SolutionComponentOrigin,
    ensure_solution_component,
    to_plain_data,
)
from shared.constants import (
    ANALYSIS_SCOPE_SOFTWARE,
    ANALYSIS_SCOPE_TECHNICAL,
    LEGACY_INFRASTRUCTURE_SANDBOX_PREFIX,
)

_PROFILED_SCOPES = {ANALYSIS_SCOPE_TECHNICAL, ANALYSIS_SCOPE_SOFTWARE}
_ONE_TIME_COST_KEYS = (
    "purchase_cost",
    "implementation_cost",
    "migration_cost",
    "testing_cost",
)
_RECURRING_COST_KEYS = ("monthly_cost", "annual_cost", "energy_cost")
_ENERGY_COMPONENT_TYPES = {
    ComponentType.SERVER,
    ComponentType.WORKSTATION,
    ComponentType.PERIPHERAL,
    ComponentType.NETWORK_DEVICE,
}
_SOFTWARE_COMPONENT_TYPES = {
    ComponentType.SOFTWARE_LICENSE,
    ComponentType.SOFTWARE_SUBSCRIPTION,
    ComponentType.SOFTWARE_SERVICE,
}


class SolutionComponentNormalizationService:
    """Prepare ``SolutionComponent`` for existing analytics without touching UI.

    Normalization never mutates the source payload.  Incomplete drafts receive
    warnings and blocking errors instead of raising routine validation
    exceptions, so a future editor can save a draft safely and show exact
    reasons why it is not analytical input yet.
    """

    def __init__(
        self,
        *,
        profile_service: AnalysisScopeProfileService | None = None,
        tco_model_service: TCOModelService | None = None,
    ):
        self.profile_service = profile_service or AnalysisScopeProfileService()
        self.tco_model_service = tco_model_service or TCOModelService()

    def normalize(
        self,
        component: SolutionComponent | Mapping[str, Any],
    ) -> SolutionComponent:
        model = ensure_solution_component(deepcopy(component))
        warnings = list(model.validation_warnings)
        blocking_errors = list(model.blocking_errors)

        self._append_missing_identity_errors(model, blocking_errors)
        self._append_legacy_sandbox_warning(model, warnings, blocking_errors)
        self._append_scope_profile_warnings(model, warnings, blocking_errors)
        self._append_financial_warnings(model, warnings)

        tco_eligible = self._has_financial_data(model)
        energy_eligible = self._is_energy_eligible(model)
        candidate_eligible = (
            bool(model.strict_analysis_participation)
            and not blocking_errors
            and not self._is_legacy_sandbox(model)
        )
        analysis_ready = candidate_eligible and tco_eligible
        npv_eligible = analysis_ready and tco_eligible
        state = self._state_for(
            blocking_errors=blocking_errors,
            candidate_eligible=candidate_eligible,
            analysis_ready=analysis_ready,
        )

        metadata = deepcopy(model.metadata)
        metadata.setdefault("normalization", {})
        metadata["normalization"].update(
            {
                "profile_checked": bool(model.scope and model.scope.value in _PROFILED_SCOPES),
                "source_mutation": "not_mutated",
            }
        )
        normalized = SolutionComponent(
            id=model.id,
            name=model.name,
            scope=model.scope,
            component_type=model.component_type,
            origin=model.origin,
            strict_analysis_participation=model.strict_analysis_participation,
            purchase_cost=float(model.purchase_cost),
            implementation_cost=float(model.implementation_cost),
            migration_cost=float(model.migration_cost),
            testing_cost=float(model.testing_cost),
            monthly_cost=float(model.monthly_cost),
            annual_cost=float(model.annual_cost),
            energy_cost=float(model.energy_cost),
            quantity=float(model.quantity),
            metrics=deepcopy(model.metrics),
            constraints=deepcopy(model.constraints),
            cost_assumptions=list(model.cost_assumptions),
            metadata=metadata,
            validation_warnings=self._unique(warnings),
            blocking_errors=self._unique(blocking_errors),
            candidate_eligible=candidate_eligible,
            tco_eligible=tco_eligible,
            energy_eligible=energy_eligible,
            npv_eligible=npv_eligible,
            analysis_ready=analysis_ready,
            normalization_state=state,
        )
        return normalized

    def normalize_many(
        self,
        components: Iterable[SolutionComponent | Mapping[str, Any]],
    ) -> list[SolutionComponent]:
        return [self.normalize(component) for component in components]

    def to_candidate_configuration(
        self,
        components: Iterable[SolutionComponent | Mapping[str, Any]],
        *,
        candidate_id: str = "solution_components",
        name: str = "Компоненты решения",
        scope: str | AnalysisScope | None = None,
        source: str | CandidateConfigurationSource = CandidateConfigurationSource.MANUAL,
        horizon_months: int | float = TCOModelService.DEFAULT_HORIZON_MONTHS,
        electricity_profile: Mapping[str, Any] | None = None,
    ) -> CandidateConfiguration:
        normalized = self.normalize_many(components)
        ready_components = [component for component in normalized if component.candidate_eligible]
        component_snapshots = [self.to_component_payload(component) for component in ready_components]
        candidate_scope = self._resolve_candidate_scope(scope, ready_components)
        candidate = CandidateConfiguration(
            id=candidate_id,
            name=name,
            scope=candidate_scope,
            components=component_snapshots,
            totals=self._totals_from_components(component_snapshots),
            metrics=self._metrics_from_components(component_snapshots),
            source=self._source(source),
            metadata={
                "source_format": "solution_components",
                "input_component_count": len(normalized),
                "ready_component_count": len(ready_components),
                "blocked_component_ids": [
                    component.id for component in normalized if not component.candidate_eligible
                ],
            },
        )
        return self.tco_model_service.attach_to_candidate(
            candidate,
            horizon_months=horizon_months,
            electricity_profile=electricity_profile,
        )

    def to_cost_model(self, component: SolutionComponent | Mapping[str, Any]) -> CostModel:
        normalized = self.normalize(component)
        return self.tco_model_service.build_component_cost_model(self.to_component_payload(normalized))

    def to_energy_input(
        self,
        component: SolutionComponent | Mapping[str, Any],
        *,
        default_hours_per_day: float = 0.0,
        default_working_days: float = 0.0,
    ) -> dict[str, Any] | None:
        normalized = self.normalize(component)
        if not normalized.energy_eligible:
            return None
        metrics = normalized.metrics
        return {
            "id": normalized.id,
            "name": normalized.name,
            "quantity": self._number(metrics.get("quantity"), default=normalized.quantity),
            "max_power": self._number(metrics.get("max_power"), default=0.0),
            "hours_per_day": self._number(
                metrics.get("hours_per_day"),
                default=default_hours_per_day,
            ),
            "working_days": self._number(
                metrics.get("working_days"),
                default=default_working_days,
            ),
            "round_the_clock": bool(metrics.get("round_the_clock", False)),
            "source_component_id": normalized.id,
        }

    def to_decision_report_component_snapshot(
        self,
        component: SolutionComponent | Mapping[str, Any],
    ) -> dict[str, Any]:
        normalized = self.normalize(component)
        cost_model = self.to_cost_model(normalized)
        return {
            "id": normalized.id,
            "name": normalized.name,
            "scope": normalized.scope.value if normalized.scope else None,
            "component_type": (
                normalized.component_type.value if normalized.component_type else None
            ),
            "origin": normalized.origin.value,
            "source_format": "solution_component",
            "normalization_state": normalized.normalization_state.value,
            "editor_status": "strict" if normalized.candidate_eligible else "draft",
            "analysis_ready": normalized.analysis_ready,
            "candidate_eligible": normalized.candidate_eligible,
            "tco_eligible": normalized.tco_eligible,
            "energy_eligible": normalized.energy_eligible,
            "npv_eligible": normalized.npv_eligible,
            "validation_warnings": list(normalized.validation_warnings),
            "blocking_errors": list(normalized.blocking_errors),
            "cost_assumptions": list(normalized.cost_assumptions),
            "cost_model": to_plain_data(cost_model),
            "metadata": deepcopy(normalized.metadata),
        }

    def to_component_payload(
        self,
        component: SolutionComponent | Mapping[str, Any],
    ) -> dict[str, Any]:
        normalized = ensure_solution_component(component)
        metrics = deepcopy(normalized.metrics)
        financial = self._financial_payload(normalized)
        energy = self._energy_payload(normalized)
        payload = {
            "id": normalized.id,
            "name": normalized.name,
            "scope": normalized.scope.value if normalized.scope else None,
            "component_type": (
                normalized.component_type.value if normalized.component_type else None
            ),
            "origin": normalized.origin.value,
            "source_format": "solution_component",
            "normalization_state": normalized.normalization_state.value,
            "editor_status": "strict" if normalized.candidate_eligible else "draft",
            "quantity": normalized.quantity,
            "purchase_cost": normalized.purchase_cost,
            "implementation_cost": normalized.implementation_cost,
            "migration_cost": normalized.migration_cost,
            "testing_cost": normalized.testing_cost,
            "monthly_cost": normalized.monthly_cost,
            "annual_cost": normalized.annual_cost,
            "annual_cost_monthly_equivalent": normalized.annual_cost / 12.0,
            "energy_cost": normalized.energy_cost,
            "energy_applicable": energy["applicable"],
            "energy_source": energy["source"],
            "total_cost": normalized.purchase_cost,
            "one_time_cost": (
                normalized.implementation_cost
                + normalized.migration_cost
                + normalized.testing_cost
            ),
            "recurring_monthly_cost": normalized.monthly_cost + normalized.annual_cost / 12.0,
            "financial_fields": financial,
            "energy": energy,
            "metadata": deepcopy(normalized.metadata),
            "validation_warnings": list(normalized.validation_warnings),
            "blocking_errors": list(normalized.blocking_errors),
        }
        for key in (
            "category",
            "source_category",
            "max_power",
            "client_seats",
            "license_units",
            "support_score",
            "functionality_score",
            "performance_score",
            "reliability_score",
            "lifespan",
            "lifetime_years",
            "cpu_score",
            "ram_score",
            "perf",
            "energy",
        ):
            if key in metrics:
                payload[key] = deepcopy(metrics[key])
        if "perf" not in payload:
            payload["perf"] = self._number(
                metrics.get("performance_score", metrics.get("functionality_score")),
                default=0.0,
            )
        if "reliability" not in payload and "reliability_score" in metrics:
            payload["reliability"] = deepcopy(metrics["reliability_score"])
        if "lifespan" not in payload and "lifetime_years" in metrics:
            payload["lifespan"] = deepcopy(metrics["lifetime_years"])
        if "energy" not in payload and "max_power" in payload:
            payload["energy"] = self._number(payload.get("max_power"), default=0.0)
        payload["metrics"] = metrics
        return payload

    def build_legacy_sandbox_adapter(
        self,
        row: Mapping[str, Any],
        *,
        sandbox_category: str,
    ) -> SolutionComponent:
        """Manual adapter for future explicit conversion from legacy sandbox rows.

        The adapter marks converted data as sandbox-originated and blocked by
        default.  Future UI can ask the user to review it and then set strict
        participation explicitly.
        """

        payload = dict(deepcopy(row))
        payload.setdefault("id", f"{sandbox_category}:{payload.get('name', 'component')}")
        payload.setdefault("name", str(payload.get("title") or payload.get("name") or "Черновик"))
        payload.setdefault("origin", SolutionComponentOrigin.CONVERTED_SANDBOX.value)
        payload.setdefault("strict_analysis_participation", False)
        metadata = dict(payload.get("metadata", {})) if isinstance(payload.get("metadata"), Mapping) else {}
        metadata.update(
            {
                "_legacy_sandbox": True,
                "legacy_sandbox_category": sandbox_category,
            }
        )
        payload["metadata"] = metadata
        return self.normalize(payload)

    def _append_missing_identity_errors(
        self,
        model: SolutionComponent,
        blocking_errors: list[str],
    ) -> None:
        if not model.id:
            blocking_errors.append("component.id is required for stable analytics")
        if not model.name:
            blocking_errors.append("component.name is required")
        if model.scope is None:
            blocking_errors.append("component.scope is required")
        if model.component_type is None:
            blocking_errors.append("component.component_type is required")

    def _append_legacy_sandbox_warning(
        self,
        model: SolutionComponent,
        warnings: list[str],
        blocking_errors: list[str],
    ) -> None:
        if not self._is_legacy_sandbox(model):
            return
        warnings.append(
            "Запись помечена как legacy-песочница и не включается в строгую аналитику автоматически."
        )
        blocking_errors.append("legacy sandbox row requires explicit manual conversion")

    def _append_scope_profile_warnings(
        self,
        model: SolutionComponent,
        warnings: list[str],
        blocking_errors: list[str],
    ) -> None:
        if model.scope is None or model.component_type is None:
            return
        if model.scope.value not in _PROFILED_SCOPES:
            warnings.append(
                "Для области компонента нет строгого профиля ПО/ТО; компонент останется вспомогательным до отдельной интеграции."
            )
            if model.scope == AnalysisScope.MIXED and model.component_type == ComponentType.BUNDLE:
                return
            return
        profile = self.profile_service.get_profile(model.scope.value)
        if model.component_type.value not in profile.component_types:
            blocking_errors.append(
                f"component_type {model.component_type.value!r} is not allowed for scope {model.scope.value!r}"
            )
        if not model.strict_analysis_participation:
            warnings.append(
                "Компонент нормализован как черновик: strict_analysis_participation=false."
            )

    def _append_financial_warnings(
        self,
        model: SolutionComponent,
        warnings: list[str],
    ) -> None:
        if not self._has_financial_data(model):
            warnings.append(
                "Стоимость компонента не задана: TCO/NPV не будут считать его вклад."
            )
            return
        positive_fields = [
            key
            for key in (*_ONE_TIME_COST_KEYS, *_RECURRING_COST_KEYS)
            if self._number(getattr(model, key), default=0.0) > 0.0
        ]
        if len(positive_fields) == 1 and positive_fields[0] in {"purchase_cost", "energy_cost"}:
            warnings.append(
                "Все расходы компонента фактически указаны одной суммой; для точного TCO лучше разделить CAPEX, разовые работы и регулярные платежи."
            )
        if model.annual_cost and not model.monthly_cost:
            warnings.append(
                "annual_cost будет интерпретирован как ежегодный платёж и пересчитан в ежемесячный OPEX для TCO."
            )
        energy = self._energy_payload(model)
        if model.energy_cost and energy["source"] == "manual":
            warnings.append(
                "Энергетическая стоимость указана вручную и не связана с мощностью/профилем работы."
            )
        if model.energy_cost and not energy["applicable"]:
            warnings.append(
                "Для выбранного типа компонента энергетический расчёт не применим; energy_cost не должен подменять OPEX."
            )
        if model.scope == AnalysisScope.MIXED and model.energy_cost and not model.metrics.get("technical_part"):
            warnings.append(
                "Для mixed-компонента с энергетической частью нужно явно описать техническую часть или разнести компонент на части."
            )

    def _has_financial_data(self, model: SolutionComponent) -> bool:
        values = [getattr(model, key) for key in (*_ONE_TIME_COST_KEYS, *_RECURRING_COST_KEYS)]
        return any(float(value or 0.0) > 0.0 for value in values)

    def _is_energy_eligible(self, model: SolutionComponent) -> bool:
        energy = self._energy_payload(model)
        return bool(energy["applicable"] and energy["source"] in {"manual", "derived_ready"})

    def _financial_payload(self, model: SolutionComponent) -> dict[str, Any]:
        return {
            "purchase_cost": float(model.purchase_cost),
            "implementation_cost": float(model.implementation_cost),
            "migration_cost": float(model.migration_cost),
            "testing_cost": float(model.testing_cost),
            "monthly_cost": float(model.monthly_cost),
            "annual_cost": float(model.annual_cost),
            "annual_cost_monthly_equivalent": float(model.annual_cost) / 12.0,
            "energy_cost": float(model.energy_cost),
            "effect_source": "not_in_cost_model",
            "note": (
                "Пользовательский эффект или экономия не смешиваются с расходами; "
                "для NPV они передаются отдельным annual_effect."
            ),
        }

    def _energy_payload(self, model: SolutionComponent) -> dict[str, Any]:
        max_power = self._number(model.metrics.get("max_power"), default=0.0)
        hours_per_day = self._number(model.metrics.get("hours_per_day"), default=0.0)
        working_days = self._number(model.metrics.get("working_days"), default=0.0)
        quantity = self._number(model.metrics.get("quantity"), default=model.quantity)
        component_type = model.component_type
        is_software = component_type in _SOFTWARE_COMPONENT_TYPES
        is_technical = component_type in _ENERGY_COMPONENT_TYPES
        explicit_mixed = bool(model.metrics.get("technical_part") or model.metrics.get("energy_applicable"))
        applicable = (
            not is_software
            and (
                model.scope == AnalysisScope.TECHNICAL
                or is_technical
                or (model.scope == AnalysisScope.MIXED and explicit_mixed)
            )
        )
        if not applicable:
            source = "not_applicable"
        elif model.energy_cost > 0 and max_power <= 0:
            source = "manual"
        elif max_power > 0 and hours_per_day > 0 and working_days > 0:
            source = "derived_ready"
        elif model.energy_cost > 0:
            source = "manual"
        else:
            source = "missing_usage_profile"
        return {
            "applicable": bool(applicable),
            "source": source,
            "manual_monthly_cost": float(model.energy_cost),
            "max_power": max_power,
            "quantity": quantity,
            "hours_per_day": hours_per_day,
            "working_days": working_days,
        }

    def _is_legacy_sandbox(self, model: SolutionComponent) -> bool:
        metadata = model.metadata
        if bool(metadata.get("_legacy_sandbox", False)):
            return True
        for key in ("id",):
            value = getattr(model, key, "")
            if str(value).startswith(LEGACY_INFRASTRUCTURE_SANDBOX_PREFIX):
                return True
        for key in ("category", "source_category"):
            value = model.metrics.get(key) or metadata.get(key)
            if str(value).startswith(LEGACY_INFRASTRUCTURE_SANDBOX_PREFIX):
                return True
        return False

    def _state_for(
        self,
        *,
        blocking_errors: Sequence[str],
        candidate_eligible: bool,
        analysis_ready: bool,
    ) -> SolutionComponentNormalizationState:
        if blocking_errors:
            return SolutionComponentNormalizationState.BLOCKED
        if analysis_ready:
            return SolutionComponentNormalizationState.ANALYSIS_READY
        if candidate_eligible:
            return SolutionComponentNormalizationState.NORMALIZED
        return SolutionComponentNormalizationState.DRAFT

    def _resolve_candidate_scope(
        self,
        scope: str | AnalysisScope | None,
        components: Sequence[SolutionComponent],
    ) -> AnalysisScope | None:
        if isinstance(scope, AnalysisScope):
            return scope
        if scope:
            return AnalysisScope(str(scope))
        scopes = {component.scope for component in components if component.scope is not None}
        if len(scopes) == 1:
            return next(iter(scopes))
        if len(scopes) > 1:
            return AnalysisScope.MIXED
        return None

    def _totals_from_components(self, components: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "capital_cost": sum(self._number(item.get("purchase_cost"), default=0.0) for item in components),
            "one_time_cost": sum(self._number(item.get("one_time_cost"), default=0.0) for item in components),
            "monthly_cost": sum(self._number(item.get("monthly_cost"), default=0.0) for item in components),
            "annual_cost": sum(self._number(item.get("annual_cost"), default=0.0) for item in components),
            "energy_cost": sum(self._number(item.get("energy_cost"), default=0.0) for item in components),
            "component_count": len(components),
        }

    def _metrics_from_components(self, components: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "component_count": len(components),
            "energy_ready_count": sum(1 for item in components if item.get("max_power")),
            "software_license_quantity": sum(
                self._number(item.get("license_units"), default=0.0) for item in components
            ),
            "client_capacity": sum(
                self._number(item.get("client_seats"), default=0.0) for item in components
            ),
        }

    def _source(self, value: str | CandidateConfigurationSource) -> CandidateConfigurationSource:
        if isinstance(value, CandidateConfigurationSource):
            return value
        return CandidateConfigurationSource(str(value))

    def _number(self, value: Any, *, default: float) -> float:
        try:
            if value is None or value == "":
                return float(default)
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _unique(self, values: Sequence[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = str(value)
            if text not in seen:
                seen.add(text)
                result.append(text)
        return result


__all__ = ["SolutionComponentNormalizationService"]
