"""Adapters for the common candidate-configuration format.

Stage 4 of the conceptual-cohesion roadmap introduces one application-level
shape for alternatives compared by AHP, GA, GA+AHP and criteria-importance
analysis.  This service keeps legacy dictionaries working while placing the
shared ``CandidateConfiguration`` contract next to them in payloads and exports.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Mapping, Sequence

from application.services.runtime_entity_normalization_service import normalize_runtime_row
from application.services.tco_model_service import TCOModelService
from domain import (
    AnalysisScope,
    CandidateConfiguration,
    CandidateConfigurationSource,
    as_analysis_scope,
    ensure_candidate_configuration,
    to_plain_data,
)
from domain.decision.ahp.aggregation import aggregate_configuration
from shared.constants import ANALYSIS_SCOPE_TECHNICAL, LEGACY_INFRASTRUCTURE_SANDBOX_PREFIX


class CandidateConfigurationService:
    """Builds and converts shared candidate alternatives.

    The methods deliberately return plain dictionaries for payload-facing APIs.
    Domain dataclasses are used inside the adapter so the project has one
    canonical contract, while old UI and exporter code can continue to consume
    JSON-like structures.
    """

    def __init__(self, tco_model_service: TCOModelService | None = None):
        self.tco_model_service = tco_model_service or TCOModelService()

    def from_ahp_configurations(
        self,
        configurations: Iterable[Mapping[str, Any]],
        *,
        scope: str | AnalysisScope | None = ANALYSIS_SCOPE_TECHNICAL,
        source: str | CandidateConfigurationSource = CandidateConfigurationSource.DEMO,
    ) -> list[CandidateConfiguration]:
        return [
            self.from_ahp_configuration(config, scope=scope, source=source)
            for config in configurations
        ]

    def from_ahp_configuration(
        self,
        configuration: Mapping[str, Any],
        *,
        scope: str | AnalysisScope | None = ANALYSIS_SCOPE_TECHNICAL,
        source: str | CandidateConfigurationSource = CandidateConfigurationSource.DEMO,
    ) -> CandidateConfiguration:
        config = deepcopy(dict(configuration))
        components = [deepcopy(dict(component)) for component in self._legacy_components(config)]
        aggregate = aggregate_configuration(config)
        metrics = self._metrics_from_aggregate(aggregate)
        totals = self._totals_from_aggregate(aggregate)
        metadata = {
            "legacy_format": "ahp_configuration",
            "legacy_meta": deepcopy(config.get("meta", {})),
            "aggregate": deepcopy(aggregate),
        }
        candidate = CandidateConfiguration(
            id=str(config.get("id") or config.get("name") or "candidate"),
            name=str(config.get("name") or config.get("id") or "Кандидатная конфигурация"),
            scope=self._scope(scope),
            components=components,
            totals=totals,
            metrics=metrics,
            source=self._source(source),
            metadata=metadata,
        )
        return self._with_tco(candidate)

    def from_ga_candidate_solutions(
        self,
        solutions: Iterable[Mapping[str, Any]],
        *,
        scope: str | AnalysisScope | None = ANALYSIS_SCOPE_TECHNICAL,
        source: str | CandidateConfigurationSource = CandidateConfigurationSource.GA,
    ) -> list[CandidateConfiguration]:
        result: list[CandidateConfiguration] = []
        for index, solution in enumerate(solutions, start=1):
            result.append(
                self.from_ga_candidate_solution(
                    solution,
                    scope=scope,
                    source=source,
                    fallback_rank=index,
                )
            )
        return result

    def from_ga_candidate_solution(
        self,
        solution: Mapping[str, Any],
        *,
        scope: str | AnalysisScope | None = ANALYSIS_SCOPE_TECHNICAL,
        source: str | CandidateConfigurationSource = CandidateConfigurationSource.GA,
        fallback_rank: int = 1,
    ) -> CandidateConfiguration:
        item = deepcopy(dict(solution))
        rank = int(item.get("rank") or fallback_rank)
        components = [deepcopy(dict(component)) for component in item.get("selected_items", [])]
        totals = deepcopy(dict(item.get("totals", {})))
        metrics = {
            "ga_score": item.get("score"),
            "raw_scores_by_criterion": deepcopy(item.get("raw_scores_by_criterion", {})),
            "directed_scores_by_criterion": deepcopy(item.get("directed_scores_by_criterion", {})),
            "normalized_scores_by_criterion": deepcopy(
                item.get("normalized_scores_by_criterion", {})
            ),
        }
        candidate = CandidateConfiguration(
            id=str(item.get("id") or f"GA-{rank}"),
            name=str(item.get("name") or f"GA-кандидат {rank}"),
            scope=self._scope(scope),
            components=components,
            totals=totals,
            metrics=metrics,
            source=self._source(source),
            metadata={
                "legacy_format": "ga_candidate_solution",
                "rank": rank,
                "mask": tuple(item.get("mask", ())),
                "selected_by_category": deepcopy(item.get("selected_by_category", {})),
                "criteria": deepcopy(item.get("criteria", [])),
                "constraints": deepcopy(item.get("constraints", [])),
            },
        )
        return self._with_tco(candidate)

    def from_runtime_entities(
        self,
        entities: Mapping[str, Sequence[Mapping[str, Any]]],
        *,
        scope: str | AnalysisScope | None = ANALYSIS_SCOPE_TECHNICAL,
        candidate_id: str = "runtime_current",
        name: str = "Текущий runtime-набор",
        source: str | CandidateConfigurationSource = CandidateConfigurationSource.MANUAL,
    ) -> CandidateConfiguration:
        components: list[dict[str, Any]] = []
        for category, rows in entities.items():
            category_name = str(category)
            if category_name.startswith(LEGACY_INFRASTRUCTURE_SANDBOX_PREFIX):
                continue
            for index, row in enumerate(rows):
                component = normalize_runtime_row(row, category=category_name)
                component.setdefault("id", f"{category_name}:{index}")
                component.setdefault("source_category", category_name)
                components.append(component)
        totals = self._totals_from_components(components)
        candidate = CandidateConfiguration(
            id=candidate_id,
            name=name,
            scope=self._scope(scope),
            components=components,
            totals=totals,
            metrics={},
            source=self._source(source),
            metadata={"legacy_format": "runtime_entities"},
        )
        return self._with_tco(candidate)

    def to_ahp_configurations(
        self,
        candidates: Iterable[CandidateConfiguration | Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        return [self.to_ahp_configuration(candidate) for candidate in candidates]

    def to_ahp_configuration(
        self,
        candidate: CandidateConfiguration | Mapping[str, Any],
    ) -> dict[str, Any]:
        model = ensure_candidate_configuration(candidate)
        legacy_meta = deepcopy(model.metadata.get("legacy_meta", {}))
        people = int(float(model.metrics.get("people", legacy_meta.get("people", 0)) or 0))
        return {
            "id": model.id,
            "name": model.name,
            "devices": [self._component_to_ahp_device(component) for component in model.components],
            "meta": {**legacy_meta, "people": people},
        }

    def attach_to_criteria_case(
        self,
        case_data: Mapping[str, Any],
        candidates: Iterable[CandidateConfiguration | Mapping[str, Any]],
    ) -> dict[str, Any]:
        result = deepcopy(dict(case_data))
        candidate_payloads = self.payload(candidates)
        if candidate_payloads:
            result["candidate_configurations"] = candidate_payloads
            score_ids = set(result.get("scores", {})) if isinstance(result.get("scores"), Mapping) else set()
            candidate_ids = {str(item["id"]) for item in candidate_payloads}
            if candidate_ids and (not score_ids or candidate_ids.issubset(score_ids)):
                result["alternatives"] = [
                    {"id": item["id"], "name": item["name"]} for item in candidate_payloads
                ]
        return result

    def payload(
        self,
        candidates: Iterable[CandidateConfiguration | Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        return [to_plain_data(ensure_candidate_configuration(candidate)) for candidate in candidates]

    def _legacy_components(self, configuration: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        components = configuration.get("components")
        if isinstance(components, list):
            return [component for component in components if isinstance(component, Mapping)]
        devices = configuration.get("devices", [])
        if isinstance(devices, list):
            return [device for device in devices if isinstance(device, Mapping)]
        return []

    def _metrics_from_aggregate(self, aggregate: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "total_performance": aggregate.get("total_performance", 0.0),
            "avg_reliability": aggregate.get("avg_reliability", 0.0),
            "lifespan": aggregate.get("lifespan", 0.0),
            "counts": deepcopy(aggregate.get("counts", {})),
            "people": aggregate.get("people", 0),
        }

    def _totals_from_aggregate(self, aggregate: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "capital_cost": aggregate.get("total_cost", 0.0),
            "total_cost": aggregate.get("total_cost", 0.0),
            "energy": aggregate.get("total_energy", 0.0),
        }

    def _totals_from_components(self, components: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        capital_cost = 0.0
        monthly_cost = 0.0
        one_time_cost = 0.0
        for component in components:
            quantity = self._number(component.get("quantity"), default=1.0)
            price = self._number(component.get("price", component.get("unit_price")), default=0.0)
            capital_cost += self._number(
                component.get("total_cost", component.get("capital_cost")),
                default=quantity * price,
            )
            monthly_cost += self._number(component.get("monthly_cost"), default=0.0)
            one_time_cost += self._number(component.get("one_time_cost"), default=0.0)
        return {
            "capital_cost": capital_cost,
            "monthly_cost": monthly_cost,
            "one_time_cost": one_time_cost,
            "component_count": len(components),
        }

    def _component_to_ahp_device(self, component: Mapping[str, Any]) -> dict[str, Any]:
        device = deepcopy(dict(component))
        device.setdefault("role", device.get("source_category", device.get("category", "unknown")))
        if "cost" not in device:
            quantity = self._number(device.get("quantity"), default=1.0)
            price = self._number(device.get("price", device.get("unit_price")), default=0.0)
            device["cost"] = self._number(device.get("total_cost"), default=quantity * price)
        if "energy" not in device and "max_power" in device:
            device["energy"] = self._number(device.get("max_power"), default=0.0)
        return device

    def _with_tco(self, candidate: CandidateConfiguration) -> CandidateConfiguration:
        return self.tco_model_service.attach_to_candidate(candidate)

    def _scope(self, value: str | AnalysisScope | None) -> AnalysisScope | None:
        if value is None:
            return None
        return as_analysis_scope(value)

    def _source(self, value: str | CandidateConfigurationSource) -> CandidateConfigurationSource:
        if isinstance(value, CandidateConfigurationSource):
            return value
        return CandidateConfigurationSource(str(value))

    def _number(self, value: Any, *, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)


__all__ = ["CandidateConfigurationService"]
