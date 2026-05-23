"""Application service for running genetic optimization over runtime entities.

The service is intentionally an adapter over the generic GA core. It knows how to
read the current runtime entity storage and how to convert rows into generic
``Item`` candidates, but the optimization logic still receives arbitrary
criteria and constraints through the universal GA contract.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Callable, Mapping, Protocol, Sequence

from application.services.analysis_scope_profile_service import (
    AnalysisScopeProfile,
    AnalysisScopeProfileService,
)
from application.services.runtime_entity_normalization_service import normalize_runtime_row
from shared.constants import CAPITAL_COST_CATEGORIES

logger = logging.getLogger(__name__)

CriterionSpec = Mapping[str, Any]
ConstraintSpec = Mapping[str, Any]
GaRunner = Callable[[dict[str, Any]], dict[str, Any]]


class RuntimeOptimizationItem:
    """Minimal item object accepted by the generic GA core."""

    def __init__(self, **properties: Any):
        self.properties = dict(properties)


class RuntimeEntitiesSource(Protocol):
    @property
    def entities(self) -> Mapping[str, Sequence[Mapping[str, Any]]]: ...


class GeneticOptimizationService:
    """Prepares runtime equipment data and runs the generic GA core.

    The class does not extend the GA with IT-specific rules. IT-infrastructure
    assumptions live here as defaults that can be replaced by caller-provided
    criteria and constraints.
    """

    DEFAULT_GA_PARAMS: dict[str, Any] = {
        "pop_size": 40,
        "generations": 80,
        "mutation_rate": 0.04,
        "elite": 2,
        "seed": 42,
        "stagnation_limit": 25,
        "init_mode": "heuristic_init",
        "top_solutions_limit": 8,
        "quality_check_enabled": True,
        "quality_check_max_items": 18,
    }

    def __init__(
        self,
        source: RuntimeEntitiesSource | Mapping[str, Any] | None = None,
        *,
        ga_runner: GaRunner | None = None,
        profile_service: AnalysisScopeProfileService | None = None,
    ):
        self.source = source
        self.ga_runner = ga_runner or self._default_ga_runner
        self.profile_service = profile_service or AnalysisScopeProfileService()

    def _default_ga_runner(self, params: dict[str, Any]) -> dict[str, Any]:
        from domain.optimization.ga import run_ga_mvp

        return run_ga_mvp(params)

    def execute(
        self,
        source: RuntimeEntitiesSource | Mapping[str, Any] | None = None,
        **options: Any,
    ) -> dict[str, Any]:
        """Compatibility alias for use-case-like calls."""
        return self.run(source=source, **options)

    def run(
        self,
        source: RuntimeEntitiesSource | Mapping[str, Any] | None = None,
        *,
        categories: Sequence[str] | None = None,
        analysis_scope: str | None = None,
        analysis_profile: AnalysisScopeProfile | None = None,
        power_lookup: Mapping[str, float] | None = None,
        target_units: float | None = None,
        max_power: float | None = None,
        criteria: Sequence[CriterionSpec | Callable[[list[RuntimeOptimizationItem]], Any]] | None = None,
        constraints: Sequence[ConstraintSpec | Callable[[list[RuntimeOptimizationItem]], Any]] | None = None,
        weights: Sequence[float] | None = None,
        pairwise_matrix: Any | None = None,
        max_budget: float | None = None,
        required_categories: Sequence[str] | None = None,
        min_selected_items: int = 1,
        max_selected_items: int | None = None,
        include_zero_quantity: bool = False,
        ga_params: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run genetic optimization over runtime entities.

        Args:
            source: Repository/service/dict with runtime ``entities``.
            categories: Entity categories to convert into candidates.
            criteria: Universal GA criteria. Defaults are generic cost/quantity/
                category-coverage criteria based on candidate properties.
            constraints: Universal GA constraints. Defaults are derived from
                ``max_budget``, ``required_categories`` and selected item count.
            weights/pairwise_matrix: Criterion weights for the GA core.
            ga_params: Additional parameters passed to ``run_ga_mvp``.
        """
        profile = analysis_profile or (
            self.profile_service.get_profile(analysis_scope) if analysis_scope is not None else None
        )
        effective_categories = tuple(
            categories
            or (profile.capital_categories if profile is not None else CAPITAL_COST_CATEGORIES)
        )
        candidates = self.build_candidates(
            source=source,
            categories=effective_categories,
            include_zero_quantity=include_zero_quantity,
        )

        criteria_specs = self._build_criteria(
            criteria,
            profile=profile,
            power_lookup=power_lookup,
        )
        dynamic_constraints = list(constraints or [])
        if profile is not None:
            dynamic_constraints.extend(
                self.profile_service.build_ga_constraints(
                    profile.scope,
                    power_lookup=power_lookup,
                    max_power=max_power,
                    target_units=target_units,
                )
            )
        constraint_specs = self._build_constraints(
            dynamic_constraints,
            max_budget=max_budget,
            required_categories=required_categories,
            min_selected_items=min_selected_items,
            max_selected_items=max_selected_items,
        )

        params = dict(self.DEFAULT_GA_PARAMS)
        params.update(dict(ga_params or {}))
        params.update(
            {
                "items": candidates,
                "criteria": criteria_specs,
                "constraints": constraint_specs,
            }
        )
        if weights is not None:
            params["weights"] = list(weights)
        elif profile is not None:
            params["weights"] = self.profile_service.default_weights(profile.scope)
        if pairwise_matrix is not None:
            params["pairwise_matrix"] = pairwise_matrix
        if profile is not None:
            params["analysis_scope"] = profile.scope
            params["analysis_profile"] = profile

        logger.info(
            "Запуск прикладного слоя GA: candidates=%s, criteria=%s, constraints=%s",
            len(candidates),
            len(criteria_specs),
            len(constraint_specs),
        )
        ga_result = self.ga_runner(params)
        return self._format_result(
            ga_result,
            candidates=candidates,
            categories=effective_categories,
            ga_params=params,
        )

    def build_candidates(
        self,
        source: RuntimeEntitiesSource | Mapping[str, Any] | None = None,
        *,
        categories: Sequence[str] | None = None,
        include_zero_quantity: bool = False,
    ) -> list[RuntimeOptimizationItem]:
        """Convert runtime rows into generic GA ``Item`` objects.

        All original row fields are preserved. The service only adds normalized
        helper properties such as ``source_category`` and ``total_cost`` so that
        default criteria/constraints can be used without requiring a fixed item
        schema.
        """
        entities = self._extract_entities(source)
        effective_categories = tuple(categories or CAPITAL_COST_CATEGORIES)
        candidates: list[RuntimeOptimizationItem] = []

        for category in effective_categories:
            for index, row in enumerate(entities.get(category, [])):
                properties = self._candidate_properties(row, category=category, index=index)
                if not include_zero_quantity and properties.get("quantity", 1.0) <= 0:
                    continue
                candidates.append(RuntimeOptimizationItem(**properties))

        return candidates

    def _extract_entities(
        self,
        source: RuntimeEntitiesSource | Mapping[str, Any] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        effective_source = source if source is not None else self.source
        if effective_source is None:
            raise ValueError("Runtime entities source is not configured")

        if isinstance(effective_source, Mapping):
            payload = effective_source.get("entities", effective_source)
        elif hasattr(effective_source, "entities"):
            payload = effective_source.entities
        else:
            raise TypeError(
                "Runtime entities source must expose an 'entities' mapping or be a mapping itself"
            )

        return {
            str(entity_name): [dict(deepcopy(row)) for row in rows]
            for entity_name, rows in dict(payload).items()
        }

    def _candidate_properties(
        self,
        row: Mapping[str, Any],
        *,
        category: str,
        index: int,
    ) -> dict[str, Any]:
        properties = normalize_runtime_row(row, category=category)
        name = str(properties.get("name") or f"{category}_{index + 1}")
        quantity = self._number(properties.get("quantity", 1.0), default=1.0)
        unit_price = self._number(
            properties.get("price", properties.get("unit_price", properties.get("cost", 0.0))),
            default=0.0,
        )
        total_cost = self._number(
            properties.get("total_cost", properties.get("capital_cost", quantity * unit_price)),
            default=quantity * unit_price,
        )

        properties.update(
            {
                "id": str(properties.get("id") or f"{category}:{index}:{name}"),
                "name": name,
                "source_category": category,
                "category": str(properties.get("category") or category),
                "source_index": index,
                "quantity": quantity,
                "unit_price": unit_price,
                "total_cost": total_cost,
                "capital_cost": total_cost,
                "selected_count": 1.0,
                "client_seats": self._number(properties.get("client_seats"), default=0.0),
            }
        )
        return properties

    def _build_criteria(
        self,
        criteria: Sequence[CriterionSpec | Callable[[list[RuntimeOptimizationItem]], Any]] | None,
        *,
        profile: AnalysisScopeProfile | None = None,
        power_lookup: Mapping[str, float] | None = None,
    ) -> list[CriterionSpec | Callable[[list[RuntimeOptimizationItem]], Any]]:
        if criteria is not None:
            return list(criteria)
        if profile is not None:
            return self.profile_service.build_ga_criteria(
                profile.scope,
                power_lookup=power_lookup,
            )

        return [
            {
                "name": "selected_quantity",
                "func": lambda subset: self._sum_property(subset, "quantity"),
                "direction": "max",
            },
            {
                "name": "category_coverage",
                "func": self._category_coverage,
                "direction": "max",
            },
            {
                "name": "capital_cost",
                "func": lambda subset: self._sum_property(subset, "total_cost"),
                "direction": "min",
            },
        ]

    def _build_constraints(
        self,
        constraints: Sequence[ConstraintSpec | Callable[[list[RuntimeOptimizationItem]], Any]] | None,
        *,
        max_budget: float | None,
        required_categories: Sequence[str] | None,
        min_selected_items: int,
        max_selected_items: int | None,
    ) -> list[ConstraintSpec | Callable[[list[RuntimeOptimizationItem]], Any]]:
        result: list[ConstraintSpec | Callable[[list[RuntimeOptimizationItem]], Any]] = list(constraints or [])

        if min_selected_items > 0:
            result.append(
                {
                    "name": "min_selected_items",
                    "func": lambda subset: len(subset),
                    "operator": ">=",
                    "bound": float(min_selected_items),
                }
            )

        if max_selected_items is not None:
            result.append(
                {
                    "name": "max_selected_items",
                    "func": lambda subset: len(subset),
                    "operator": "<=",
                    "bound": float(max_selected_items),
                }
            )

        if max_budget is not None:
            result.append(
                {
                    "name": "budget",
                    "func": lambda subset: self._sum_property(subset, "total_cost"),
                    "operator": "<=",
                    "bound": float(max_budget),
                }
            )

        for category in required_categories or ():
            category_name = str(category)
            result.append(
                {
                    "name": f"required_category_{category_name}",
                    "func": lambda subset, expected=category_name: sum(
                        1
                        for item in subset
                        if item.properties.get("source_category") == expected
                        or item.properties.get("category") == expected
                    ),
                    "operator": ">=",
                    "bound": 1.0,
                }
            )

        return result

    def _format_result(
        self,
        ga_result: Mapping[str, Any],
        *,
        candidates: Sequence[RuntimeOptimizationItem],
        categories: Sequence[str],
        ga_params: Mapping[str, Any],
    ) -> dict[str, Any]:
        selected_items = [dict(item) for item in ga_result.get("best_items", [])]
        selected_by_category = self._group_by_category(selected_items)
        totals = self._totals(selected_items)
        status = "error" if ga_result.get("error") else "ok"
        candidate_solutions = [
            self._format_candidate_solution(solution)
            for solution in ga_result.get("top_solutions", [])
        ]

        summary = {
            "status": status,
            "error": ga_result.get("error"),
            "selected_items": selected_items,
            "selected_by_category": selected_by_category,
            "totals": totals,
            "criteria": deepcopy(ga_result.get("criteria", [])),
            "constraints": deepcopy(ga_result.get("constraints", [])),
            "candidate_solutions": candidate_solutions,
            "parameters": {
                "candidate_count": len(candidates),
                "categories": list(categories),
                "ga": self._public_ga_params(ga_params),
                "analysis_profile": self._profile_metadata(ga_params),
                "criterion_names": list(ga_result.get("criterion_names", [])),
                "constraint_names": [
                    constraint.get("name")
                    for constraint in ga_result.get("constraints_metadata", [])
                ],
            },
            "ga_result": deepcopy(dict(ga_result)),
        }
        summary["export_payload"] = self._export_payload(summary)
        return summary

    def _profile_metadata(self, ga_params: Mapping[str, Any]) -> dict[str, Any] | None:
        profile = ga_params.get("analysis_profile")
        if isinstance(profile, AnalysisScopeProfile):
            return profile.to_dict()
        if isinstance(profile, Mapping):
            return deepcopy(dict(profile))
        scope = ga_params.get("analysis_scope")
        if isinstance(scope, str):
            return self.profile_service.profile_metadata(scope)
        return None

    def _public_ga_params(self, ga_params: Mapping[str, Any]) -> dict[str, Any]:
        hidden = {"items", "criteria", "constraints", "pairwise_matrix", "analysis_profile"}
        result: dict[str, Any] = {}
        for key, value in ga_params.items():
            if key in hidden or callable(value):
                continue
            if key == "weights":
                result[key] = list(value)
            elif isinstance(value, (str, int, float, bool, type(None))):
                result[key] = value
            elif isinstance(value, (list, tuple)) and all(
                isinstance(item, (str, int, float, bool, type(None))) for item in value
            ):
                result[key] = list(value)
        return result

    def _export_payload(self, summary: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "genetic_optimization": {
                "status": summary["status"],
                "error": summary["error"],
                "selected_items": deepcopy(summary["selected_items"]),
                "selected_by_category": deepcopy(summary["selected_by_category"]),
                "totals": deepcopy(summary["totals"]),
                "criteria": deepcopy(summary["criteria"]),
                "constraints": deepcopy(summary["constraints"]),
                "candidate_solutions": deepcopy(summary.get("candidate_solutions", [])),
                "quality_check": deepcopy(summary["ga_result"].get("quality_check", {})),
                "best_score": summary["ga_result"].get("best_agg"),
                "termination_reason": summary["ga_result"].get("termination_reason"),
                "generations_ran": summary["ga_result"].get("generations_ran"),
                "fitness_scale": summary["ga_result"].get("fitness_scale"),
                "analysis_profile": deepcopy(summary.get("parameters", {}).get("analysis_profile")),
            }
        }


    def _format_candidate_solution(self, solution: Mapping[str, Any]) -> dict[str, Any]:
        selected_items = [dict(item) for item in solution.get("items", [])]
        return {
            "rank": solution.get("rank"),
            "mask": tuple(solution.get("mask", ())),
            "selected_items": selected_items,
            "selected_by_category": self._group_by_category(selected_items),
            "totals": self._totals(selected_items),
            "criteria": deepcopy(solution.get("criteria", [])),
            "constraints": deepcopy(solution.get("constraints", [])),
            "score": solution.get("agg"),
            "raw_scores_by_criterion": deepcopy(solution.get("raw_scores_by_criterion", {})),
            "directed_scores_by_criterion": deepcopy(solution.get("directed_scores_by_criterion", {})),
            "normalized_scores_by_criterion": deepcopy(solution.get("normalized_scores_by_criterion", {})),
        }

    def _group_by_category(self, selected_items: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for item in selected_items:
            category = str(item.get("source_category") or item.get("category") or "unknown")
            grouped.setdefault(category, []).append(dict(item))
        return grouped

    def _totals(self, selected_items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        total_capital_cost = sum(self._number(item.get("total_cost"), default=0.0) for item in selected_items)
        total_quantity = sum(self._number(item.get("quantity"), default=0.0) for item in selected_items)
        categories = sorted(
            {
                str(item.get("source_category") or item.get("category") or "unknown")
                for item in selected_items
            }
        )
        return {
            "selected_count": len(selected_items),
            "selected_quantity": total_quantity,
            "capital_cost": total_capital_cost,
            "category_count": len(categories),
            "categories": categories,
            "client_seats": sum(
                self._number(item.get("client_seats"), default=0.0)
                for item in selected_items
            ),
        }

    def _sum_property(self, subset: Sequence[RuntimeOptimizationItem], property_name: str) -> float:
        return sum(self._number(item.properties.get(property_name), default=0.0) for item in subset)

    def _category_coverage(self, subset: Sequence[RuntimeOptimizationItem]) -> float:
        return float(
            len(
                {
                    str(item.properties.get("source_category") or item.properties.get("category"))
                    for item in subset
                    if item.properties.get("source_category") or item.properties.get("category")
                }
            )
        )

    def _number(self, value: Any, *, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)
