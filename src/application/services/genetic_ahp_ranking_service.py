"""Application service that links GA search with AHP ranking.

The service keeps both layers separated:
- the GA service searches for feasible candidate configurations;
- this service takes the best GA candidates and ranks them with AHP.

It works over the generic GA result contract and does not require the GA core to
know anything about the IT-infrastructure domain.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from application.services.genetic_optimization_service import GeneticOptimizationService
from domain.decision.ahp.math_utils import (
    ahp_priority_and_consistency,
    build_pairwise_matrix_from_scores,
)

logger = logging.getLogger(__name__)

AhpCriterionSpec = Mapping[str, Any]
SolutionEvaluator = Callable[[Mapping[str, Any]], Any]


class GeneticAhpRankingService:
    """Runs GA and then explains/ranks its candidate solutions with AHP."""

    def __init__(self, genetic_service: GeneticOptimizationService):
        self.genetic_service = genetic_service

    def execute(self, **options: Any) -> dict[str, Any]:
        return self.run(**options)

    def run(
        self,
        *,
        ahp_criteria: Sequence[AhpCriterionSpec | SolutionEvaluator] | None = None,
        ahp_weights: Sequence[float] | None = None,
        ahp_pairwise_matrix: Any | None = None,
        ahp_top_limit: int = 8,
        saaty_cap: bool = True,
        **ga_options: Any,
    ) -> dict[str, Any]:
        """Generate candidate configurations with GA and rank them with AHP.

        All ``ga_options`` are passed through to ``GeneticOptimizationService``.
        ``ahp_criteria`` is optional; by default AHP reuses the normalized GA
        criteria so that the search and explanation use the same criterion set.
        """
        ga_params = dict(ga_options.get("ga_params") or {})
        ga_params["top_solutions_limit"] = max(int(ahp_top_limit), int(ga_params.get("top_solutions_limit", 0) or 0), 1)
        ga_options["ga_params"] = ga_params

        genetic_summary = self.genetic_service.run(**ga_options)
        if genetic_summary.get("status") == "error":
            report = {
                "status": "skipped",
                "reason": genetic_summary.get("error") or "GA did not produce feasible candidates.",
                "candidate_count": 0,
            }
            return self._combined_result(genetic_summary, report)

        candidates = list(genetic_summary.get("candidate_solutions", []))
        candidates = [candidate for candidate in candidates if candidate.get("selected_items")]
        if not candidates:
            report = {
                "status": "skipped",
                "reason": "GA did not return candidate_solutions for AHP ranking.",
                "candidate_count": 0,
            }
            return self._combined_result(genetic_summary, report)

        criteria = self._criteria_from_input(ahp_criteria, genetic_summary)
        if not criteria:
            report = {
                "status": "skipped",
                "reason": "No AHP criteria are available for ranking.",
                "candidate_count": len(candidates),
            }
            return self._combined_result(genetic_summary, report)

        value_matrix = self._build_value_matrix(candidates, criteria)
        normalized_value_matrix = self._normalize_value_matrix(value_matrix, criteria)
        criteria_weights, criteria_consistency, weights_source = self._criteria_weights(
            criteria,
            genetic_summary=genetic_summary,
            ahp_weights=ahp_weights,
            ahp_pairwise_matrix=ahp_pairwise_matrix,
        )

        alt_ids = [f"GA-{index}" for index in range(1, len(candidates) + 1)]
        alt_weights_per_criterion = []
        alt_consistency_per_criterion = []
        alt_pairwise_matrices = []

        for criterion_index, criterion in enumerate(criteria):
            values = normalized_value_matrix[criterion_index, :]
            pairwise = build_pairwise_matrix_from_scores(values.tolist(), saaty_cap=saaty_cap)
            weights, lam, ci_value, cr_value = ahp_priority_and_consistency(pairwise)
            alt_pairwise_matrices.append(pairwise.tolist())
            alt_weights_per_criterion.append(weights.tolist())
            alt_consistency_per_criterion.append(
                {
                    "criterion": criterion["name"],
                    "lambda": lam,
                    "CI": ci_value,
                    "CR": cr_value,
                }
            )

        alt_w_matrix = np.array(alt_weights_per_criterion, dtype=float)
        final_scores = (criteria_weights @ alt_w_matrix).flatten()
        if float(final_scores.sum()) > 0:
            final_scores_norm = final_scores / float(final_scores.sum())
        else:
            final_scores_norm = np.ones_like(final_scores) / float(len(final_scores))

        ranked_indices = sorted(range(len(candidates)), key=lambda index: float(final_scores_norm[index]), reverse=True)
        ranking = []
        for position, index in enumerate(ranked_indices, start=1):
            candidate = candidates[index]
            ranking.append(
                {
                    "rank": position,
                    "id": alt_ids[index],
                    "name": self._candidate_name(candidate, alt_ids[index]),
                    "score": float(final_scores_norm[index]),
                    "ga_rank": candidate.get("rank"),
                    "ga_score": candidate.get("score"),
                    "totals": deepcopy(candidate.get("totals", {})),
                    "selected_items": deepcopy(candidate.get("selected_items", [])),
                }
            )

        report = {
            "status": "ok",
            "method": "GA candidates ranked by AHP",
            "candidate_count": len(candidates),
            "criteria": {
                "names": [criterion["name"] for criterion in criteria],
                "directions": [criterion["direction_label"] for criterion in criteria],
                "weights": criteria_weights.tolist(),
                "weights_source": weights_source,
                "consistency": criteria_consistency,
            },
            "alternatives": {
                "ids": alt_ids,
                "value_matrix": value_matrix.tolist(),
                "normalized_value_matrix": normalized_value_matrix.tolist(),
                "pairwise_matrices": alt_pairwise_matrices,
                "weights_per_criterion": alt_weights_per_criterion,
                "consistency_per_criterion": alt_consistency_per_criterion,
            },
            "final": {
                "raw_final_scores": final_scores.tolist(),
                "normalized_final_scores": final_scores_norm.tolist(),
                "ranking": ranking,
                "winner_id": ranking[0]["id"],
                "winner_name": ranking[0]["name"],
                "winner": ranking[0],
            },
            "explanation": (
                "ГА сформировал набор допустимых кандидатных конфигураций, "
                "после чего AHP ранжировал эти альтернативы по согласованным критериям."
            ),
        }
        return self._combined_result(genetic_summary, report)

    def _combined_result(self, genetic_summary: Mapping[str, Any], ahp_report: Mapping[str, Any]) -> dict[str, Any]:
        export_payload = deepcopy(genetic_summary.get("export_payload", {}))
        export_payload["genetic_ahp_ranking"] = deepcopy(ahp_report)
        return {
            "status": "ok" if ahp_report.get("status") == "ok" else "partial",
            "genetic_optimization": deepcopy(dict(genetic_summary)),
            "ahp_report": deepcopy(dict(ahp_report)),
            "export_payload": export_payload,
        }

    def _criteria_from_input(
        self,
        criteria_input: Sequence[AhpCriterionSpec | SolutionEvaluator] | None,
        genetic_summary: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        if criteria_input is not None:
            result = []
            for index, spec in enumerate(criteria_input):
                fallback_name = f"ahp_criterion_{index + 1}"
                if callable(spec):
                    result.append(
                        {
                            "name": getattr(spec, "__name__", fallback_name),
                            "func": spec,
                            "direction": 1,
                            "direction_label": "max",
                        }
                    )
                    continue
                func = spec.get("func", spec.get("function"))
                field = spec.get("field")
                name = str(spec.get("name") or field or getattr(func, "__name__", fallback_name))
                direction = self._direction_to_int(spec.get("direction", "max"), name)
                result.append(
                    {
                        "name": name,
                        "func": func,
                        "field": field,
                        "direction": direction,
                        "direction_label": "max" if direction == 1 else "min",
                    }
                )
            return result

        ga_result = genetic_summary.get("ga_result", {}) if isinstance(genetic_summary.get("ga_result"), Mapping) else {}
        criteria_meta = list(ga_result.get("criteria_metadata", []))
        result = []
        for index, meta in enumerate(criteria_meta):
            name = str(meta.get("name") or f"criterion_{index + 1}")
            # Normalized GA scores are already transformed to "larger is better".
            result.append(
                {
                    "name": name,
                    "field": ("normalized_scores_by_criterion", name),
                    "direction": 1,
                    "direction_label": str(meta.get("direction") or "max"),
                    "uses_ga_normalized_score": True,
                }
            )
        return result

    def _build_value_matrix(
        self,
        candidates: Sequence[Mapping[str, Any]],
        criteria: Sequence[Mapping[str, Any]],
    ) -> np.ndarray:
        values = np.zeros((len(criteria), len(candidates)), dtype=float)
        for criterion_index, criterion in enumerate(criteria):
            for candidate_index, candidate in enumerate(candidates):
                values[criterion_index, candidate_index] = self._value_for_candidate(candidate, criterion)
        return values

    def _value_for_candidate(self, candidate: Mapping[str, Any], criterion: Mapping[str, Any]) -> float:
        func = criterion.get("func")
        if callable(func):
            value = func(candidate)
        else:
            field = criterion.get("field")
            if isinstance(field, (list, tuple)) and len(field) == 2:
                container = candidate.get(field[0], {})
                value = container.get(field[1], 0.0) if isinstance(container, Mapping) else 0.0
            elif isinstance(field, str):
                value = self._nested_get(candidate, field, default=0.0)
            else:
                value = candidate.get("normalized_scores_by_criterion", {}).get(criterion.get("name"), 0.0)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = 0.0
        if not np.isfinite(numeric):
            return 0.0
        return numeric

    def _normalize_value_matrix(
        self,
        values: np.ndarray,
        criteria: Sequence[Mapping[str, Any]],
    ) -> np.ndarray:
        normalized = np.ones_like(values, dtype=float)
        for index, criterion in enumerate(criteria):
            row = values[index, :].astype(float)
            if int(criterion.get("direction", 1)) == -1:
                row = -row
            min_value = float(np.min(row))
            max_value = float(np.max(row))
            if abs(max_value - min_value) <= 1e-12:
                normalized[index, :] = np.ones(row.shape, dtype=float)
            else:
                normalized[index, :] = (row - min_value) / (max_value - min_value)
        return np.clip(normalized, 0.0, 1.0)

    def _criteria_weights(
        self,
        criteria: Sequence[Mapping[str, Any]],
        *,
        genetic_summary: Mapping[str, Any],
        ahp_weights: Sequence[float] | None,
        ahp_pairwise_matrix: Any | None,
    ) -> tuple[np.ndarray, dict[str, Any], str]:
        criteria_count = len(criteria)
        if ahp_pairwise_matrix is not None:
            matrix = np.array(ahp_pairwise_matrix, dtype=float)
            if matrix.shape != (criteria_count, criteria_count):
                raise ValueError("ahp_pairwise_matrix must be square and match AHP criteria count")
            weights, lam, ci_value, cr_value = ahp_priority_and_consistency(matrix)
            return weights, {"lambda": lam, "CI": ci_value, "CR": cr_value}, "ahp_pairwise_matrix"

        if ahp_weights is not None:
            return self._normalize_weights(ahp_weights, criteria_count), {}, "direct_ahp_weights"

        ga_weights = self._weights_from_ga_summary(genetic_summary, criteria)
        if ga_weights is not None:
            return ga_weights, {}, "ga_weights"

        return np.ones(criteria_count, dtype=float) / float(criteria_count), {}, "equal_default"

    def _weights_from_ga_summary(
        self,
        genetic_summary: Mapping[str, Any],
        criteria: Sequence[Mapping[str, Any]],
    ) -> np.ndarray | None:
        ga_result = genetic_summary.get("ga_result", {}) if isinstance(genetic_summary.get("ga_result"), Mapping) else {}
        ga_names = list(ga_result.get("criterion_names", []))
        ga_weights = list(ga_result.get("weights", []))
        if not ga_names or len(ga_names) != len(ga_weights):
            return None
        by_name = {str(name): float(weight) for name, weight in zip(ga_names, ga_weights)}
        requested = []
        for criterion in criteria:
            name = str(criterion.get("name"))
            if name not in by_name:
                return None
            requested.append(by_name[name])
        return self._normalize_weights(requested, len(criteria))

    def _normalize_weights(self, weights: Sequence[Any], count: int) -> np.ndarray:
        vector = np.array(weights, dtype=float)
        if vector.shape != (count,):
            raise ValueError("AHP weights must match AHP criteria count")
        if not np.all(np.isfinite(vector)) or np.any(vector < 0):
            raise ValueError("AHP weights must be finite non-negative numbers")
        total = float(vector.sum())
        if total <= 1e-12:
            raise ValueError("AHP weights sum must be greater than zero")
        return vector / total

    def _direction_to_int(self, direction: Any, name: str) -> int:
        if isinstance(direction, str):
            value = direction.strip().lower()
            if value in {"max", "maximize", "maximise", "+", "1"}:
                return 1
            if value in {"min", "minimize", "minimise", "-", "-1"}:
                return -1
        else:
            try:
                numeric = int(direction)
            except (TypeError, ValueError):
                numeric = 0
            if numeric in (-1, 1):
                return numeric
        raise ValueError(f"AHP criterion '{name}' direction must be 'max' or 'min'")

    def _nested_get(self, mapping: Mapping[str, Any], path: str, *, default: Any) -> Any:
        current: Any = mapping
        for part in path.split("."):
            if not isinstance(current, Mapping) or part not in current:
                return default
            current = current[part]
        return current

    def _candidate_name(self, candidate: Mapping[str, Any], fallback: str) -> str:
        items = list(candidate.get("selected_items", []))
        if not items:
            return fallback
        names = [str(item.get("name")) for item in items[:3] if item.get("name")]
        suffix = "" if len(items) <= 3 else f" +{len(items) - 3}"
        return ", ".join(names) + suffix if names else fallback


__all__ = ["GeneticAhpRankingService"]
