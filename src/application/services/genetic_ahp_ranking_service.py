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
        ``ahp_criteria`` is optional; by default AHP independently evaluates
        the GA candidate pool by raw candidate criterion values and anchored
        GA normalization bounds instead of reusing ready normalized GA scores.
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
        candidate_configurations = list(genetic_summary.get("candidate_configurations", []))
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
        utility_matrix = self._build_utility_matrix(value_matrix, candidates, criteria)
        assessment_mode = self._assessment_mode(criteria)
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
            utilities = utility_matrix[criterion_index, :]
            pairwise = build_pairwise_matrix_from_scores(utilities.tolist(), saaty_cap=saaty_cap)
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
                    "ga_rank": self._candidate_ga_rank(candidate, index),
                    "ga_score": candidate.get("score"),
                    "totals": deepcopy(candidate.get("totals", {})),
                    "selected_items": deepcopy(candidate.get("selected_items", [])),
                }
            )

        criterion_diagnostics = self._criterion_diagnostics(
            criteria=criteria,
            value_matrix=value_matrix,
            utility_matrix=utility_matrix,
        )
        agreement = self._build_agreement_report(
            candidates=candidates,
            alt_ids=alt_ids,
            ranked_indices=ranked_indices,
            ahp_scores=final_scores_norm,
            criteria=criteria,
            value_matrix=value_matrix,
            utility_matrix=utility_matrix,
            criterion_diagnostics=criterion_diagnostics,
        )
        hybrid_assessment = self._build_hybrid_assessment(
            candidates=candidates,
            alt_ids=alt_ids,
            ranked_indices=ranked_indices,
            ahp_scores=final_scores_norm,
            agreement=agreement,
        )

        report = {
            "status": "ok",
            "method": "GA candidates ranked by AHP",
            "assessment_mode": assessment_mode,
            "candidate_count": len(candidates),
            "criteria": {
                "names": [criterion["name"] for criterion in criteria],
                "directions": [criterion["direction_label"] for criterion in criteria],
                "weights": criteria_weights.tolist(),
                "weights_source": weights_source,
                "consistency": criteria_consistency,
                "normalization_scope": [
                    criterion.get("normalization_scope", "candidate_pool") for criterion in criteria
                ],
                "normalization_mins": [criterion.get("normalization_min") for criterion in criteria],
                "normalization_maxs": [criterion.get("normalization_max") for criterion in criteria],
            },
            "alternatives": {
                "ids": alt_ids,
                "value_matrix": value_matrix.tolist(),
                "utility_matrix": utility_matrix.tolist(),
                "normalized_value_matrix": utility_matrix.tolist(),
                "pairwise_matrices": alt_pairwise_matrices,
                "weights_per_criterion": alt_weights_per_criterion,
                "consistency_per_criterion": alt_consistency_per_criterion,
            },
            "criterion_diagnostics": criterion_diagnostics,
            "final": {
                "raw_final_scores": final_scores.tolist(),
                "normalized_final_scores": final_scores_norm.tolist(),
                "ranking": ranking,
                "winner_id": ranking[0]["id"],
                "winner_name": ranking[0]["name"],
                "winner": ranking[0],
            },
            "candidate_configurations": deepcopy(candidate_configurations),
            "independent_assessment": {
                "candidate_pool": self._candidate_pool_snapshot(candidates, alt_ids),
                "candidate_configurations": deepcopy(candidate_configurations),
                "ga_scores": [candidate.get("score") for candidate in candidates],
                "ahp_scores": final_scores_norm.tolist(),
                "criterion_values": self._matrix_by_candidate(alt_ids, criteria, value_matrix),
                "criterion_utilities": self._matrix_by_candidate(alt_ids, criteria, utility_matrix),
            },
            "agreement": agreement,
            "hybrid_assessment": hybrid_assessment,
            "explanation": (
                "ГА сформировал набор допустимых кандидатных конфигураций. "
                "AHP затем независимо оценил этот же пул по значениям критериев кандидатов "
                "и якорной нормализации из GA-результата, а не по готовому GA-score."
            ),
        }
        return self._combined_result(genetic_summary, report)

    def _combined_result(self, genetic_summary: Mapping[str, Any], ahp_report: Mapping[str, Any]) -> dict[str, Any]:
        export_payload = deepcopy(genetic_summary.get("export_payload", {}))
        export_payload["genetic_ahp_ranking"] = deepcopy(ahp_report)
        if ahp_report.get("candidate_configurations"):
            export_payload["candidate_configurations"] = deepcopy(
                ahp_report.get("candidate_configurations", [])
            )
        export_payload["ga_ahp_agreement"] = deepcopy(ahp_report.get("agreement", {}))
        export_payload["independent_assessment"] = deepcopy(
            ahp_report.get("independent_assessment", {})
        )
        if "hybrid_assessment" in ahp_report:
            export_payload["hybrid_assessment"] = deepcopy(ahp_report.get("hybrid_assessment", {}))
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

        ga_result = (
            genetic_summary.get("ga_result", {})
            if isinstance(genetic_summary.get("ga_result"), Mapping)
            else {}
        )
        criteria_meta = list(ga_result.get("criteria_metadata", []))
        criterion_names = [str(name) for name in ga_result.get("criterion_names", [])]
        criterion_directions = list(ga_result.get("criterion_directions", []))
        normalization_mins = list(ga_result.get("normalization_mins") or [])
        normalization_maxs = list(ga_result.get("normalization_maxs") or [])

        result = []
        for index, meta in enumerate(criteria_meta):
            name = str(meta.get("name") or f"criterion_{index + 1}")
            direction_label = str(
                meta.get("direction")
                or (criterion_directions[index] if index < len(criterion_directions) else "max")
            )
            direction = self._direction_to_int(direction_label, name)
            try:
                bounds_index = criterion_names.index(name)
            except ValueError:
                bounds_index = index
            normalization_min = (
                normalization_mins[bounds_index] if bounds_index < len(normalization_mins) else None
            )
            normalization_max = (
                normalization_maxs[bounds_index] if bounds_index < len(normalization_maxs) else None
            )
            result.append(
                {
                    "name": name,
                    "field": ("raw_scores_by_criterion", name),
                    "directed_field": ("directed_scores_by_criterion", name),
                    "direction": direction,
                    "direction_label": direction_label,
                    "normalization_min": normalization_min,
                    "normalization_max": normalization_max,
                    "normalization_scope": "ga_search_space",
                    "assessment_source": "independent_candidate_metrics",
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

    def _build_utility_matrix(
        self,
        value_matrix: np.ndarray,
        candidates: Sequence[Mapping[str, Any]],
        criteria: Sequence[Mapping[str, Any]],
    ) -> np.ndarray:
        utilities = np.ones_like(value_matrix, dtype=float)
        local_normalized = self._normalize_value_matrix(value_matrix, criteria)
        for criterion_index, criterion in enumerate(criteria):
            if criterion.get("assessment_source") != "independent_candidate_metrics":
                utilities[criterion_index, :] = local_normalized[criterion_index, :]
                continue

            min_value = self._finite_float_or_none(criterion.get("normalization_min"))
            max_value = self._finite_float_or_none(criterion.get("normalization_max"))
            if min_value is None or max_value is None or abs(max_value - min_value) <= 1e-12:
                utilities[criterion_index, :] = local_normalized[criterion_index, :]
                continue

            directed_values = np.array(
                [self._directed_value_for_candidate(candidate, criterion) for candidate in candidates],
                dtype=float,
            )
            utilities[criterion_index, :] = (directed_values - min_value) / (max_value - min_value)
        return np.clip(utilities, 0.0, 1.0)

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

    def _directed_value_for_candidate(
        self, candidate: Mapping[str, Any], criterion: Mapping[str, Any]
    ) -> float:
        directed_field = criterion.get("directed_field")
        if directed_field is not None:
            directed_criterion = dict(criterion)
            directed_criterion["field"] = directed_field
            return self._value_for_candidate(candidate, directed_criterion)

        value = self._value_for_candidate(candidate, criterion)
        return value if int(criterion.get("direction", 1)) == 1 else -value

    def _finite_float_or_none(self, value: Any) -> float | None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return numeric if np.isfinite(numeric) else None

    def _assessment_mode(self, criteria: Sequence[Mapping[str, Any]]) -> str:
        if criteria and all(
            criterion.get("assessment_source") == "independent_candidate_metrics"
            for criterion in criteria
        ):
            return "independent_candidate_metrics"
        return "custom_ahp_criteria"

    def _build_hybrid_assessment(
        self,
        *,
        candidates: Sequence[Mapping[str, Any]],
        alt_ids: Sequence[str],
        ranked_indices: Sequence[int],
        ahp_scores: np.ndarray,
        agreement: Mapping[str, Any] | None,
        lambda_value: float = 0.5,
    ) -> dict[str, Any]:
        """Build a neutral hybrid ranking over the same GA/AHP candidate pool.

        Raw GA and AHP scores use different scales, so both score series are
        first projected to 0..1 over the candidate pool. If a score series is
        constant, its normalized contribution is set to 0.5 for every candidate:
        the method is present in the report, but it does not distort the final
        ranking when it cannot distinguish alternatives.
        """
        candidate_count = len(candidates)
        if candidate_count == 0:
            return {
                "status": "skipped",
                "reason": "No candidates are available for hybrid assessment.",
                "candidate_count": 0,
            }

        lambda_value = self._bounded_lambda(lambda_value)
        ga_scores = [self._finite_score_or_none(candidate.get("score")) for candidate in candidates]
        ahp_score_values = [self._finite_score_or_none(value) for value in ahp_scores.tolist()]
        ga_normalized, ga_norm_info = self._normalize_score_series(ga_scores)
        ahp_normalized, ahp_norm_info = self._normalize_score_series(ahp_score_values)

        ahp_rank_by_index = {candidate_index: rank for rank, candidate_index in enumerate(ranked_indices, start=1)}
        ranking = self._hybrid_ranking_for_lambda(
            candidates=candidates,
            alt_ids=alt_ids,
            ga_scores=ga_scores,
            ahp_scores=ahp_score_values,
            ga_normalized=ga_normalized,
            ahp_normalized=ahp_normalized,
            ahp_rank_by_index=ahp_rank_by_index,
            lambda_value=lambda_value,
        )
        warnings = self._hybrid_warnings(agreement=agreement, ga_norm_info=ga_norm_info, ahp_norm_info=ahp_norm_info)
        winner = ranking[0] if ranking else None
        sensitivity = self._hybrid_lambda_sensitivity(
            candidates=candidates,
            alt_ids=alt_ids,
            ga_scores=ga_scores,
            ahp_scores=ahp_score_values,
            ga_normalized=ga_normalized,
            ahp_normalized=ahp_normalized,
            ahp_rank_by_index=ahp_rank_by_index,
            lambda_values=(0.4, 0.5, 0.6),
        )
        winner_explanation = self._hybrid_winner_explanation(
            winner=winner,
            ranking=ranking,
            lambda_value=lambda_value,
            sensitivity=sensitivity,
        )
        return {
            "status": "ok",
            "method": "weighted_normalized_ga_ahp_score",
            "lambda": float(lambda_value),
            "lambda_label": self._lambda_label(lambda_value),
            "candidate_count": candidate_count,
            "normalization": {
                "ga": "minmax_over_candidate_pool",
                "ahp": "minmax_over_candidate_pool",
                "ga_details": ga_norm_info,
                "ahp_details": ahp_norm_info,
            },
            "ranking": ranking,
            "winner_id": winner.get("id") if winner else None,
            "winner_name": winner.get("name") if winner else None,
            "winner": winner,
            "winner_explanation": winner_explanation,
            "sensitivity": sensitivity,
            "warnings": warnings,
            "explanation": (
                "Hybrid score combines normalized GA and AHP scores: "
                "H_i = λ × Ĝ_i + (1 - λ) × Â_i. "
                "Agreement diagnostics and λ-sensitivity are reported separately and do not change the formula."
            ),
        }

    def _hybrid_ranking_for_lambda(
        self,
        *,
        candidates: Sequence[Mapping[str, Any]],
        alt_ids: Sequence[str],
        ga_scores: Sequence[float | None],
        ahp_scores: Sequence[float | None],
        ga_normalized: Sequence[float],
        ahp_normalized: Sequence[float],
        ahp_rank_by_index: Mapping[int, int],
        lambda_value: float,
    ) -> list[dict[str, Any]]:
        lambda_value = self._bounded_lambda(lambda_value)
        hybrid_rows = []
        for index, candidate in enumerate(candidates):
            ga_part = lambda_value * float(ga_normalized[index])
            ahp_part = (1.0 - lambda_value) * float(ahp_normalized[index])
            hybrid_score = ga_part + ahp_part
            hybrid_rows.append(
                {
                    "source_index": index,
                    "id": alt_ids[index],
                    "name": self._candidate_name(candidate, alt_ids[index]),
                    "ga_rank": self._candidate_ga_rank(candidate, index),
                    "ahp_rank": int(ahp_rank_by_index.get(index, index + 1)),
                    "hybrid_score": float(hybrid_score),
                    "ga_score": ga_scores[index],
                    "ahp_score": ahp_scores[index],
                    "ga_score_normalized": float(ga_normalized[index]),
                    "ahp_score_normalized": float(ahp_normalized[index]),
                    "ga_contribution": float(ga_part),
                    "ahp_contribution": float(ahp_part),
                    "score_disagreement": float(abs(float(ga_normalized[index]) - float(ahp_normalized[index]))),
                    "totals": deepcopy(candidate.get("totals", {})),
                }
            )

        sorted_rows = sorted(
            hybrid_rows,
            key=lambda row: (
                float(row["hybrid_score"]),
                -int(row["ga_rank"]),
                -int(row["ahp_rank"]),
            ),
            reverse=True,
        )
        ranking = []
        for rank, row in enumerate(sorted_rows, start=1):
            clean_row = dict(row)
            clean_row.pop("source_index", None)
            clean_row["rank"] = rank
            ranking.append(clean_row)
        return ranking

    def _hybrid_lambda_sensitivity(
        self,
        *,
        candidates: Sequence[Mapping[str, Any]],
        alt_ids: Sequence[str],
        ga_scores: Sequence[float | None],
        ahp_scores: Sequence[float | None],
        ga_normalized: Sequence[float],
        ahp_normalized: Sequence[float],
        ahp_rank_by_index: Mapping[int, int],
        lambda_values: Sequence[float],
    ) -> dict[str, Any]:
        winners = []
        for raw_lambda in lambda_values:
            lambda_value = self._bounded_lambda(raw_lambda)
            ranking = self._hybrid_ranking_for_lambda(
                candidates=candidates,
                alt_ids=alt_ids,
                ga_scores=ga_scores,
                ahp_scores=ahp_scores,
                ga_normalized=ga_normalized,
                ahp_normalized=ahp_normalized,
                ahp_rank_by_index=ahp_rank_by_index,
                lambda_value=lambda_value,
            )
            winner = ranking[0] if ranking else {}
            winners.append(
                {
                    "lambda": float(lambda_value),
                    "lambda_label": self._lambda_label(lambda_value),
                    "winner_id": winner.get("id"),
                    "winner_name": winner.get("name"),
                    "winner_ga_rank": winner.get("ga_rank"),
                    "winner_ahp_rank": winner.get("ahp_rank"),
                    "winner_hybrid_score": winner.get("hybrid_score"),
                }
            )

        winner_ids = {str(item.get("winner_id")) for item in winners if item.get("winner_id")}
        stable_winner = len(winner_ids) <= 1
        if stable_winner:
            summary = (
                "Гибридный лидер устойчив при λ=0.4, 0.5 и 0.6; "
                "небольшое изменение доверия к GA/AHP не меняет итоговый выбор."
            )
        else:
            summary = (
                "Гибридный лидер зависит от λ; результат следует трактовать осторожно "
                "и показывать вместе с независимыми рейтингами GA и AHP."
            )
        return {
            "lambda_values": [float(self._bounded_lambda(value)) for value in lambda_values],
            "winners": winners,
            "stable_winner": stable_winner,
            "summary": summary,
        }

    def _hybrid_winner_explanation(
        self,
        *,
        winner: Mapping[str, Any] | None,
        ranking: Sequence[Mapping[str, Any]],
        lambda_value: float,
        sensitivity: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not winner or not ranking:
            return {}

        ga_winner = min(ranking, key=lambda row: int(row.get("ga_rank") or 10**9))
        ahp_winner = min(ranking, key=lambda row: int(row.get("ahp_rank") or 10**9))
        winner_id = str(winner.get("id") or "")
        ga_winner_id = str(ga_winner.get("id") or "")
        ahp_winner_id = str(ahp_winner.get("id") or "")
        lambda_label = self._lambda_label(lambda_value)
        score = winner.get("hybrid_score")
        score_text = f"{float(score):.4f}" if isinstance(score, (int, float)) else "—"

        if winner_id == ga_winner_id == ahp_winner_id:
            reason = "он одновременно является лидером GA и AHP"
        elif winner_id == ga_winner_id:
            reason = (
                f"он является лидером GA и остаётся на AHP-ранге {winner.get('ahp_rank')}; "
                "AHP-преимущество другого варианта не перекрывает его GA-вклад"
            )
        elif winner_id == ahp_winner_id:
            reason = (
                f"он является лидером AHP и сохраняет GA-ранг {winner.get('ga_rank')}; "
                "преимущество по AHP достаточно сильно влияет на компромиссную оценку"
            )
        else:
            reason = (
                "он не обязательно лидер отдельного метода, но получил лучший баланс "
                "нормализованных вкладов GA и AHP"
            )

        stability = "устойчив" if sensitivity.get("stable_winner") else "чувствителен к λ"
        summary = (
            f"Гибридная оценка выбрала {winner_id} ({winner.get('name', '—')}) со score={score_text}, "
            f"потому что {reason}. Режим: {lambda_label}; результат {stability}."
        )
        return {
            "summary": summary,
            "winner_id": winner_id,
            "winner_name": winner.get("name"),
            "hybrid_score": winner.get("hybrid_score"),
            "ga_rank": winner.get("ga_rank"),
            "ahp_rank": winner.get("ahp_rank"),
            "ga_winner_id": ga_winner_id,
            "ahp_winner_id": ahp_winner_id,
            "lambda": float(self._bounded_lambda(lambda_value)),
            "lambda_label": lambda_label,
            "stable_across_sensitivity": bool(sensitivity.get("stable_winner")),
        }

    def _bounded_lambda(self, value: Any) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.5
        if not np.isfinite(numeric):
            return 0.5
        return min(1.0, max(0.0, numeric))

    def _lambda_label(self, value: float) -> str:
        if abs(value - 0.5) <= 1e-12:
            return "нейтральный компромисс"
        if value > 0.5:
            return "с приоритетом GA"
        return "с приоритетом AHP"

    def _normalize_score_series(self, scores: Sequence[float | None]) -> tuple[list[float], dict[str, Any]]:
        finite_values = [float(value) for value in scores if value is not None and np.isfinite(float(value))]
        if not finite_values:
            return [0.5 for _ in scores], {
                "status": "no_finite_scores",
                "min": None,
                "max": None,
                "constant_value": None,
                "missing_count": len(scores),
            }

        min_value = min(finite_values)
        max_value = max(finite_values)
        missing_count = len(scores) - len(finite_values)
        if abs(max_value - min_value) <= 1e-12:
            return [0.5 for _ in scores], {
                "status": "constant_score",
                "min": min_value,
                "max": max_value,
                "constant_value": min_value,
                "missing_count": missing_count,
            }

        normalized = []
        for value in scores:
            numeric = self._finite_score_or_none(value)
            if numeric is None:
                numeric = min_value
            normalized.append(float((numeric - min_value) / (max_value - min_value)))
        return normalized, {
            "status": "ok",
            "min": min_value,
            "max": max_value,
            "constant_value": None,
            "missing_count": missing_count,
        }

    def _hybrid_warnings(
        self,
        *,
        agreement: Mapping[str, Any] | None,
        ga_norm_info: Mapping[str, Any],
        ahp_norm_info: Mapping[str, Any],
    ) -> list[str]:
        warnings = []
        agreement_status = agreement.get("status") if isinstance(agreement, Mapping) else None
        if agreement_status == "conflict":
            warnings.append(
                "Гибридная оценка рассчитана как компромиссный ориентир, но GA и AHP находятся в конфликте."
            )
        if ga_norm_info.get("status") in {"constant_score", "no_finite_scores"}:
            warnings.append(
                "GA-score не различает кандидатов в гибридной нормализации; вклад GA задан одинаковым для всех вариантов."
            )
        if ahp_norm_info.get("status") in {"constant_score", "no_finite_scores"}:
            warnings.append(
                "AHP-score не различает кандидатов в гибридной нормализации; вклад AHP задан одинаковым для всех вариантов."
            )
        return warnings

    def _build_agreement_report(
        self,
        *,
        candidates: Sequence[Mapping[str, Any]],
        alt_ids: Sequence[str],
        ranked_indices: Sequence[int],
        ahp_scores: np.ndarray,
        criteria: Sequence[Mapping[str, Any]],
        value_matrix: np.ndarray,
        utility_matrix: np.ndarray,
        criterion_diagnostics: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        candidate_count = len(candidates)
        ahp_rank_by_index = {candidate_index: rank for rank, candidate_index in enumerate(ranked_indices, start=1)}
        ga_rank_by_index = {
            index: self._candidate_ga_rank(candidate, index)
            for index, candidate in enumerate(candidates)
        }

        rank_deltas = []
        absolute_deltas = []
        for index, candidate in enumerate(candidates):
            ga_rank = ga_rank_by_index[index]
            ahp_rank = ahp_rank_by_index[index]
            delta = ahp_rank - ga_rank
            absolute_deltas.append(abs(delta))
            rank_deltas.append(
                {
                    "id": alt_ids[index],
                    "name": self._candidate_name(candidate, alt_ids[index]),
                    "ga_rank": ga_rank,
                    "ahp_rank": ahp_rank,
                    "rank_delta": delta,
                    "ga_score": candidate.get("score"),
                    "ahp_score": float(ahp_scores[index]),
                }
            )

        top_3_overlap = self._top_overlap(ga_rank_by_index, ahp_rank_by_index, limit=3)
        top_5_overlap = self._top_overlap(ga_rank_by_index, ahp_rank_by_index, limit=5)
        max_rank_delta = max(absolute_deltas, default=0)
        mean_abs_rank_delta = float(np.mean(absolute_deltas)) if absolute_deltas else 0.0
        winner_index = int(ranked_indices[0]) if ranked_indices else 0
        winner_ga_rank = ga_rank_by_index.get(winner_index, winner_index + 1)
        winner_rank_delta = ahp_rank_by_index.get(winner_index, 1) - winner_ga_rank
        status, status_label = self._agreement_status(
            candidate_count=candidate_count,
            top_3_overlap=top_3_overlap,
            top_5_overlap=top_5_overlap,
            max_rank_delta=max_rank_delta,
            winner_ga_rank=winner_ga_rank,
        )
        winner_comparison = self._winner_comparison(
            candidates=candidates,
            alt_ids=alt_ids,
            ranked_indices=ranked_indices,
            ahp_scores=ahp_scores,
            ga_rank_by_index=ga_rank_by_index,
            criteria=criteria,
            value_matrix=value_matrix,
            utility_matrix=utility_matrix,
        )
        winner_interpretation = dict(winner_comparison.get("interpretation", {}))
        diagnostics = dict(criterion_diagnostics or {})
        warnings = self._agreement_warnings(
            status=status,
            candidate_count=candidate_count,
            winner_ga_rank=winner_ga_rank,
            max_rank_delta=max_rank_delta,
            top_3_overlap=top_3_overlap,
            criterion_diagnostics=diagnostics,
        )

        return {
            "status": status,
            "status_label": status_label,
            "summary": self._agreement_summary(
                status_label=status_label,
                top_3_overlap=top_3_overlap,
                top_5_overlap=top_5_overlap,
                max_rank_delta=max_rank_delta,
                winner_ga_rank=winner_ga_rank,
            ),
            "candidate_count": candidate_count,
            "top_3_overlap": top_3_overlap,
            "top_5_overlap": top_5_overlap,
            "max_rank_delta": max_rank_delta,
            "mean_abs_rank_delta": mean_abs_rank_delta,
            "winner_ga_rank": winner_ga_rank,
            "winner_rank_delta": winner_rank_delta,
            "rank_deltas": rank_deltas,
            "winner_comparison": winner_comparison,
            "winner_interpretation": winner_interpretation,
            "criterion_diagnostics": diagnostics,
            "warnings": warnings,
            "thresholds": {
                "high": "top-3 совпадает, максимальный сдвиг ранга не больше 1",
                "moderate": "top-3 почти совпадает, максимальный сдвиг ранга не больше 3",
                "conflict": "AHP-лидер вне GA top-5 или top-3 не пересекается",
            },
        }

    def _criterion_diagnostics(
        self,
        *,
        criteria: Sequence[Mapping[str, Any]],
        value_matrix: np.ndarray,
        utility_matrix: np.ndarray,
    ) -> dict[str, Any]:
        inactive = []
        active = []
        for criterion_index, criterion in enumerate(criteria):
            criterion_name = str(criterion.get("name") or f"criterion_{criterion_index + 1}")
            values = value_matrix[criterion_index, :].astype(float) if value_matrix.size else np.array([], dtype=float)
            utilities = utility_matrix[criterion_index, :].astype(float) if utility_matrix.size else np.array([], dtype=float)
            finite_values = values[np.isfinite(values)]
            finite_utilities = utilities[np.isfinite(utilities)]
            is_constant_value = bool(
                len(finite_values) > 0
                and len(finite_values) == len(values)
                and float(np.max(finite_values) - np.min(finite_values)) <= 1e-12
            )
            is_constant_utility = bool(
                len(finite_utilities) > 0
                and len(finite_utilities) == len(utilities)
                and float(np.max(finite_utilities) - np.min(finite_utilities)) <= 1e-12
            )
            row = {
                "criterion": criterion_name,
                "direction": str(criterion.get("direction_label", "max")),
                "candidate_count": int(len(values)),
                "min_value": float(np.min(finite_values)) if len(finite_values) else None,
                "max_value": float(np.max(finite_values)) if len(finite_values) else None,
                "min_utility": float(np.min(finite_utilities)) if len(finite_utilities) else None,
                "max_utility": float(np.max(finite_utilities)) if len(finite_utilities) else None,
                "affects_ranking": not is_constant_utility,
            }
            if is_constant_value:
                constant_value = float(finite_values[0])
                row.update(
                    {
                        "reason": "constant_value",
                        "constant_value": constant_value,
                        "constant_utility": float(finite_utilities[0]) if len(finite_utilities) else None,
                        "message": (
                            f"Критерий {criterion_name} не влияет на ранжирование: "
                            f"у всех кандидатов значение {self._format_plain_number(constant_value)}."
                        ),
                    }
                )
                inactive.append(row)
            elif is_constant_utility:
                row.update(
                    {
                        "reason": "constant_utility",
                        "message": (
                            f"Критерий {criterion_name} не влияет на ранжирование после нормализации: "
                            "у всех кандидатов одинаковая полезность."
                        ),
                    }
                )
                inactive.append(row)
            else:
                active.append(row)

        inactive_messages = [str(item["message"]) for item in inactive if item.get("message")]
        return {
            "status": "has_inactive_criteria" if inactive else "ok",
            "inactive_criteria_count": len(inactive),
            "active_criteria_count": len(active),
            "inactive_criteria": inactive,
            "active_criteria": active,
            "warnings": inactive_messages,
            "summary": self._criterion_diagnostics_summary(inactive),
        }

    def _criterion_diagnostics_summary(self, inactive_criteria: Sequence[Mapping[str, Any]]) -> str:
        if not inactive_criteria:
            return "Все критерии различают хотя бы часть кандидатных конфигураций."
        names = ", ".join(str(item.get("criterion")) for item in inactive_criteria[:3])
        suffix = "" if len(inactive_criteria) <= 3 else f" и ещё {len(inactive_criteria) - 3}"
        return (
            "Обнаружены критерии, которые не влияют на ранжирование кандидатов: "
            f"{names}{suffix}."
        )

    def _format_plain_number(self, value: float) -> str:
        if abs(value - round(value)) <= 1e-9:
            return f"{int(round(value)):,}".replace(",", " ")
        return f"{value:.4f}".rstrip("0").rstrip(".")

    def _candidate_ga_rank(self, candidate: Mapping[str, Any], fallback_index: int) -> int:
        try:
            rank = int(candidate.get("rank"))
        except (TypeError, ValueError):
            rank = fallback_index + 1
        return rank if rank > 0 else fallback_index + 1

    def _top_overlap(
        self,
        ga_rank_by_index: Mapping[int, int],
        ahp_rank_by_index: Mapping[int, int],
        *,
        limit: int,
    ) -> int:
        if not ga_rank_by_index or not ahp_rank_by_index:
            return 0
        effective_limit = min(limit, len(ga_rank_by_index), len(ahp_rank_by_index))
        ga_top = {index for index, rank in ga_rank_by_index.items() if rank <= effective_limit}
        ahp_top = {index for index, rank in ahp_rank_by_index.items() if rank <= effective_limit}
        return len(ga_top & ahp_top)

    def _agreement_status(
        self,
        *,
        candidate_count: int,
        top_3_overlap: int,
        top_5_overlap: int,
        max_rank_delta: int,
        winner_ga_rank: int,
    ) -> tuple[str, str]:
        if candidate_count <= 1:
            return "high", "высокая согласованность"

        top_3_limit = min(3, candidate_count)
        top_5_limit = min(5, candidate_count)
        if winner_ga_rank > top_5_limit or top_3_overlap == 0:
            return "conflict", "конфликт оценок"
        if top_3_overlap == top_3_limit and max_rank_delta <= 1:
            return "high", "высокая согласованность"
        if top_3_overlap >= max(1, top_3_limit - 1) and max_rank_delta <= 3:
            return "moderate", "умеренная согласованность"
        if top_5_overlap >= max(1, top_5_limit - 2):
            return "low", "слабая согласованность"
        return "conflict", "конфликт оценок"

    def _winner_comparison(
        self,
        *,
        candidates: Sequence[Mapping[str, Any]],
        alt_ids: Sequence[str],
        ranked_indices: Sequence[int],
        ahp_scores: np.ndarray,
        ga_rank_by_index: Mapping[int, int],
        criteria: Sequence[Mapping[str, Any]],
        value_matrix: np.ndarray,
        utility_matrix: np.ndarray,
    ) -> dict[str, Any]:
        if not candidates or not ranked_indices:
            return {}

        ahp_winner_index = int(ranked_indices[0])
        ga_winner_index = min(ga_rank_by_index, key=lambda index: ga_rank_by_index[index])
        criterion_deltas = []
        for criterion_index, criterion in enumerate(criteria):
            criterion_deltas.append(
                {
                    "criterion": str(criterion.get("name")),
                    "direction": str(criterion.get("direction_label", "max")),
                    "ahp_winner_value": float(value_matrix[criterion_index, ahp_winner_index]),
                    "ga_winner_value": float(value_matrix[criterion_index, ga_winner_index]),
                    "value_delta": float(
                        value_matrix[criterion_index, ahp_winner_index]
                        - value_matrix[criterion_index, ga_winner_index]
                    ),
                    "ahp_winner_utility": float(utility_matrix[criterion_index, ahp_winner_index]),
                    "ga_winner_utility": float(utility_matrix[criterion_index, ga_winner_index]),
                    "utility_delta": float(
                        utility_matrix[criterion_index, ahp_winner_index]
                        - utility_matrix[criterion_index, ga_winner_index]
                    ),
                }
            )

        advantages = sorted(
            [item for item in criterion_deltas if item["utility_delta"] > 1e-9],
            key=lambda item: item["utility_delta"],
            reverse=True,
        )
        tradeoffs = sorted(
            [item for item in criterion_deltas if item["utility_delta"] < -1e-9],
            key=lambda item: item["utility_delta"],
        )
        same_winner = ahp_winner_index == ga_winner_index
        ga_score_delta = self._safe_score_delta(
            candidates[ahp_winner_index].get("score"),
            candidates[ga_winner_index].get("score"),
        )
        ahp_score_delta = float(ahp_scores[ahp_winner_index] - ahp_scores[ga_winner_index])
        interpretation = self._winner_interpretation(
            same_winner=same_winner,
            ahp_winner_id=alt_ids[ahp_winner_index],
            ga_winner_id=alt_ids[ga_winner_index],
            ahp_score_delta=ahp_score_delta,
            ga_score_delta=ga_score_delta,
            ahp_winner_ahp_score=float(ahp_scores[ahp_winner_index]),
            ga_winner_ahp_score=float(ahp_scores[ga_winner_index]),
            ahp_winner_ga_score=candidates[ahp_winner_index].get("score"),
            ga_winner_ga_score=candidates[ga_winner_index].get("score"),
            criterion_deltas=criterion_deltas,
        )

        return {
            "same_winner": same_winner,
            "ahp_winner": {
                "id": alt_ids[ahp_winner_index],
                "name": self._candidate_name(candidates[ahp_winner_index], alt_ids[ahp_winner_index]),
                "ga_rank": ga_rank_by_index[ahp_winner_index],
                "ahp_rank": 1,
                "ga_score": candidates[ahp_winner_index].get("score"),
                "ahp_score": float(ahp_scores[ahp_winner_index]),
            },
            "ga_winner": {
                "id": alt_ids[ga_winner_index],
                "name": self._candidate_name(candidates[ga_winner_index], alt_ids[ga_winner_index]),
                "ga_rank": ga_rank_by_index[ga_winner_index],
                "ahp_rank": int(ranked_indices.index(ga_winner_index) + 1),
                "ga_score": candidates[ga_winner_index].get("score"),
                "ahp_score": float(ahp_scores[ga_winner_index]),
            },
            "ga_score_delta": ga_score_delta,
            "ahp_score_delta": ahp_score_delta,
            "criterion_deltas": criterion_deltas,
            "utility_advantages": advantages[:3],
            "utility_tradeoffs": tradeoffs[:3],
            "summary": self._winner_comparison_summary(
                same_winner=same_winner,
                advantages=advantages,
                tradeoffs=tradeoffs,
            ),
            "interpretation": interpretation,
        }

    def _winner_interpretation(
        self,
        *,
        same_winner: bool,
        ahp_winner_id: str,
        ga_winner_id: str,
        ahp_score_delta: float,
        ga_score_delta: float | None,
        ahp_winner_ahp_score: float,
        ga_winner_ahp_score: float,
        ahp_winner_ga_score: Any,
        ga_winner_ga_score: Any,
        criterion_deltas: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        ahp_score_delta_percent = self._relative_percent(ahp_score_delta, ga_winner_ahp_score)
        ga_score_delta_percent = self._relative_percent(ga_score_delta, ga_winner_ga_score)
        key_deltas = self._key_metric_deltas(criterion_deltas)

        if same_winner:
            summary = (
                "AHP и GA выбрали одного лидера; дополнительная интерпретация "
                "сводится к проверке вклада критериев."
            )
            bullets = [summary]
        else:
            parts = [
                f"{self._format_signed_percent(ahp_score_delta_percent)} по AHP score",
                f"{self._format_signed_percent(ga_score_delta_percent)} по GA score",
            ]
            parts.extend(item["text"] for item in key_deltas[:3])
            summary = (
                f"AHP выбрал {ahp_winner_id} вместо GA-лидера {ga_winner_id}: "
                + ", ".join(parts)
                + "."
            )
            bullets = [
                f"Преимущество AHP-лидера над GA-лидером по AHP score: {self._format_signed_percent(ahp_score_delta_percent)}.",
                f"Изменение GA score относительно GA-лидера: {self._format_signed_percent(ga_score_delta_percent)}.",
            ]
            bullets.extend(item["sentence"] for item in key_deltas[:3])

        return {
            "summary": summary,
            "same_winner": same_winner,
            "ahp_winner_id": ahp_winner_id,
            "ga_winner_id": ga_winner_id,
            "ahp_score_delta": ahp_score_delta,
            "ahp_score_delta_percent": ahp_score_delta_percent,
            "ga_score_delta": ga_score_delta,
            "ga_score_delta_percent": ga_score_delta_percent,
            "ahp_winner_ahp_score": ahp_winner_ahp_score,
            "ga_winner_ahp_score": ga_winner_ahp_score,
            "ahp_winner_ga_score": self._finite_score_or_none(ahp_winner_ga_score),
            "ga_winner_ga_score": self._finite_score_or_none(ga_winner_ga_score),
            "key_metric_deltas": key_deltas,
            "bullets": bullets,
        }

    def _key_metric_deltas(self, criterion_deltas: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        result = []
        for criterion in ("capital_cost", "client_capacity", "total_power_watts"):
            item = next(
                (row for row in criterion_deltas if str(row.get("criterion")) == criterion),
                None,
            )
            if not item:
                continue
            delta = self._finite_score_or_none(item.get("value_delta"))
            if delta is None or abs(delta) <= 1e-9:
                continue
            text = self._metric_delta_text(criterion, delta)
            result.append(
                {
                    "criterion": criterion,
                    "value_delta": delta,
                    "direction": item.get("direction"),
                    "text": text,
                    "sentence": self._metric_delta_sentence(criterion, delta),
                    "ahp_winner_value": item.get("ahp_winner_value"),
                    "ga_winner_value": item.get("ga_winner_value"),
                    "utility_delta": item.get("utility_delta"),
                }
            )
        return result

    def _metric_delta_text(self, criterion: str, delta: float) -> str:
        if criterion == "capital_cost":
            return f"{self._format_signed_money(delta)} стоимости"
        if criterion == "client_capacity":
            return f"{self._format_signed_number(delta)} клиентских мест"
        if criterion == "total_power_watts":
            return f"{self._format_signed_number(delta)} Вт мощности"
        return f"{self._format_signed_number(delta)} по {criterion}"

    def _metric_delta_sentence(self, criterion: str, delta: float) -> str:
        if criterion == "capital_cost":
            if delta < 0:
                return f"AHP-лидер дешевле GA-лидера на {self._format_money(abs(delta))}."
            return f"AHP-лидер дороже GA-лидера на {self._format_money(abs(delta))}."
        if criterion == "client_capacity":
            if delta < 0:
                return f"AHP-лидер даёт на {self._format_plain_number(abs(delta))} клиентских мест меньше."
            return f"AHP-лидер даёт на {self._format_plain_number(abs(delta))} клиентских мест больше."
        if criterion == "total_power_watts":
            if delta < 0:
                return f"AHP-лидер потребляет на {self._format_plain_number(abs(delta))} Вт меньше."
            return f"AHP-лидер потребляет на {self._format_plain_number(abs(delta))} Вт больше."
        return f"Разница по критерию {criterion}: {self._format_signed_number(delta)}."

    def _relative_percent(self, delta: float | None, reference: Any) -> float | None:
        if delta is None:
            return None
        reference_value = self._finite_score_or_none(reference)
        if reference_value is None or abs(reference_value) <= 1e-12:
            return None
        return 100.0 * float(delta) / abs(reference_value)

    def _finite_score_or_none(self, value: Any) -> float | None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return numeric if np.isfinite(numeric) else None

    def _format_signed_percent(self, value: float | None) -> str:
        if value is None:
            return "нет данных"
        return f"{value:+.2f}%"

    def _format_signed_money(self, value: float) -> str:
        sign = "+" if value > 0 else "-" if value < 0 else ""
        return f"{sign}{self._format_money(abs(value))}"

    def _format_money(self, value: float) -> str:
        return f"{self._format_plain_number(value)} руб."

    def _format_signed_number(self, value: float) -> str:
        sign = "+" if value > 0 else "-" if value < 0 else ""
        return f"{sign}{self._format_plain_number(abs(value))}"

    def _safe_score_delta(self, left: Any, right: Any) -> float | None:
        try:
            left_value = float(left)
            right_value = float(right)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(left_value) or not np.isfinite(right_value):
            return None
        return left_value - right_value

    def _winner_comparison_summary(
        self,
        *,
        same_winner: bool,
        advantages: Sequence[Mapping[str, Any]],
        tradeoffs: Sequence[Mapping[str, Any]],
    ) -> str:
        if same_winner:
            return "AHP и GA выбрали одного лидера."

        advantage_names = ", ".join(str(item["criterion"]) for item in advantages[:2])
        tradeoff_names = ", ".join(str(item["criterion"]) for item in tradeoffs[:2])
        if advantage_names and tradeoff_names:
            return (
                "AHP выбрал другого лидера: он сильнее по критериям "
                f"{advantage_names}, но слабее по критериям {tradeoff_names}."
            )
        if advantage_names:
            return f"AHP выбрал другого лидера: он сильнее по критериям {advantage_names}."
        return "AHP выбрал другого лидера; требуется проверить вклад критериев."

    def _agreement_warnings(
        self,
        *,
        status: str,
        candidate_count: int,
        winner_ga_rank: int,
        max_rank_delta: int,
        top_3_overlap: int,
        criterion_diagnostics: Mapping[str, Any] | None = None,
    ) -> list[str]:
        warnings = []
        diagnostics_warnings = (
            criterion_diagnostics.get("warnings", [])
            if isinstance(criterion_diagnostics, Mapping)
            else []
        )
        if isinstance(diagnostics_warnings, list):
            warnings.extend(str(item) for item in diagnostics_warnings if str(item).strip())
        if candidate_count > 1 and winner_ga_rank > min(5, candidate_count):
            warnings.append(
                "AHP выбрал вариант, который находится за пределами GA top-5; "
                "результат нужно рассматривать как спорный компромисс."
            )
        if status in {"low", "conflict"} and max_rank_delta >= 4:
            warnings.append(
                "Обнаружен сильный скачок ранга между GA и AHP; "
                "проверьте веса и вклад отдельных критериев."
            )
        if status == "conflict" and top_3_overlap == 0:
            warnings.append("Top-3 GA и top-3 AHP не имеют общих вариантов.")
        return warnings

    def _agreement_summary(
        self,
        *,
        status_label: str,
        top_3_overlap: int,
        top_5_overlap: int,
        max_rank_delta: int,
        winner_ga_rank: int,
    ) -> str:
        return (
            f"Согласованность GA и AHP: {status_label}. "
            f"Пересечение top-3: {top_3_overlap}, пересечение top-5: {top_5_overlap}. "
            f"Максимальный сдвиг ранга: {max_rank_delta}. "
            f"AHP-лидер имеет GA-ранг {winner_ga_rank}."
        )

    def _candidate_pool_snapshot(
        self, candidates: Sequence[Mapping[str, Any]], alt_ids: Sequence[str]
    ) -> list[dict[str, Any]]:
        pool = []
        for candidate, alt_id in zip(candidates, alt_ids):
            pool.append(
                {
                    "id": alt_id,
                    "name": self._candidate_name(candidate, alt_id),
                    "ga_rank": candidate.get("rank"),
                    "ga_score": candidate.get("score"),
                    "totals": deepcopy(candidate.get("totals", {})),
                    "selected_items": deepcopy(candidate.get("selected_items", [])),
                }
            )
        return pool

    def _matrix_by_candidate(
        self,
        alt_ids: Sequence[str],
        criteria: Sequence[Mapping[str, Any]],
        matrix: np.ndarray,
    ) -> list[dict[str, Any]]:
        names = [str(criterion.get("name")) for criterion in criteria]
        by_candidate = []
        for candidate_index, alt_id in enumerate(alt_ids):
            by_candidate.append(
                {
                    "id": alt_id,
                    "values": {
                        name: float(matrix[criterion_index, candidate_index])
                        for criterion_index, name in enumerate(names)
                    },
                }
            )
        return by_candidate

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
        ga_result = (
            genetic_summary.get("ga_result", {})
            if isinstance(genetic_summary.get("ga_result"), Mapping)
            else {}
        )
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
