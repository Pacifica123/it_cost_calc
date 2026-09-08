"""Scenario sensitivity analysis over Hybrid rows stored in DecisionReport.

The service is deliberately post-analytical. It does not rerun GA or AHP and it
never invents candidates that are absent from the exported shared pool.
Instead it reproduces the transparent Hybrid formula for alternative lambda
values and can apply a TCO ceiling to the already exported alternatives.

Two details are intentionally explicit:
- lambda stability is solved from exact pairwise intersections of the affine
  Hybrid score lines; the sampled sweep is retained only for visualization;
- TCO scenarios use normalization anchored to the exported baseline pool so a
  budget filter changes feasibility only. A subset-renormalized grid is kept as
  a diagnostic to expose the normalization-context effect.
"""

from __future__ import annotations

from copy import deepcopy
from math import isfinite
from typing import Any, Mapping, Sequence


class DecisionSensitivityAnalysisService:
    """Build a reproducible what-if model from ``DecisionReport`` Hybrid rows."""

    DEFAULT_LAMBDA_STEP = 0.05
    MAX_BUDGET_LEVELS = 24
    _EPSILON = 1e-12

    def build(self, report: Mapping[str, Any]) -> dict[str, Any]:
        analysis_results = self._mapping(report.get("analysis_results"))
        hybrid_root = self._mapping(analysis_results.get("hybrid_assessment"))
        scoped = hybrid_root.get("by_scope")

        if isinstance(scoped, Mapping):
            raw_scopes = [
                (str(scope), payload)
                for scope, payload in scoped.items()
                if isinstance(payload, Mapping)
            ]
        elif hybrid_root:
            raw_scopes = [("all", hybrid_root)]
        else:
            raw_scopes = []

        scopes: dict[str, Any] = {}
        for scope, payload in raw_scopes:
            scopes[scope] = self._build_scope(scope, payload)

        warnings: list[str] = []
        if not scopes:
            warnings.append(
                "В DecisionReport нет результата Hybrid: сначала нужно сформировать общий пул, "
                "выполнить GA/AHP и запустить гибридную оценку."
            )

        return {
            "schema_version": 2,
            "title": "Интерактивный анализ чувствительности решения",
            "project": deepcopy(self._mapping(report.get("project"))),
            "method": {
                "formula": "H_i(lambda) = lambda * GA_norm_i + (1 - lambda) * AHP_norm_i",
                "lambda_range": [0.0, 1.0],
                "lambda_step": self.DEFAULT_LAMBDA_STEP,
                "lambda_solver": "exact_pairwise_affine_intersections",
                "lambda_sampling_role": "visualization_only",
                "lambda_interpretation": (
                    "lambda=0 усиливает вклад AHP, lambda=1 усиливает вклад GA, "
                    "lambda=0.5 соответствует нейтральному компромиссу."
                ),
                "budget_interpretation": (
                    "TCO-порог является post-analysis фильтром уже экспортированного пула. "
                    "Он не создаёт новые альтернативы и не перезапускает GA/AHP."
                ),
                "tco_normalization_policy": (
                    "primary=fixed_baseline_normalization; diagnostic=subset_renormalization"
                ),
            },
            "scopes": scopes,
            "warnings": warnings,
            "limitations": [
                "Lambda-сценарии пересчитывают только прозрачный Hybrid-слой над сохранёнными GA/AHP-score.",
                "TCO-сценарии исследуют допустимость только среди уже экспортированных альтернатив.",
                "Основная TCO-карта сохраняет baseline min-max диапазоны; это изолирует эффект допустимости от перенормировки.",
                "Диагностическая subset-renormalization карта показывает контекстный эффект, но не используется как основной вывод.",
                "Увеличение TCO-порога не восстанавливает кандидатов, которые отсутствовали в исходном общем пуле.",
                "Модуль является аналитической надстройкой ПУАЗ и не заменяет исходные методы выбора дипломного проекта.",
            ],
        }

    def _build_scope(self, scope: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        ranking = payload.get("ranking")
        if not isinstance(ranking, Sequence) or isinstance(ranking, (str, bytes)):
            ranking = []

        candidates = [
            candidate
            for item in ranking
            if isinstance(item, Mapping)
            if (candidate := self._candidate_row(item)) is not None
        ]
        if not candidates:
            return {
                "scope": scope,
                "status": "incomplete",
                "reason": "Hybrid-результат не содержит сопоставимых строк GA/AHP.",
                "candidates": [],
                "lambda_values": self._lambda_values(),
                "budget_levels": [None],
                "lambda_sweep": [],
                "grid": [],
                "subset_renormalization_grid": [],
                "exact_lambda": {},
                "warnings": ["Для анализа чувствительности нужны GA-score и AHP-score в Hybrid ranking."],
            }

        lambda_values = self._lambda_values()
        baseline_lambda = self._bounded_lambda(payload.get("lambda", 0.5))
        budget_levels = [None, *self._budget_levels(candidates)]
        baseline_reference = self._normalization_reference(candidates)

        lambda_sweep = [
            self._scenario(
                candidates,
                lambda_value=value,
                tco_limit=None,
                normalization_reference=baseline_reference,
            )
            for value in lambda_values
        ]
        baseline = self._scenario(
            candidates,
            lambda_value=baseline_lambda,
            tco_limit=None,
            normalization_reference=baseline_reference,
        )

        grid = self._scenario_grid(
            candidates,
            budget_levels=budget_levels,
            lambda_values=lambda_values,
            normalization_reference=baseline_reference,
        )
        subset_renormalization_grid = self._scenario_grid(
            candidates,
            budget_levels=budget_levels,
            lambda_values=lambda_values,
            normalization_reference=None,
        )

        exact_lambda = self._exact_lambda_analysis(
            candidates,
            normalization_reference=baseline_reference,
            baseline_winner_id=baseline.get("winner_id"),
        )
        stability = exact_lambda["stability"]

        original_winner_id = str(
            payload.get("winner_id") or self._mapping(payload.get("winner")).get("id") or ""
        )
        finite_tco_count = sum(1 for candidate in candidates if candidate.get("tco") is not None)
        warnings: list[str] = []
        if finite_tco_count < len(candidates):
            warnings.append(
                f"У {len(candidates) - finite_tco_count} альтернатив нет конечного TCO; "
                "при включении TCO-порога они считаются недопустимыми для этого сценария."
            )
        if baseline.get("winner_id") and original_winner_id and baseline.get("winner_id") != original_winner_id:
            warnings.append(
                "Базовый пересчёт Hybrid не совпал с сохранённым winner_id; проверьте полноту экспортированных score."
            )
        if self._grid_has_context_effect(grid, subset_renormalization_grid):
            warnings.append(
                "Subset-renormalization меняет хотя бы один TCO×lambda сценарий относительно фиксированной baseline-нормировки; "
                "для основного вывода используется фиксированная нормировка."
            )

        return {
            "scope": scope,
            "status": "ok",
            "source_label": payload.get("source_label"),
            "candidates": candidates,
            "lambda_values": lambda_values,
            "budget_levels": budget_levels,
            "normalization": {
                "primary_tco_mode": "fixed_baseline_normalization",
                "diagnostic_tco_mode": "subset_renormalization",
                "baseline_reference": baseline_reference,
            },
            "baseline": {
                **baseline,
                "original_hybrid_winner_id": original_winner_id or None,
                "matches_original_hybrid": (
                    not original_winner_id or baseline.get("winner_id") == original_winner_id
                ),
            },
            "lambda_sweep": lambda_sweep,
            "exact_lambda": exact_lambda,
            "grid": grid,
            "subset_renormalization_grid": subset_renormalization_grid,
            "stability": stability,
            "warnings": warnings,
        }

    def _candidate_row(self, item: Mapping[str, Any]) -> dict[str, Any] | None:
        candidate_id = str(item.get("id") or item.get("candidate_id") or "")
        ga_score = self._finite_float_or_none(item.get("ga_score"))
        ahp_score = self._finite_float_or_none(item.get("ahp_score"))
        if not candidate_id or (ga_score is None and ahp_score is None):
            return None

        totals = self._mapping(item.get("totals"))
        tco_block = self._mapping(totals.get("tco"))
        tco = self._finite_float_or_none(
            tco_block.get("total_ownership_cost", totals.get("total_ownership_cost"))
        )
        return {
            "id": candidate_id,
            "name": str(item.get("name") or candidate_id),
            "ga_rank": self._positive_int(item.get("ga_rank"), fallback=999999),
            "ahp_rank": self._positive_int(item.get("ahp_rank"), fallback=999999),
            "ga_score": ga_score,
            "ahp_score": ahp_score,
            "pareto_status": str(item.get("pareto_status") or "нет данных"),
            "tco": tco,
        }

    def _scenario_grid(
        self,
        candidates: Sequence[Mapping[str, Any]],
        *,
        budget_levels: Sequence[float | None],
        lambda_values: Sequence[float],
        normalization_reference: Mapping[str, Mapping[str, float]] | None,
    ) -> list[dict[str, Any]]:
        grid: list[dict[str, Any]] = []
        for budget in budget_levels:
            for lambda_value in lambda_values:
                scenario = self._scenario(
                    candidates,
                    lambda_value=lambda_value,
                    tco_limit=budget,
                    normalization_reference=normalization_reference,
                )
                grid.append(
                    {
                        "lambda": lambda_value,
                        "tco_limit": budget,
                        "winner_id": scenario.get("winner_id"),
                        "winner_name": scenario.get("winner_name"),
                        "feasible_count": scenario.get("feasible_count", 0),
                        "ranking_ids": [str(row.get("id")) for row in scenario.get("ranking", [])],
                    }
                )
        return grid

    def _scenario(
        self,
        candidates: Sequence[Mapping[str, Any]],
        *,
        lambda_value: float,
        tco_limit: float | None,
        normalization_reference: Mapping[str, Mapping[str, float]] | None = None,
    ) -> dict[str, Any]:
        lambda_value = self._bounded_lambda(lambda_value)
        feasible = [
            candidate
            for candidate in candidates
            if tco_limit is None
            or (
                candidate.get("tco") is not None
                and float(candidate["tco"]) <= float(tco_limit) + 1e-9
            )
        ]
        if not feasible:
            return {
                "lambda": lambda_value,
                "tco_limit": tco_limit,
                "feasible_count": 0,
                "winner_id": None,
                "winner_name": None,
                "ranking": [],
            }

        if normalization_reference is None:
            active_reference = self._normalization_reference(feasible)
        else:
            active_reference = normalization_reference

        ranking: list[dict[str, Any]] = []
        for index, candidate in enumerate(feasible):
            candidate_id = str(candidate.get("id") or "")
            normalized = active_reference.get(candidate_id, {})
            ga_value = float(normalized.get("ga", 0.5))
            ahp_value = float(normalized.get("ahp", 0.5))
            score = lambda_value * ga_value + (1.0 - lambda_value) * ahp_value
            disagreement = abs(
                self._positive_int(candidate.get("ga_rank"), fallback=index + 1)
                - self._positive_int(candidate.get("ahp_rank"), fallback=index + 1)
            )
            ranking.append(
                {
                    "id": candidate.get("id"),
                    "name": candidate.get("name"),
                    "ga_score_normalized": ga_value,
                    "ahp_score_normalized": ahp_value,
                    "hybrid_score": score,
                    "rank_disagreement": disagreement,
                    "pareto_status": candidate.get("pareto_status"),
                    "tco": candidate.get("tco"),
                }
            )

        ranking.sort(
            key=lambda row: (
                float(row["hybrid_score"]),
                float(row["ga_score_normalized"]) + float(row["ahp_score_normalized"]),
                str(row.get("pareto_status")) == "недоминируемая",
                -int(row.get("rank_disagreement") or 0),
            ),
            reverse=True,
        )
        for rank, row in enumerate(ranking, start=1):
            row["rank"] = rank

        winner = ranking[0]
        return {
            "lambda": lambda_value,
            "tco_limit": tco_limit,
            "feasible_count": len(feasible),
            "winner_id": winner.get("id"),
            "winner_name": winner.get("name"),
            "ranking": ranking,
        }

    def _exact_lambda_analysis(
        self,
        candidates: Sequence[Mapping[str, Any]],
        *,
        normalization_reference: Mapping[str, Mapping[str, float]],
        baseline_winner_id: Any,
    ) -> dict[str, Any]:
        lines: list[dict[str, Any]] = []
        for candidate in candidates:
            candidate_id = str(candidate.get("id") or "")
            normalized = normalization_reference.get(candidate_id, {})
            ahp_value = float(normalized.get("ahp", 0.5))
            ga_value = float(normalized.get("ga", 0.5))
            lines.append(
                {
                    "id": candidate_id,
                    "name": str(candidate.get("name") or candidate_id),
                    "intercept": ahp_value,
                    "slope": ga_value - ahp_value,
                    "ga_normalized": ga_value,
                    "ahp_normalized": ahp_value,
                }
            )

        pairwise_crossings: list[dict[str, Any]] = []
        breakpoints = {0.0, 1.0}
        for left_index, left in enumerate(lines):
            for right in lines[left_index + 1 :]:
                slope_delta = float(left["slope"]) - float(right["slope"])
                intercept_delta = float(right["intercept"]) - float(left["intercept"])
                if abs(slope_delta) <= self._EPSILON:
                    identical = abs(intercept_delta) <= self._EPSILON
                    pairwise_crossings.append(
                        {
                            "candidate_a": left["id"],
                            "candidate_b": right["id"],
                            "lambda": None,
                            "relation": "identical" if identical else "parallel",
                        }
                    )
                    continue
                crossing = intercept_delta / slope_delta
                if -self._EPSILON <= crossing <= 1.0 + self._EPSILON:
                    bounded = min(1.0, max(0.0, float(crossing)))
                    breakpoints.add(round(bounded, 12))
                    pairwise_crossings.append(
                        {
                            "candidate_a": left["id"],
                            "candidate_b": right["id"],
                            "lambda": bounded,
                            "relation": "single_crossing",
                        }
                    )
                else:
                    pairwise_crossings.append(
                        {
                            "candidate_a": left["id"],
                            "candidate_b": right["id"],
                            "lambda": float(crossing),
                            "relation": "outside_range",
                        }
                    )

        ordered = sorted(breakpoints)
        breakpoint_winners = [
            {
                "lambda": point,
                "winner_id": scenario.get("winner_id"),
                "winner_name": scenario.get("winner_name"),
            }
            for point in ordered
            for scenario in [
                self._scenario(
                    candidates,
                    lambda_value=point,
                    tco_limit=None,
                    normalization_reference=normalization_reference,
                )
            ]
        ]

        raw_intervals: list[dict[str, Any]] = []
        for left, right in zip(ordered, ordered[1:]):
            if right - left <= self._EPSILON:
                continue
            midpoint = (left + right) / 2.0
            scenario = self._scenario(
                candidates,
                lambda_value=midpoint,
                tco_limit=None,
                normalization_reference=normalization_reference,
            )
            raw_intervals.append(
                {
                    "winner_id": scenario.get("winner_id"),
                    "winner_name": scenario.get("winner_name"),
                    "lambda_from": left,
                    "lambda_to": right,
                    "lambda_span": right - left,
                }
            )

        winner_intervals: list[dict[str, Any]] = []
        for interval in raw_intervals:
            if (
                winner_intervals
                and winner_intervals[-1].get("winner_id") == interval.get("winner_id")
                and abs(float(winner_intervals[-1]["lambda_to"]) - float(interval["lambda_from"]))
                <= self._EPSILON
            ):
                winner_intervals[-1]["lambda_to"] = interval["lambda_to"]
                winner_intervals[-1]["lambda_span"] = (
                    float(winner_intervals[-1]["lambda_to"])
                    - float(winner_intervals[-1]["lambda_from"])
                )
            else:
                winner_intervals.append(dict(interval))

        spans: dict[str, float] = {}
        names: dict[str, str] = {}
        for interval in winner_intervals:
            winner_id = interval.get("winner_id")
            if not winner_id:
                continue
            key = str(winner_id)
            spans[key] = spans.get(key, 0.0) + float(interval.get("lambda_span") or 0.0)
            names[key] = str(interval.get("winner_name") or key)

        shares = sorted(
            (
                {
                    "id": candidate_id,
                    "name": names.get(candidate_id, candidate_id),
                    "lambda_span": span,
                    "share_pct": round(100.0 * span, 4),
                }
                for candidate_id, span in spans.items()
            ),
            key=lambda item: (-float(item["share_pct"]), str(item["id"])),
        )
        baseline_key = str(baseline_winner_id) if baseline_winner_id else ""
        baseline_span = spans.get(baseline_key, 0.0)
        switches = max(len(winner_intervals) - 1, 0)
        switch_points = [
            {
                "lambda": winner_intervals[index]["lambda_from"],
                "from_winner_id": winner_intervals[index - 1].get("winner_id"),
                "to_winner_id": winner_intervals[index].get("winner_id"),
            }
            for index in range(1, len(winner_intervals))
        ]

        return {
            "line_model": "affine_in_lambda_on_fixed_pool",
            "candidate_lines": lines,
            "pairwise_crossings": pairwise_crossings,
            "breakpoints": ordered,
            "breakpoint_winners": breakpoint_winners,
            "winner_switch_points": switch_points,
            "winner_intervals": winner_intervals,
            "boundary_note": (
                "winner_intervals describe open regions between analytical breakpoints; "
                "breakpoint_winners records the deterministic tie-break exactly at each boundary."
            ),
            "stability": {
                "method": "exact_lambda_measure",
                "baseline_winner_stability_pct": round(100.0 * baseline_span, 4),
                "switch_count": switches,
                "winner_shares": shares,
                "winner_intervals": winner_intervals,
            },
        }

    def _normalization_reference(
        self, candidates: Sequence[Mapping[str, Any]]
    ) -> dict[str, dict[str, float]]:
        ga_values = [self._finite_float_or_none(row.get("ga_score")) for row in candidates]
        ahp_values = [self._finite_float_or_none(row.get("ahp_score")) for row in candidates]
        ga_normalized = self._normalize(ga_values)
        ahp_normalized = self._normalize(ahp_values)
        return {
            str(candidate.get("id") or index): {
                "ga": ga_normalized[index],
                "ahp": ahp_normalized[index],
            }
            for index, candidate in enumerate(candidates)
        }

    def _grid_has_context_effect(
        self,
        fixed_grid: Sequence[Mapping[str, Any]],
        subset_grid: Sequence[Mapping[str, Any]],
    ) -> bool:
        for fixed, subset in zip(fixed_grid, subset_grid):
            if fixed.get("ranking_ids") != subset.get("ranking_ids"):
                return True
        return False

    def _budget_levels(self, candidates: Sequence[Mapping[str, Any]]) -> list[float]:
        values = sorted(
            {
                round(float(value), 2)
                for candidate in candidates
                if (value := candidate.get("tco")) is not None
            }
        )
        if len(values) <= self.MAX_BUDGET_LEVELS:
            return values

        last = len(values) - 1
        indexes = {
            round(position * last / (self.MAX_BUDGET_LEVELS - 1))
            for position in range(self.MAX_BUDGET_LEVELS)
        }
        return [values[index] for index in sorted(indexes)]

    def _lambda_values(self) -> list[float]:
        count = round(1.0 / self.DEFAULT_LAMBDA_STEP)
        return [round(index * self.DEFAULT_LAMBDA_STEP, 2) for index in range(count + 1)]

    def _normalize(self, values: Sequence[float | None]) -> list[float]:
        finite = [float(value) for value in values if value is not None]
        if not finite:
            return [0.5 for _ in values]
        minimum = min(finite)
        maximum = max(finite)
        if len(values) == 1 and len(finite) == 1:
            return [1.0]
        if abs(maximum - minimum) <= self._EPSILON:
            return [0.5 for _ in values]
        return [
            ((float(value) if value is not None else minimum) - minimum) / (maximum - minimum)
            for value in values
        ]

    def _bounded_lambda(self, value: Any) -> float:
        numeric = self._finite_float_or_none(value)
        if numeric is None:
            return 0.5
        return min(1.0, max(0.0, numeric))

    @staticmethod
    def _mapping(value: Any) -> dict[str, Any]:
        return dict(value) if isinstance(value, Mapping) else {}

    @staticmethod
    def _positive_int(value: Any, *, fallback: int) -> int:
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            return fallback
        return numeric if numeric > 0 else fallback

    @staticmethod
    def _finite_float_or_none(value: Any) -> float | None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return numeric if isfinite(numeric) else None


__all__ = ["DecisionSensitivityAnalysisService"]
