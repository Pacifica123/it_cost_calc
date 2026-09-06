"""Scenario sensitivity analysis over Hybrid rows stored in DecisionReport.

The service is deliberately post-analytical. It does not rerun GA or AHP and it
never invents candidates that are absent from the exported shared pool.
Instead it reproduces the transparent Hybrid formula for alternative lambda
values and can apply a TCO ceiling to the already exported alternatives.
"""

from __future__ import annotations

from copy import deepcopy
from math import isfinite
from typing import Any, Mapping, Sequence


class DecisionSensitivityAnalysisService:
    """Build a reproducible what-if model from ``DecisionReport`` Hybrid rows."""

    DEFAULT_LAMBDA_STEP = 0.05
    MAX_BUDGET_LEVELS = 24

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
            "schema_version": 1,
            "title": "Интерактивный анализ чувствительности решения",
            "project": deepcopy(self._mapping(report.get("project"))),
            "method": {
                "formula": "H_i(lambda) = lambda * GA_norm_i + (1 - lambda) * AHP_norm_i",
                "lambda_range": [0.0, 1.0],
                "lambda_step": self.DEFAULT_LAMBDA_STEP,
                "lambda_interpretation": (
                    "lambda=0 усиливает вклад AHP, lambda=1 усиливает вклад GA, "
                    "lambda=0.5 соответствует нейтральному компромиссу."
                ),
                "budget_interpretation": (
                    "TCO-порог является post-analysis фильтром уже экспортированного пула. "
                    "Он не создаёт новые альтернативы и не перезапускает GA/AHP."
                ),
            },
            "scopes": scopes,
            "warnings": warnings,
            "limitations": [
                "Lambda-сценарии пересчитывают только прозрачный Hybrid-слой над сохранёнными GA/AHP-score.",
                "TCO-сценарии исследуют допустимость только среди уже экспортированных альтернатив.",
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
                "warnings": ["Для анализа чувствительности нужны GA-score и AHP-score в Hybrid ranking."],
            }

        lambda_values = self._lambda_values()
        baseline_lambda = self._bounded_lambda(payload.get("lambda", 0.5))
        budget_levels = [None, *self._budget_levels(candidates)]

        lambda_sweep = [
            self._scenario(candidates, lambda_value=value, tco_limit=None)
            for value in lambda_values
        ]
        baseline = self._scenario(candidates, lambda_value=baseline_lambda, tco_limit=None)

        grid: list[dict[str, Any]] = []
        for budget in budget_levels:
            for lambda_value in lambda_values:
                scenario = self._scenario(
                    candidates,
                    lambda_value=lambda_value,
                    tco_limit=budget,
                )
                grid.append(
                    {
                        "lambda": lambda_value,
                        "tco_limit": budget,
                        "winner_id": scenario.get("winner_id"),
                        "winner_name": scenario.get("winner_name"),
                        "feasible_count": scenario.get("feasible_count", 0),
                    }
                )

        original_winner_id = str(payload.get("winner_id") or self._mapping(payload.get("winner")).get("id") or "")
        stability = self._stability(lambda_sweep, baseline.get("winner_id"))
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

        return {
            "scope": scope,
            "status": "ok",
            "source_label": payload.get("source_label"),
            "candidates": candidates,
            "lambda_values": lambda_values,
            "budget_levels": budget_levels,
            "baseline": {
                **baseline,
                "original_hybrid_winner_id": original_winner_id or None,
                "matches_original_hybrid": (
                    not original_winner_id or baseline.get("winner_id") == original_winner_id
                ),
            },
            "lambda_sweep": lambda_sweep,
            "grid": grid,
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

    def _scenario(
        self,
        candidates: Sequence[Mapping[str, Any]],
        *,
        lambda_value: float,
        tco_limit: float | None,
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

        ga_normalized = self._normalize([self._finite_float_or_none(row.get("ga_score")) for row in feasible])
        ahp_normalized = self._normalize([self._finite_float_or_none(row.get("ahp_score")) for row in feasible])

        ranking: list[dict[str, Any]] = []
        for index, candidate in enumerate(feasible):
            score = lambda_value * ga_normalized[index] + (1.0 - lambda_value) * ahp_normalized[index]
            disagreement = abs(
                self._positive_int(candidate.get("ga_rank"), fallback=index + 1)
                - self._positive_int(candidate.get("ahp_rank"), fallback=index + 1)
            )
            ranking.append(
                {
                    "id": candidate.get("id"),
                    "name": candidate.get("name"),
                    "ga_score_normalized": ga_normalized[index],
                    "ahp_score_normalized": ahp_normalized[index],
                    "hybrid_score": score,
                    "rank_disagreement": disagreement,
                    "pareto_status": candidate.get("pareto_status"),
                    "tco": candidate.get("tco"),
                }
            )

        ranking.sort(
            key=lambda row: (
                float(row["hybrid_score"]),
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

    def _stability(
        self,
        sweep: Sequence[Mapping[str, Any]],
        baseline_winner_id: Any,
    ) -> dict[str, Any]:
        winner_ids = [row.get("winner_id") for row in sweep if row.get("winner_id")]
        if not winner_ids:
            return {
                "baseline_winner_stability_pct": 0.0,
                "switch_count": 0,
                "winner_shares": [],
                "winner_intervals": [],
            }

        counts: dict[str, int] = {}
        names: dict[str, str] = {}
        for row in sweep:
            winner_id = row.get("winner_id")
            if not winner_id:
                continue
            key = str(winner_id)
            counts[key] = counts.get(key, 0) + 1
            names[key] = str(row.get("winner_name") or key)

        denominator = len(winner_ids)
        shares = sorted(
            (
                {
                    "id": candidate_id,
                    "name": names.get(candidate_id, candidate_id),
                    "wins": wins,
                    "share_pct": round(100.0 * wins / denominator, 1),
                }
                for candidate_id, wins in counts.items()
            ),
            key=lambda item: (-float(item["share_pct"]), str(item["id"])),
        )

        switches = 0
        previous: Any = None
        for winner_id in winner_ids:
            if previous is not None and winner_id != previous:
                switches += 1
            previous = winner_id

        intervals: list[dict[str, Any]] = []
        start_index = 0
        for index in range(1, len(sweep) + 1):
            current_id = sweep[start_index].get("winner_id")
            boundary = index == len(sweep) or sweep[index].get("winner_id") != current_id
            if not boundary:
                continue
            start_lambda = sweep[start_index].get("lambda")
            end_lambda = sweep[index - 1].get("lambda")
            intervals.append(
                {
                    "winner_id": current_id,
                    "winner_name": sweep[start_index].get("winner_name"),
                    "lambda_from": start_lambda,
                    "lambda_to": end_lambda,
                }
            )
            start_index = index

        baseline_key = str(baseline_winner_id) if baseline_winner_id else ""
        baseline_wins = counts.get(baseline_key, 0)
        return {
            "baseline_winner_stability_pct": round(100.0 * baseline_wins / denominator, 1),
            "switch_count": switches,
            "winner_shares": shares,
            "winner_intervals": intervals,
        }

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
        if abs(maximum - minimum) <= 1e-12:
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
