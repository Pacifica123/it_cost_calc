"""Hybrid decision assessment over the shared ПО/ТО candidate pool.

The service is intentionally lightweight.  It does not run GA, AHP or Pareto by
itself; it only combines the already visible results for the same scoped
candidate pool:

* GA rank/score is read from candidate metadata produced by the GA pool adapter;
* AHP score is read from the last standalone AHP report;
* Pareto is used as a status/warning, not as an extra hidden numeric weight.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from domain import CandidateConfiguration, ensure_candidate_configuration


@dataclass(frozen=True, slots=True)
class _CandidateRow:
    id: str
    name: str
    ga_rank: int
    ga_score: float | None
    totals: Mapping[str, Any]


class HybridDecisionAssessmentService:
    """Builds a transparent GA + AHP + Pareto-status summary for one scope."""

    def assess(
        self,
        candidates: Sequence[CandidateConfiguration | Mapping[str, Any]],
        *,
        ahp_report: Mapping[str, Any] | None = None,
        pareto_report: Mapping[str, Any] | None = None,
        lambda_value: float = 0.5,
        source_label: str = "общий пул альтернатив",
    ) -> dict[str, Any]:
        rows = self._candidate_rows(candidates)
        if not rows:
            return {
                "status": "incomplete",
                "reason": "Гибридная оценка не рассчитана: общий пул альтернатив пуст.",
                "candidate_count": 0,
                "missing_methods": ["candidate_pool", "ahp"],
                "ranking": [],
                "warnings": [],
                "source_label": source_label,
            }

        ahp_scores, ahp_rank_by_id = self._ahp_scores(ahp_report)
        pareto_status_by_id = self._pareto_status(pareto_report)
        missing_methods = []
        if not ahp_scores:
            missing_methods.append("ahp")

        if missing_methods:
            return {
                "status": "incomplete",
                "reason": "Гибридная оценка не рассчитана: сначала запустите самостоятельный AHP по текущему пулу.",
                "candidate_count": len(rows),
                "missing_methods": missing_methods,
                "ranking": [self._incomplete_row(row, pareto_status_by_id) for row in rows],
                "warnings": [
                    "Гибридная панель не запускает AHP автоматически, чтобы не возвращаться к схеме GA+AHP внутри GA."
                ],
                "source_label": source_label,
            }

        comparable = [row for row in rows if row.id in ahp_scores]
        if not comparable:
            return {
                "status": "incomplete",
                "reason": "Гибридная оценка не рассчитана: AHP-результат относится к другому набору альтернатив.",
                "candidate_count": len(rows),
                "missing_methods": ["matching_ahp_rows"],
                "ranking": [self._incomplete_row(row, pareto_status_by_id) for row in rows],
                "warnings": [
                    "ID альтернатив из общего пула не совпали с ID последнего AHP-отчёта."
                ],
                "source_label": source_label,
            }

        lambda_value = self._bounded_lambda(lambda_value)
        ga_scores = [row.ga_score for row in comparable]
        ahp_score_values = [ahp_scores[row.id] for row in comparable]
        ga_normalized, ga_norm_info = self._normalize_scores(ga_scores)
        ahp_normalized, ahp_norm_info = self._normalize_scores(ahp_score_values)

        ranking = []
        for index, row in enumerate(comparable):
            ga_part = lambda_value * ga_normalized[index]
            ahp_part = (1.0 - lambda_value) * ahp_normalized[index]
            ahp_rank = int(ahp_rank_by_id.get(row.id, index + 1))
            ranking.append(
                {
                    "id": row.id,
                    "name": row.name,
                    "ga_rank": row.ga_rank,
                    "ahp_rank": ahp_rank,
                    "ga_score": row.ga_score,
                    "ahp_score": ahp_scores[row.id],
                    "ga_score_normalized": ga_normalized[index],
                    "ahp_score_normalized": ahp_normalized[index],
                    "ga_contribution": ga_part,
                    "ahp_contribution": ahp_part,
                    "hybrid_score": ga_part + ahp_part,
                    "rank_disagreement": abs(row.ga_rank - ahp_rank),
                    "pareto_status": pareto_status_by_id.get(row.id, "нет данных"),
                    "totals": dict(row.totals),
                }
            )

        ranking = sorted(
            ranking,
            key=lambda item: (
                float(item["hybrid_score"]),
                str(item.get("pareto_status")) == "недоминируемая",
                -int(item.get("rank_disagreement") or 0),
            ),
            reverse=True,
        )
        for rank, item in enumerate(ranking, start=1):
            item["rank"] = rank
            item["comment"] = self._comment(item)

        warnings = self._warnings(
            ranking,
            rows=rows,
            comparable=comparable,
            pareto_report=pareto_report,
            ga_norm_info=ga_norm_info,
            ahp_norm_info=ahp_norm_info,
        )
        winner = ranking[0] if ranking else None
        return {
            "status": "ok",
            "method": "scoped_pool_ga_ahp_with_pareto_status",
            "lambda": lambda_value,
            "lambda_label": self._lambda_label(lambda_value),
            "candidate_count": len(rows),
            "used_candidate_count": len(comparable),
            "source_label": source_label,
            "ranking": ranking,
            "winner_id": winner.get("id") if winner else None,
            "winner_name": winner.get("name") if winner else None,
            "winner": winner,
            "normalization": {
                "ga": "minmax_over_shared_candidate_pool",
                "ahp": "minmax_over_standalone_ahp_result",
                "ga_details": ga_norm_info,
                "ahp_details": ahp_norm_info,
            },
            "warnings": warnings,
            "summary": self._summary(winner, warnings, lambda_value),
        }

    def _candidate_rows(
        self,
        candidates: Sequence[CandidateConfiguration | Mapping[str, Any]],
    ) -> list[_CandidateRow]:
        rows: list[_CandidateRow] = []
        for index, candidate in enumerate(candidates, start=1):
            model = ensure_candidate_configuration(candidate)
            rows.append(
                _CandidateRow(
                    id=model.id or f"candidate_{index}",
                    name=model.name or model.id or f"Альтернатива {index}",
                    ga_rank=self._positive_int(model.metadata.get("rank"), fallback=index),
                    ga_score=self._score_from_candidate(model),
                    totals=dict(model.totals),
                )
            )
        return rows

    def _score_from_candidate(self, candidate: CandidateConfiguration) -> float | None:
        for value in (
            candidate.metrics.get("ga_score"),
            candidate.metrics.get("score"),
            candidate.metadata.get("ga_score"),
        ):
            score = self._finite_float_or_none(value)
            if score is not None:
                return score
        return None

    def _ahp_scores(self, report: Mapping[str, Any] | None) -> tuple[dict[str, float], dict[str, int]]:
        if not isinstance(report, Mapping):
            return {}, {}
        nested = report.get("ahp_report")
        if isinstance(nested, Mapping):
            report = nested
        final = report.get("final") if isinstance(report.get("final"), Mapping) else {}
        ranking = final.get("ranking") if isinstance(final, Mapping) else None
        if ranking is None:
            ranking = report.get("ranking")

        scores: dict[str, float] = {}
        ranks: dict[str, int] = {}
        if not isinstance(ranking, Sequence) or isinstance(ranking, (str, bytes)):
            return scores, ranks

        for position, item in enumerate(ranking, start=1):
            candidate_id = ""
            score_value: Any = None
            rank = position
            if isinstance(item, Mapping):
                candidate_id = str(item.get("id") or item.get("alternative_id") or "")
                score_value = item.get("score", item.get("ahp_score", item.get("value")))
                rank = self._positive_int(item.get("rank"), fallback=position)
            elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) >= 2:
                candidate_id = str(item[0])
                score_value = item[1]
            score = self._finite_float_or_none(score_value)
            if candidate_id and score is not None:
                scores[candidate_id] = score
                ranks[candidate_id] = rank
        return scores, ranks

    def _pareto_status(self, report: Mapping[str, Any] | None) -> dict[str, str]:
        if not isinstance(report, Mapping):
            return {}
        result: dict[str, str] = {}
        nondominated = report.get("final_nondominated")
        if isinstance(nondominated, Sequence) and not isinstance(nondominated, (str, bytes)):
            for candidate_id in nondominated:
                result[str(candidate_id)] = "недоминируемая"
        ranking = report.get("ranking")
        if isinstance(ranking, Sequence) and not isinstance(ranking, (str, bytes)):
            for item in ranking:
                if not isinstance(item, Mapping):
                    continue
                candidate_id = str(item.get("id") or "")
                if not candidate_id:
                    continue
                is_nondominated = bool(item.get("nondominated"))
                result[candidate_id] = "недоминируемая" if is_nondominated else "доминируемая"
        return result

    def _incomplete_row(self, row: _CandidateRow, pareto_status_by_id: Mapping[str, str]) -> dict[str, Any]:
        return {
            "id": row.id,
            "name": row.name,
            "ga_rank": row.ga_rank,
            "ga_score": row.ga_score,
            "ahp_rank": None,
            "ahp_score": None,
            "hybrid_score": None,
            "rank_disagreement": None,
            "pareto_status": pareto_status_by_id.get(row.id, "нет данных"),
            "comment": "ожидает AHP-результат",
            "totals": dict(row.totals),
        }

    def _normalize_scores(self, values: Sequence[float | None]) -> tuple[list[float], dict[str, Any]]:
        finite = [float(value) for value in values if value is not None]
        if not finite:
            return [0.5 for _ in values], {
                "status": "no_finite_scores",
                "min": None,
                "max": None,
                "missing_count": len(values),
            }
        minimum = min(finite)
        maximum = max(finite)
        missing_count = len(values) - len(finite)
        if abs(maximum - minimum) <= 1e-12:
            return [0.5 for _ in values], {
                "status": "constant_score",
                "min": minimum,
                "max": maximum,
                "missing_count": missing_count,
            }
        normalized = []
        for value in values:
            numeric = float(value) if value is not None else minimum
            normalized.append((numeric - minimum) / (maximum - minimum))
        return normalized, {
            "status": "ok",
            "min": minimum,
            "max": maximum,
            "missing_count": missing_count,
        }

    def _warnings(
        self,
        ranking: Sequence[Mapping[str, Any]],
        *,
        rows: Sequence[_CandidateRow],
        comparable: Sequence[_CandidateRow],
        pareto_report: Mapping[str, Any] | None,
        ga_norm_info: Mapping[str, Any],
        ahp_norm_info: Mapping[str, Any],
    ) -> list[str]:
        warnings: list[str] = []
        if len(comparable) < len(rows):
            warnings.append("Часть альтернатив из общего пула отсутствует в последнем AHP-результате.")
        if not isinstance(pareto_report, Mapping):
            warnings.append("Pareto-статус не рассчитан; он показан как 'нет данных' и не влияет на score.")
        winner = ranking[0] if ranking else {}
        if str(winner.get("pareto_status")) == "доминируемая":
            warnings.append(
                "Гибридный лидер помечен Pareto как доминируемый; рекомендацию нужно проверить вручную."
            )
        if ga_norm_info.get("status") in {"constant_score", "no_finite_scores"}:
            warnings.append("GA-score не различает кандидатов в текущем пуле.")
        if ahp_norm_info.get("status") in {"constant_score", "no_finite_scores"}:
            warnings.append("AHP-score не различает кандидатов в текущем пуле.")
        if ranking:
            max_disagreement = max(int(item.get("rank_disagreement") or 0) for item in ranking)
            if max_disagreement >= 3:
                warnings.append("Есть заметное расхождение рангов GA и AHP; гибрид является компромиссной витриной.")
        return warnings

    def _summary(
        self,
        winner: Mapping[str, Any] | None,
        warnings: Sequence[str],
        lambda_value: float,
    ) -> str:
        if not winner:
            return "Гибридная оценка не сформировала победителя."
        score = winner.get("hybrid_score")
        score_text = f"{float(score):.4f}" if isinstance(score, (int, float)) else "—"
        text = (
            f"Гибридный лидер: {winner.get('id')} — {winner.get('name')} "
            f"со score={score_text}. Режим: {self._lambda_label(lambda_value)}. "
            f"Pareto: {winner.get('pareto_status', 'нет данных')}."
        )
        if warnings:
            text += " " + " ".join(str(item) for item in warnings[:2])
        return text

    def _comment(self, row: Mapping[str, Any]) -> str:
        pareto = str(row.get("pareto_status") or "нет данных")
        disagreement = int(row.get("rank_disagreement") or 0)
        if row.get("ga_rank") == 1 and row.get("ahp_rank") == 1 and pareto == "недоминируемая":
            return "подтверждён всеми блоками"
        if row.get("ga_rank") == 1 and row.get("ahp_rank") == 1:
            return "лидер GA и AHP"
        if pareto == "доминируемая":
            return "есть Pareto-предупреждение"
        if disagreement <= 1:
            return "методы близки"
        if disagreement <= 3:
            return "заметный сдвиг ранга"
        return "сильное расхождение"

    def _bounded_lambda(self, value: Any) -> float:
        numeric = self._finite_float_or_none(value)
        if numeric is None:
            return 0.5
        return min(1.0, max(0.0, numeric))

    def _lambda_label(self, value: float) -> str:
        if abs(value - 0.5) <= 1e-12:
            return "нейтральный компромисс"
        return "с приоритетом GA" if value > 0.5 else "с приоритетом AHP"

    def _positive_int(self, value: Any, *, fallback: int) -> int:
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            return fallback
        return numeric if numeric > 0 else fallback

    def _finite_float_or_none(self, value: Any) -> float | None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return numeric if numeric == numeric and numeric not in {float("inf"), float("-inf")} else None


__all__ = ["HybridDecisionAssessmentService"]
