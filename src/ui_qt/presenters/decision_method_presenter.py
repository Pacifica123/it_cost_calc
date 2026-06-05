from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ui_qt.presenters.app_presenter import QtAppPresenter
from ui_qt.presenters.workspace_presenter import format_money


@dataclass(frozen=True)
class DecisionMethodSummary:
    """Compact state for standalone AHP/Pareto Qt steps."""

    scope: str
    status: str
    pool_count: int
    ranked_count: int
    winner_name: str
    winner_score: float | None


class AhpPresenter:
    """Presenter for standalone AHP over the current scoped candidate pool."""

    def __init__(self, app_presenter: QtAppPresenter | None = None, *, scope: str) -> None:
        self.app_presenter = app_presenter or QtAppPresenter()
        self.scope = str(scope)

    def summary(self) -> DecisionMethodSummary:
        pool = self.app_presenter.get_candidate_pool_snapshot(self.scope)
        report = self.app_presenter.get_ahp_result(self.scope)
        final = report.get("final", {}) if isinstance(report.get("final"), Mapping) else {}
        ranking = final.get("ranking", []) if isinstance(final.get("ranking"), list) else []
        winner = final.get("winner", {}) if isinstance(final.get("winner"), Mapping) else {}
        return DecisionMethodSummary(
            scope=self.scope,
            status=self._status(report, len(pool.candidates)),
            pool_count=len(pool.candidates),
            ranked_count=len(ranking),
            winner_name=str(winner.get("name") or "—"),
            winner_score=self._score(winner.get("score")),
        )

    def can_run(self) -> bool:
        return bool(self.app_presenter.get_candidate_pool_snapshot(self.scope).candidates)

    def run(self) -> dict[str, Any]:
        return self.app_presenter.run_ahp_for_candidate_pool(self.scope)

    def ranking_rows(self) -> list[dict[str, Any]]:
        report = self.app_presenter.get_ahp_result(self.scope)
        final = report.get("final", {}) if isinstance(report.get("final"), Mapping) else {}
        ranking = final.get("ranking", []) if isinstance(final.get("ranking"), list) else []
        rows: list[dict[str, Any]] = []
        for row in ranking:
            if not isinstance(row, Mapping):
                continue
            totals = row.get("totals", {}) if isinstance(row.get("totals"), Mapping) else {}
            rows.append(
                {
                    "rank": row.get("rank"),
                    "name": row.get("name"),
                    "score": self._format_score(row.get("score")),
                    "ga_rank": row.get("ga_rank", "—"),
                    "delta": self._rank_delta(row),
                    "total_cost": format_money(totals.get("capital_cost", 0.0)),
                }
            )
        return rows

    def criteria_rows(self) -> list[dict[str, Any]]:
        report = self.app_presenter.get_ahp_result(self.scope)
        criteria = report.get("criteria", {}) if isinstance(report.get("criteria"), Mapping) else {}
        names = list(criteria.get("names", []))
        labels = list(criteria.get("labels", []))
        weights = list(criteria.get("weights", []))
        directions = list(criteria.get("directions", []))
        rows: list[dict[str, Any]] = []
        for index, criterion_id in enumerate(names):
            rows.append(
                {
                    "criterion": labels[index] if index < len(labels) else criterion_id,
                    "weight": self._format_score(weights[index] if index < len(weights) else None),
                    "direction": directions[index] if index < len(directions) else "—",
                }
            )
        return rows

    def status_text(self) -> str:
        report = self.app_presenter.get_ahp_result(self.scope)
        if report.get("status") == "ok":
            summary = self.summary()
            return f"AHP готов: {summary.ranked_count} альтернатив."
        if report:
            return str(report.get("reason") or "AHP не рассчитан.")
        return "AHP ожидает запуск."

    def _status(self, report: Mapping[str, Any], pool_count: int) -> str:
        if pool_count <= 0:
            return "Нет пула"
        if not report:
            return "Не запускался"
        if report.get("status") == "ok":
            return "Готово"
        return "Пропущен"

    def _rank_delta(self, row: Mapping[str, Any]) -> str:
        try:
            delta = int(row.get("rank")) - int(row.get("ga_rank"))
        except (TypeError, ValueError):
            return "—"
        if delta == 0:
            return "0"
        return f"{delta:+d}"

    def _format_score(self, value: Any) -> str:
        score = self._score(value)
        return "—" if score is None else f"{score:.4f}"

    @staticmethod
    def _score(value: Any) -> float | None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return numeric if numeric == numeric and numeric not in {float("inf"), float("-inf")} else None


class ParetoPresenter:
    """Presenter for Pareto/criteria-importance over the current scoped pool."""

    def __init__(self, app_presenter: QtAppPresenter | None = None, *, scope: str) -> None:
        self.app_presenter = app_presenter or QtAppPresenter()
        self.scope = str(scope)

    def summary(self) -> DecisionMethodSummary:
        pool = self.app_presenter.get_candidate_pool_snapshot(self.scope)
        report = self.app_presenter.get_pareto_result(self.scope)
        ranking = report.get("ranking", []) if isinstance(report.get("ranking"), list) else []
        winner_name = str(report.get("winner_name") or "—")
        winner_id = str(report.get("winner_id") or "")
        weighted_sum = report.get("weighted_sum", {}) if isinstance(report.get("weighted_sum"), Mapping) else {}
        return DecisionMethodSummary(
            scope=self.scope,
            status=self._status(report, len(pool.candidates)),
            pool_count=len(pool.candidates),
            ranked_count=len(ranking),
            winner_name=winner_name,
            winner_score=self._score(weighted_sum.get(winner_id)),
        )

    def can_run(self) -> bool:
        return bool(self.app_presenter.get_candidate_pool_snapshot(self.scope).candidates)

    def run(self) -> dict[str, Any]:
        return self.app_presenter.run_pareto_for_candidate_pool(self.scope)

    def ranking_rows(self) -> list[dict[str, Any]]:
        report = self.app_presenter.get_pareto_result(self.scope)
        ranking = report.get("ranking", []) if isinstance(report.get("ranking"), list) else []
        rows: list[dict[str, Any]] = []
        for index, row in enumerate(ranking, start=1):
            if not isinstance(row, Mapping):
                continue
            rows.append(
                {
                    "rank": index,
                    "name": row.get("name", row.get("id", "—")),
                    "score": self._format_score(row.get("weighted_sum")),
                    "status": "недом." if row.get("nondominated") else "доминир.",
                }
            )
        return rows

    def nondominated_rows(self) -> list[dict[str, Any]]:
        report = self.app_presenter.get_pareto_result(self.scope)
        nondominated = report.get("final_nondominated", [])
        if not isinstance(nondominated, list):
            return []
        names_by_id = {
            str(row.get("id")): str(row.get("name") or row.get("id"))
            for row in report.get("ranking", [])
            if isinstance(row, Mapping)
        }
        return [
            {"id": str(candidate_id), "name": names_by_id.get(str(candidate_id), str(candidate_id))}
            for candidate_id in nondominated
        ]

    def status_text(self) -> str:
        report = self.app_presenter.get_pareto_result(self.scope)
        if report.get("status") == "ok":
            summary = self.summary()
            return f"Pareto готов: {summary.ranked_count} альтернатив."
        if report:
            return str(report.get("reason") or "Pareto не рассчитан.")
        return "Pareto ожидает AHP."

    def _status(self, report: Mapping[str, Any], pool_count: int) -> str:
        if pool_count <= 0:
            return "Нет пула"
        if not report:
            return "Не запускался"
        if report.get("status") == "ok":
            return "Готово"
        return "Пропущен"

    def _format_score(self, value: Any) -> str:
        score = self._score(value)
        return "—" if score is None else f"{score:.2f}"

    @staticmethod
    def _score(value: Any) -> float | None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return numeric if numeric == numeric and numeric not in {float("inf"), float("-inf")} else None
