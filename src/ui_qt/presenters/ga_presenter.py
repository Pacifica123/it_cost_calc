from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ui_qt.presenters.app_presenter import QtAppPresenter
from ui_qt.presenters.workspace_presenter import WorkspacePresenter, format_money


@dataclass(frozen=True)
class GaInput:
    """Compact GA run parameters for one ПО/ТО workspace."""

    population: int = 24
    generations: int = 30
    mutation_rate: float = 0.04
    seed: int = 42
    budget: float = 600_000.0
    max_power: float | None = None
    min_units: float = 3.0
    max_units: float = 3.0


@dataclass(frozen=True)
class GaSummary:
    """Small summary shown on the GA step."""

    scope: str
    label: str
    row_count: int
    candidate_count: int
    pool_count: int
    status: str
    best_score: float
    best_cost: float
    best_items: int


class GaPresenter:
    """Presenter for the Qt GA screen.

    It keeps GA execution, scope filtering and candidate-pool updates outside
    QWidget classes.  The UI receives only plain dictionaries and small summary
    objects.
    """

    def __init__(
        self,
        app_presenter: QtAppPresenter | None = None,
        *,
        scope: str,
        workspace_presenter: WorkspacePresenter | None = None,
    ) -> None:
        self.app_presenter = app_presenter or QtAppPresenter()
        self.workspace_presenter = workspace_presenter or WorkspacePresenter(
            self.app_presenter,
            scope=scope,
        )

    @property
    def scope(self) -> str:
        return self.workspace_presenter.scope

    @property
    def label(self) -> str:
        return self.workspace_presenter.label

    def default_input(self) -> GaInput:
        summary = self.workspace_presenter.summary()
        budget = summary.capex_total + summary.opex_once + summary.opex_monthly * 12
        if budget <= 0:
            budget = 600_000.0
        return GaInput(budget=float(budget), min_units=3.0, max_units=3.0)

    def summary(self) -> GaSummary:
        workspace_summary = self.workspace_presenter.summary()
        result = self.app_presenter.get_ga_result(self.scope)
        pool = self.app_presenter.get_candidate_pool_snapshot(self.scope)
        totals = result.get("totals", {}) if isinstance(result.get("totals"), Mapping) else {}
        return GaSummary(
            scope=self.scope,
            label=self.label,
            row_count=workspace_summary.row_count,
            candidate_count=self._candidate_count(result),
            pool_count=len(pool.candidates),
            status=self._status(result, workspace_summary.row_count),
            best_score=self._number(
                (result.get("ga_result") or {}).get("best_agg")
                if isinstance(result.get("ga_result"), Mapping)
                else None,
            ),
            best_cost=self._number(totals.get("capital_cost")),
            best_items=int(self._number(totals.get("selected_count"))),
        )

    def can_run(self) -> bool:
        return self.workspace_presenter.has_data()

    def run(self, values: GaInput) -> dict[str, Any]:
        min_units = float(values.min_units) if values.min_units > 0 else None
        max_units = float(values.max_units) if values.max_units > 0 else None
        if min_units is not None and max_units is not None and max_units < min_units:
            max_units = min_units

        result = self.app_presenter.run_genetic_optimization(
            scope=self.scope,
            max_budget=float(values.budget) if values.budget > 0 else None,
            max_power=values.max_power,
            target_units=min_units,
            max_target_units=max_units,
            max_selected_items=int(max_units) if max_units is not None else None,
            ga_params={
                "pop_size": int(values.population),
                "generations": int(values.generations),
                "mutation_rate": float(values.mutation_rate),
                "seed": int(values.seed),
                "top_solutions_limit": 8,
                "quality_check_enabled": False,
                "max_random_attempts": 200,
            },
        )
        return result

    def user_error_message(self, error: Exception | Mapping[str, Any]) -> str:
        if isinstance(error, Mapping):
            raw = str(error.get("error") or error.get("reason") or "")
        else:
            raw = str(error)
        text = raw.lower()
        if "список item-ов пуст" in text or "items" in text and "empty" in text:
            return "Нет подходящих данных. Проверьте заполнение каталога."
        if "допустимого решения" in text or "feasible" in text:
            return "Нет подходящей конфигурации. Ослабьте бюджет или ограничения."
        if "runtime entities source" in text:
            return "Исходные данные недоступны. Перезапустите приложение."
        if "budget" in text or "бюджет" in text:
            return "Бюджет не позволяет собрать конфигурацию. Увеличьте его."
        return "Расчёт не выполнен. Проверьте данные и ограничения."

    def candidate_rows(self) -> list[dict[str, Any]]:
        result = self.app_presenter.get_ga_result(self.scope)
        rows: list[dict[str, Any]] = []
        solutions = result.get("candidate_solutions", [])
        if not isinstance(solutions, list):
            solutions = []
        for index, solution in enumerate(solutions, start=1):
            if not isinstance(solution, Mapping):
                continue
            totals = solution.get("totals", {}) if isinstance(solution.get("totals"), Mapping) else {}
            rank = int(solution.get("rank") or index)
            rows.append(
                {
                    "candidate_id": str(solution.get("id") or f"GA-{rank}"),
                    "rank": rank,
                    "name": str(solution.get("name") or f"GA-кандидат {rank}"),
                    "score": self._format_score(solution.get("score")),
                    "total_cost": format_money(totals.get("capital_cost", 0.0)),
                    "source": "GA",
                }
            )
        return rows

    def best_items_rows(self) -> list[dict[str, Any]]:
        result = self.app_presenter.get_ga_result(self.scope)
        selected = result.get("selected_items", [])
        if not isinstance(selected, list):
            return []
        rows: list[dict[str, Any]] = []
        for item in selected:
            if not isinstance(item, Mapping):
                continue
            rows.append(
                {
                    "category": str(item.get("source_category") or item.get("category") or "—"),
                    "name": str(item.get("name") or "—"),
                    "quantity": self._number(item.get("quantity"), default=1.0),
                    "cost": format_money(item.get("total_cost", item.get("capital_cost", 0.0))),
                }
            )
        return rows

    def pool_summary_text(self) -> str:
        return self.app_presenter.get_candidate_pool_snapshot(self.scope).summary()

    def _candidate_count(self, result: Mapping[str, Any]) -> int:
        parameters = result.get("parameters", {}) if isinstance(result.get("parameters"), Mapping) else {}
        return int(self._number(parameters.get("candidate_count")))

    def _status(self, result: Mapping[str, Any], row_count: int) -> str:
        if row_count <= 0:
            return "Нет данных"
        if not result:
            return "Не запускался"
        if result.get("status") == "ok":
            return "Готово"
        return "Ошибка"

    def _format_score(self, value: Any) -> str:
        return f"{self._number(value):.4f}"

    @staticmethod
    def _number(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)
