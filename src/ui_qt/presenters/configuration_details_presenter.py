from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ui_qt.presenters.app_presenter import QtAppPresenter
from ui_qt.presenters.workspace_presenter import format_money


_CATEGORY_LABELS = {
    "computer": "Компьютеры",
    "computers": "Компьютеры",
    "licenses": "Лицензии",
    "network": "Сеть",
    "router": "Маршрутизаторы",
    "server": "Серверы",
    "servers": "Серверы",
    "switch": "Коммутаторы",
    "workstation": "Рабочие станции",
}


@dataclass(frozen=True)
class ConfigurationDetails:
    candidate_id: str
    name: str
    source: str
    total_cost: str
    component_count: int
    components: list[dict[str, Any]]


class ConfigurationDetailsPresenter:
    """Read-only view of one candidate from the shared decision pool."""

    def __init__(self, app_presenter: QtAppPresenter, *, scope: str) -> None:
        self.app_presenter = app_presenter
        self.scope = str(scope)

    def details(self, candidate_id: str) -> ConfigurationDetails | None:
        candidate = self._candidate(candidate_id)
        if candidate is None:
            return None
        components = candidate.get("components", [])
        if not isinstance(components, list):
            components = []
        rows = [self._component_row(item) for item in components if isinstance(item, Mapping)]
        totals = candidate.get("totals", {}) if isinstance(candidate.get("totals"), Mapping) else {}
        return ConfigurationDetails(
            candidate_id=str(candidate.get("id") or candidate_id),
            name=str(candidate.get("name") or candidate_id),
            source=self._source_label(candidate.get("source")),
            total_cost=format_money(totals.get("capital_cost", 0.0)),
            component_count=len(rows),
            components=rows,
        )

    def _candidate(self, candidate_id: str) -> Mapping[str, Any] | None:
        requested = str(candidate_id)
        snapshot = self.app_presenter.get_candidate_pool_snapshot(self.scope)
        return next(
            (
                candidate
                for candidate in snapshot.candidates
                if isinstance(candidate, Mapping) and str(candidate.get("id")) == requested
            ),
            None,
        )

    def _component_row(self, component: Mapping[str, Any]) -> dict[str, Any]:
        quantity = self._number(component.get("quantity"), default=1.0)
        total = component.get("total_cost")
        if total is None:
            total = component.get("capital_cost")
        if total is None:
            total = self._number(component.get("price")) * quantity
        category = str(component.get("source_category") or component.get("category") or "other")
        return {
            "category": _CATEGORY_LABELS.get(category.lower(), category),
            "name": str(component.get("name") or "Без названия"),
            "quantity": self._format_quantity(quantity),
            "cost": format_money(total),
        }

    @staticmethod
    def _source_label(value: Any) -> str:
        labels = {"ga": "ГА", "ahp": "AHP", "manual": "Вручную", "demo": "Демо"}
        source = str(getattr(value, "value", value) or "")
        return labels.get(source.lower(), source or "Не указан")

    @staticmethod
    def _number(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _format_quantity(value: float) -> str:
        return str(int(value)) if value.is_integer() else f"{value:g}"
