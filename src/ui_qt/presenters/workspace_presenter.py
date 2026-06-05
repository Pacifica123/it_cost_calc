from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from application.services.analysis_scope_profile_service import AnalysisScopeProfileService
from shared.constants import ANALYSIS_SCOPE_SOFTWARE, ANALYSIS_SCOPE_TECHNICAL
from ui_qt.presenters.app_presenter import QtAppPresenter

_CATEGORY_LABELS = {
    "server": "Серверы",
    "client": "Клиенты",
    "network": "Сеть",
    "licenses": "Лицензии",
    "subscription_licenses": "Подписки",
    "server_rental": "Аренда",
    "migration": "Миграция",
    "testing": "Тесты",
    "backup": "Резервирование",
    "labor_costs": "Трудозатраты",
    "server_administration": "Администрирование",
}

_SCOPE_EXPORT_LABELS = {
    ANALYSIS_SCOPE_SOFTWARE: "ПО",
    ANALYSIS_SCOPE_TECHNICAL: "ТО",
}


@dataclass(frozen=True)
class WorkspaceCategory:
    """One CAPEX/OPEX category shown on the Qt data workspace."""

    entity_name: str
    label: str
    group: str
    rows: tuple[dict[str, Any], ...]
    count: int
    total: float


@dataclass(frozen=True)
class WorkspaceSummary:
    """Compact card data for a ПО/ТО data workspace."""

    scope: str
    label: str
    title: str
    capex_total: float
    opex_monthly: float
    opex_once: float
    row_count: int
    capex_count: int
    opex_count: int


class WorkspacePresenter:
    """Presenter for Qt ПО/ТО data screens.

    It keeps scope filtering outside widgets and exposes only plain dictionaries
    so the future GA/AHP/Pareto screens can reuse the same source of truth.
    """

    def __init__(
        self,
        app_presenter: QtAppPresenter | None = None,
        *,
        scope: str = ANALYSIS_SCOPE_TECHNICAL,
        profile_service: AnalysisScopeProfileService | None = None,
    ) -> None:
        self.app_presenter = app_presenter or QtAppPresenter()
        self.profile_service = profile_service or AnalysisScopeProfileService()
        self.profile = self.profile_service.get_profile(scope)

    @property
    def scope(self) -> str:
        return self.profile.scope

    @property
    def label(self) -> str:
        return _SCOPE_EXPORT_LABELS.get(self.profile.scope, self.profile.label)

    @property
    def title(self) -> str:
        return self.profile.title

    def capital_categories(self) -> tuple[WorkspaceCategory, ...]:
        return tuple(
            self._category(entity_name, group="CAPEX")
            for entity_name in self.profile.capital_categories
        )

    def operational_categories(self) -> tuple[WorkspaceCategory, ...]:
        return tuple(
            self._category(entity_name, group="OPEX")
            for entity_name in self.profile.operational_categories
        )

    def categories(self) -> tuple[WorkspaceCategory, ...]:
        return (*self.capital_categories(), *self.operational_categories())

    def summary(self) -> WorkspaceSummary:
        capex = self.capital_categories()
        opex = self.operational_categories()
        return WorkspaceSummary(
            scope=self.scope,
            label=self.label,
            title=self.title,
            capex_total=sum(item.total for item in capex),
            opex_monthly=sum(self._row_monthly(row) for item in opex for row in item.rows),
            opex_once=sum(self._row_once(row) for item in opex for row in item.rows),
            row_count=sum(item.count for item in (*capex, *opex)),
            capex_count=sum(item.count for item in capex),
            opex_count=sum(item.count for item in opex),
        )

    def capex_table_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for category in self.capital_categories():
            for row in category.rows:
                rows.append(
                    {
                        "category": category.label,
                        "name": str(row.get("name", "")),
                        "quantity": self._row_quantity(row),
                        "price": self._row_unit_cost(row),
                        "total": self._row_capex(row),
                    }
                )
        return rows

    def opex_table_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for category in self.operational_categories():
            for row in category.rows:
                rows.append(
                    {
                        "category": category.label,
                        "name": str(row.get("name", "")),
                        "monthly_cost": self._row_monthly(row),
                        "one_time_cost": self._row_once(row),
                    }
                )
        return rows

    def category_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "group": category.group,
                "category": category.label,
                "items": category.count,
                "total": category.total,
            }
            for category in self.categories()
        ]

    def has_data(self) -> bool:
        return self.summary().row_count > 0

    def _category(self, entity_name: str, *, group: str) -> WorkspaceCategory:
        rows = tuple(self.app_presenter.list_entities(entity_name))
        total = (
            sum(self._row_capex(row) for row in rows)
            if group == "CAPEX"
            else sum(self._row_monthly(row) + self._row_once(row) for row in rows)
        )
        return WorkspaceCategory(
            entity_name=entity_name,
            label=_CATEGORY_LABELS.get(entity_name, entity_name),
            group=group,
            rows=rows,
            count=len(rows),
            total=total,
        )

    def _row_capex(self, row: Mapping[str, Any]) -> float:
        explicit_total = self._number(row.get("total_cost"), default=None)
        if explicit_total is not None:
            return explicit_total
        return self._row_unit_cost(row) * self._row_quantity(row)

    def _row_unit_cost(self, row: Mapping[str, Any]) -> float:
        for key in ("price", "purchase_cost", "unit_cost"):
            value = self._number(row.get(key), default=None)
            if value is not None:
                return value
        return 0.0

    def _row_quantity(self, row: Mapping[str, Any]) -> float:
        return self._number(row.get("quantity"), default=1.0) or 1.0

    def _row_monthly(self, row: Mapping[str, Any]) -> float:
        return self._number(row.get("monthly_cost"), default=0.0) or 0.0

    def _row_once(self, row: Mapping[str, Any]) -> float:
        return self._number(row.get("one_time_cost"), default=0.0) or 0.0

    @staticmethod
    def _number(value: Any, *, default: float | None) -> float | None:
        try:
            if value is None or value == "":
                return default
            return float(str(value).replace(" ", "").replace(",", "."))
        except (TypeError, ValueError):
            return default


def format_money(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    return f"{number:,.2f} ₽".replace(",", " ")
