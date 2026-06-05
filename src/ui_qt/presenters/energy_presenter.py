from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

from ui_qt.presenters.app_presenter import QtAppPresenter


@dataclass(frozen=True)
class EnergyProfileInput:
    """User-facing calculation parameters for the Qt energy screen."""

    hours_per_day: float = 8.0
    working_days: float = 22.0
    cost_per_kwh: float = 1.0

    def as_dict(self) -> dict[str, float]:
        return {
            "hours_per_day": float(self.hours_per_day),
            "working_days": float(self.working_days),
            "cost_per_kwh": float(self.cost_per_kwh),
        }


@dataclass(frozen=True)
class EnergySummary:
    """Compact summary values shown by the Qt screen."""

    total_cost: float
    total_kwh: float
    item_count: int
    total_power: float


class EnergyPresenter:
    """Qt boundary for electricity calculations.

    The presenter keeps widgets away from application/domain services.  It
    exposes plain dictionaries and numbers that can be tested without Qt.
    """

    DEFAULT_PROFILE = EnergyProfileInput()

    def __init__(self, app_presenter: QtAppPresenter | None = None) -> None:
        self.app_presenter = app_presenter or QtAppPresenter()
        self._profile = self.DEFAULT_PROFILE
        self._report: dict[str, Any] = {"total_cost": 0.0, "items": []}

    @property
    def profile(self) -> EnergyProfileInput:
        return self._profile

    def equipment_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for row in self.app_presenter.list_energy_rows():
            rows.append(self._normalize_equipment_row(row))
        return rows

    def table_rows(self) -> list[dict[str, Any]]:
        return [self._to_table_row(row) for row in self.equipment_rows()]

    def round_the_clock_names(self) -> set[str]:
        return set(self.app_presenter.list_round_the_clock_equipment_names())

    def calculate(self, profile: EnergyProfileInput | Mapping[str, Any]) -> dict[str, Any]:
        normalized_profile = self._normalize_profile(profile)
        equipment_rows = self.equipment_rows()
        report = self.app_presenter.calculate_electricity_costs(
            hours_per_day=normalized_profile.hours_per_day,
            working_days=normalized_profile.working_days,
            cost_per_kwh=normalized_profile.cost_per_kwh,
            equipment_rows=equipment_rows,
        )
        self._profile = normalized_profile
        self._report = deepcopy(report)
        return deepcopy(report)

    def summary(self) -> EnergySummary:
        items = list(self._report.get("items", []))
        if not items:
            rows = self.equipment_rows()
            return EnergySummary(
                total_cost=float(self._report.get("total_cost", 0.0) or 0.0),
                total_kwh=0.0,
                item_count=len(rows),
                total_power=self._total_power(rows),
            )
        return EnergySummary(
            total_cost=float(self._report.get("total_cost", 0.0) or 0.0),
            total_kwh=sum(self._number(row.get("energy_consumption")) for row in items),
            item_count=len(items),
            total_power=self._total_power(items),
        )

    def report(self) -> dict[str, Any]:
        return deepcopy(self._report)

    def refresh(self) -> EnergySummary:
        return self.summary()

    def has_equipment(self) -> bool:
        return bool(self.equipment_rows())

    def _normalize_profile(self, profile: EnergyProfileInput | Mapping[str, Any]) -> EnergyProfileInput:
        if isinstance(profile, EnergyProfileInput):
            return profile
        return EnergyProfileInput(
            hours_per_day=self._number(profile.get("hours_per_day"), default=8.0),
            working_days=self._number(profile.get("working_days"), default=22.0),
            cost_per_kwh=self._number(profile.get("cost_per_kwh"), default=1.0),
        )

    def _normalize_equipment_row(self, row: Mapping[str, Any]) -> dict[str, Any]:
        max_power = row.get("max_power", row.get("max_power_watts", 0.0))
        return {
            "name": str(row.get("name", "")),
            "category": str(row.get("category", "")),
            "quantity": self._number(row.get("quantity")),
            "max_power": self._number(max_power),
        }

    def _to_table_row(self, row: Mapping[str, Any]) -> dict[str, Any]:
        name = str(row.get("name", ""))
        return {
            "name": name,
            "category": self._category_label(row.get("category")),
            "quantity": self._number(row.get("quantity")),
            "max_power": self._number(row.get("max_power")),
            "mode": "24/7" if name in self.round_the_clock_names() else "рабочий",
        }

    @staticmethod
    def _total_power(rows: list[Mapping[str, Any]]) -> float:
        return sum(
            EnergyPresenter._number(row.get("quantity"))
            * EnergyPresenter._number(row.get("max_power"))
            for row in rows
        )

    @staticmethod
    def _number(value: Any, *, default: float = 0.0) -> float:
        try:
            if value in (None, ""):
                return float(default)
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _category_label(value: Any) -> str:
        mapping = {"server": "сервер", "client": "клиент", "network": "сеть"}
        return mapping.get(str(value), str(value or "—"))
