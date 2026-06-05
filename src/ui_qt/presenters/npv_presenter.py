from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ui_qt.presenters.app_presenter import QtAppPresenter


@dataclass(frozen=True)
class NpvInput:
    investment: float
    discount_rate: float
    cash_flows: tuple[float, ...]
    annual_effect: float = 0.0
    horizon_years: int = 5


@dataclass(frozen=True)
class NpvSummary:
    npv: float
    investment: float
    rate: float
    years: int
    final_accumulated: float
    status: str


class NpvPresenter:
    """Qt presenter for NPV without QWidget or Tkinter dependencies."""

    def __init__(self, app_presenter: QtAppPresenter | None = None) -> None:
        self.app_presenter = app_presenter or QtAppPresenter()
        self._financial_basis: dict[str, Any] | None = None
        self._last_report: dict[str, Any] | None = None

    def default_input(self) -> NpvInput:
        return NpvInput(
            investment=0.0,
            discount_rate=0.1,
            cash_flows=tuple(0.0 for _ in range(5)),
            annual_effect=0.0,
            horizon_years=5,
        )

    def prepare_from_costs(
        self,
        *,
        horizon_years: int,
        annual_effect: float,
        discount_rate: float,
    ) -> NpvInput:
        basis = self.app_presenter.prepare_npv_basis_from_current_costs(
            horizon_years=horizon_years,
            annual_effect=annual_effect,
            discount_rate=discount_rate,
        )
        self._financial_basis = deepcopy(basis)
        return NpvInput(
            investment=float(basis.get("investment", 0.0) or 0.0),
            discount_rate=discount_rate,
            cash_flows=tuple(float(value) for value in basis.get("cash_flows", [])),
            annual_effect=annual_effect,
            horizon_years=int(basis.get("horizon_years", horizon_years) or horizon_years),
        )

    def calculate(self, values: NpvInput | Mapping[str, Any]) -> dict[str, Any]:
        npv_input = self._ensure_input(values)
        report = self.app_presenter.build_npv_report(
            investment=npv_input.investment,
            discount_rate=npv_input.discount_rate,
            cash_flows=list(npv_input.cash_flows),
            financial_basis=self._financial_basis,
        )
        self._last_report = deepcopy(report)
        return report

    def clear_basis(self) -> None:
        self._financial_basis = None

    def summary(self, report: Mapping[str, Any] | None = None) -> NpvSummary:
        payload = dict(report or self._last_report or {})
        rows = list(payload.get("rows", []))
        npv_value = float(payload.get("npv", 0.0) or 0.0)
        investment = self._row_number(rows[0], "investment") if rows else 0.0
        rate = self._basis_rate(payload)
        final_accumulated = (
            self._row_number(rows[-1], "accumulated_npv") if rows else npv_value
        )
        years = max(len(rows) - 1, 0)
        if not rows:
            status = "Нет расчёта"
        elif npv_value >= 0:
            status = "Окупается"
        else:
            status = "Не окупается"
        return NpvSummary(
            npv=npv_value,
            investment=investment,
            rate=rate,
            years=years,
            final_accumulated=final_accumulated,
            status=status,
        )

    def table_rows(self, report: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        payload = dict(report or self._last_report or {})
        rows = list(payload.get("rows", []))
        return [self._format_table_row(row) for row in rows]

    def basis_rows(self, report: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        payload = dict(report or self._last_report or {})
        basis = payload.get("financial_basis")
        if not isinstance(basis, Mapping):
            basis = self._financial_basis or {}
        sources = basis.get("cost_sources", {}) if isinstance(basis, Mapping) else {}
        if not isinstance(sources, Mapping):
            sources = {}
        return [
            {"name": "CAPEX", "value": float(sources.get("capex", 0.0) or 0.0)},
            {"name": "Разово", "value": float(sources.get("one_time_costs", 0.0) or 0.0)},
            {"name": "OPEX/мес", "value": float(sources.get("monthly_opex", 0.0) or 0.0)},
            {"name": "Энергия", "value": float(sources.get("electricity_monthly_cost", 0.0) or 0.0)},
        ]

    def warnings(self, report: Mapping[str, Any] | None = None) -> list[str]:
        payload = dict(report or self._last_report or {})
        warnings = payload.get("warnings", [])
        return [str(value) for value in warnings]

    def _ensure_input(self, values: NpvInput | Mapping[str, Any]) -> NpvInput:
        if isinstance(values, NpvInput):
            return values
        cash_flows = values.get("cash_flows", ())
        return NpvInput(
            investment=float(values.get("investment", 0.0) or 0.0),
            discount_rate=float(values.get("discount_rate", 0.0) or 0.0),
            cash_flows=tuple(float(value) for value in cash_flows),
            annual_effect=float(values.get("annual_effect", 0.0) or 0.0),
            horizon_years=int(values.get("horizon_years", 0) or 0),
        )

    def _format_table_row(self, row: Mapping[str, Any]) -> dict[str, Any]:
        discount_factor = row.get("discount_factor")
        return {
            "year": int(float(row.get("year", 0) or 0)),
            "cash_flow": float(row.get("cash_flow", 0.0) or 0.0),
            "discount_factor": "—" if discount_factor is None else f"{float(discount_factor):.4f}",
            "present_value": float(row.get("present_value", 0.0) or 0.0),
            "accumulated_npv": float(row.get("accumulated_npv", 0.0) or 0.0),
        }

    def _basis_rate(self, payload: Mapping[str, Any]) -> float:
        basis = payload.get("financial_basis")
        if isinstance(basis, Mapping) and "discount_rate" in basis:
            return float(basis.get("discount_rate", 0.0) or 0.0)
        if self._financial_basis and "discount_rate" in self._financial_basis:
            return float(self._financial_basis.get("discount_rate", 0.0) or 0.0)
        return 0.0

    @staticmethod
    def _row_number(rows: Mapping[str, Any] | Any, key: str) -> float:
        if not isinstance(rows, Mapping):
            return 0.0
        return float(rows.get(key, 0.0) or 0.0)


def parse_cash_flows(text: str, *, fallback_years: int = 5, fallback_value: float = 0.0) -> tuple[float, ...]:
    """Parse comma-separated cash flows for UI and tests."""

    parts = [part.strip() for part in text.replace(";", ",").split(",")]
    values = [float(part.replace(" ", "")) for part in parts if part]
    if values:
        return tuple(values)
    years = max(int(fallback_years), 1)
    return tuple(float(fallback_value) for _ in range(years))


def format_cash_flows(values: Sequence[float]) -> str:
    return ", ".join(f"{float(value):.2f}" for value in values)
