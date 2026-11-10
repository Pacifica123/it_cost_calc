"""Presenter for the «Планы» screen: TCO by years, amortization and the replacement plan."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping

from domain.finance.planning import (
    AmortizationRow,
    ReplacementPlan,
    ServiceLives,
    TcoByYears,
    amortization_summary,
    build_amortization_rows,
    build_replacement_plan,
    build_tco_by_years,
)
from shared.constants import ANALYSIS_SCOPE_SOFTWARE, ANALYSIS_SCOPE_TECHNICAL
from ui_qt.presenters.app_presenter import QtAppPresenter
from ui_qt.presenters.session_extras import load_extra, save_extra

SESSION_KEY = "plans_settings"


def money(value: float) -> str:
    return f"{value:,.0f}".replace(",", " ") + " ₽"


@dataclass(frozen=True)
class PlanningInput:
    """Everything the planning tabs need. Money per year at today's prices, growth and rate in percent."""

    one_time: float = 0.0
    opex_annual: float = 0.0
    electricity_annual: float = 0.0
    horizon_years: int = 5
    opex_growth_percent: float = 5.0
    electricity_growth_percent: float = 7.0
    discount_rate_percent: float = 10.0
    price_growth_percent: float = 5.0
    server_months: int = 48
    hardware_months: int = 36
    software_months: int = 12

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PlanningInput":
        values: dict[str, Any] = {}
        for item in fields(cls):
            if item.name not in payload:
                continue
            try:
                raw = float(payload[item.name])
            except (TypeError, ValueError):
                continue
            values[item.name] = int(raw) if item.type in ("int", int) else raw
        return cls(**values)

    def lives(self) -> ServiceLives:
        return ServiceLives(
            server_months=max(1, int(self.server_months)),
            hardware_months=max(1, int(self.hardware_months)),
            software_months=max(1, int(self.software_months)),
        )


class PlanningPresenter:
    def __init__(self, app_presenter: QtAppPresenter | None = None) -> None:
        self.app_presenter = app_presenter or QtAppPresenter()
        self.components: list[dict[str, Any]] = []
        self.source_status = "missing"

    # --- data ----------------------------------------------------------------------------------

    def saved_input(self) -> PlanningInput:
        return PlanningInput.from_mapping(load_extra(self.app_presenter, SESSION_KEY))

    def save_input(self, values: PlanningInput) -> None:
        save_extra(self.app_presenter, SESSION_KEY, asdict(values))

    def load_selection(self, values: PlanningInput) -> PlanningInput:
        """Take CAPEX, one-time costs, OPEX and electricity of the Hybrid winners (ТО and ПО)."""
        totals = self.app_presenter.prepare_cost_summary()
        source = totals.get("financial_source", {}) if isinstance(totals.get("financial_source"), Mapping) else {}
        self.source_status = str(source.get("status") or "missing")
        self.components = self.selected_components()
        one_time = float(totals.get("total_capital", 0.0) or 0.0) + float(totals.get("total_operational_one_time", 0.0) or 0.0)
        opex_annual = float(totals.get("total_operational_monthly", 0.0) or 0.0) * 12
        electricity_annual = float(totals.get("electricity_costs", 0.0) or 0.0) * 12
        return PlanningInput(**{**asdict(values), "one_time": round(one_time, 2), "opex_annual": round(opex_annual, 2), "electricity_annual": round(electricity_annual, 2)})

    def selected_components(self) -> list[dict[str, Any]]:
        components: list[dict[str, Any]] = []
        for scope in (ANALYSIS_SCOPE_TECHNICAL, ANALYSIS_SCOPE_SOFTWARE):
            candidate = self.app_presenter.selected_hybrid_candidate(scope)
            if not candidate:
                continue
            items = candidate.get("components", [])
            if isinstance(items, list):
                components.extend(dict(item) for item in items if isinstance(item, Mapping))
        return components

    # --- calculations --------------------------------------------------------------------------

    @staticmethod
    def tco(values: PlanningInput) -> TcoByYears:
        return build_tco_by_years(
            one_time=values.one_time,
            opex_annual=values.opex_annual,
            electricity_annual=values.electricity_annual,
            horizon_years=values.horizon_years,
            opex_growth=values.opex_growth_percent / 100,
            electricity_growth=values.electricity_growth_percent / 100,
            discount_rate=values.discount_rate_percent / 100,
        )

    def amortization(self, values: PlanningInput) -> list[AmortizationRow]:
        return build_amortization_rows(self.components, values.lives())

    def replacement(self, values: PlanningInput) -> ReplacementPlan:
        return build_replacement_plan(self.amortization(values), values.horizon_years, values.price_growth_percent / 100)

    # --- rows for tables -----------------------------------------------------------------------

    @staticmethod
    def tco_rows(result: TcoByYears) -> list[dict[str, Any]]:
        rows = [
            {
                "year": "0 (старт)",
                "opex": "—",
                "electricity": "—",
                "total": money(result.one_time),
                "discounted": money(result.one_time),
                "cumulative": money(result.one_time),
            }
        ]
        for year in result.years:
            rows.append(
                {
                    "year": str(year.year),
                    "opex": money(year.opex),
                    "electricity": money(year.electricity),
                    "total": money(year.total),
                    "discounted": money(year.discounted),
                    "cumulative": money(year.cumulative),
                }
            )
        return rows

    @staticmethod
    def tco_summary(result: TcoByYears) -> str:
        extra = result.growth_extra
        return (
            f"TCO за {len(result.years)} г.: {money(result.total)} "
            f"(дисконтированно {money(result.total_discounted)}). "
            f"Рост цен добавляет {money(extra)} к расчёту без роста ({money(result.total_flat)})."
        )

    @staticmethod
    def amortization_rows(rows: list[AmortizationRow]) -> list[dict[str, Any]]:
        return [
            {
                "name": row.name,
                "kind": row.kind,
                "quantity": f"{row.quantity:g}",
                "total": money(row.total),
                "months": str(row.months),
                "monthly": money(row.monthly),
            }
            for row in rows
        ]

    @staticmethod
    def amortization_text(rows: list[AmortizationRow]) -> str:
        if not rows:
            return "Нет капитальных позиций: выберите конфигурацию в ТО/ПО (Hybrid) и нажмите «Из выбора»."
        summary = amortization_summary(rows)
        return (
            f"Стоимость {money(summary.total)}, амортизация {money(summary.monthly)} в месяц "
            f"({money(summary.annual)} в год): ТО {money(summary.hardware_monthly)}, ПО {money(summary.software_monthly)}."
        )

    @staticmethod
    def replacement_rows(plan: ReplacementPlan) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for year in plan.years:
            if not year.purchases:
                rows.append({"year": str(year.year), "items": "—", "total": money(0)})
                continue
            items = ", ".join(f"{item.name} × {item.quantity:g}" for item in year.purchases)
            rows.append({"year": str(year.year), "items": items, "total": money(year.total)})
        return rows

    @staticmethod
    def replacement_text(plan: ReplacementPlan) -> str:
        if plan.peak_year is None:
            return "За горизонт плана ничего не требует замены."
        peak = plan.years[plan.peak_year - 1]
        return (
            f"Бюджет замены {money(plan.total)}, в среднем {money(plan.average_per_year)} в год. "
            f"Пик: {plan.peak_year}-й год, {money(peak.total)}."
        )

    def report_section(self, values: PlanningInput) -> dict[str, Any]:
        """Plain dict for exporters."""
        tco = self.tco(values)
        amortization = self.amortization(values)
        plan = self.replacement(values)
        return {
            "inputs": asdict(values),
            "tco_summary": self.tco_summary(tco),
            "tco_rows": self.tco_rows(tco),
            "amortization_summary": self.amortization_text(amortization),
            "amortization_rows": self.amortization_rows(amortization),
            "replacement_summary": self.replacement_text(plan),
            "replacement_rows": self.replacement_rows(plan),
        }
