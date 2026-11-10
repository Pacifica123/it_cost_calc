"""Presenter for the «Лицензии» tab: renewal calendar and .ics export."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

from domain.finance.licenses import (
    LicenseCalendar,
    LicenseSource,
    build_license_calendar,
    collect_license_sources,
    licenses_to_ics,
    next_renewals,
    period_label,
)
from ui_qt.presenters.app_presenter import QtAppPresenter
from ui_qt.presenters.planning_presenter import PlanningInput, money
from ui_qt.presenters.session_extras import load_extra, save_extra

SESSION_KEY = "licenses_settings"


@dataclass(frozen=True)
class LicenseSettings:
    start: date
    remind_days: int = 14

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], *, today: date | None = None) -> "LicenseSettings":
        start = parse_date(str(payload.get("start", ""))) or (today or date.today())
        try:
            remind = max(0, min(90, int(float(payload.get("remind_days", 14)))))
        except (TypeError, ValueError):
            remind = 14
        return cls(start=start, remind_days=remind)

    def to_mapping(self) -> dict[str, Any]:
        return {"start": self.start.isoformat(), "remind_days": self.remind_days}


def parse_date(text: str) -> date | None:
    text = text.strip()
    for parser in (_iso, _ru):
        value = parser(text)
        if value is not None:
            return value
    return None


def _iso(text: str) -> date | None:
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _ru(text: str) -> date | None:
    parts = text.split(".")
    if len(parts) != 3:
        return None
    try:
        day, month, year = (int(part) for part in parts)
        return date(year, month, day)
    except ValueError:
        return None


def ru_date(value: date) -> str:
    return value.strftime("%d.%m.%Y")


@dataclass(frozen=True)
class LicenseView:
    sources: tuple[LicenseSource, ...]
    calendar: LicenseCalendar
    settings: LicenseSettings
    horizon_years: int


class LicensesPresenter:
    def __init__(self, app_presenter: QtAppPresenter) -> None:
        self.app_presenter = app_presenter

    def saved_settings(self) -> LicenseSettings:
        return LicenseSettings.from_mapping(load_extra(self.app_presenter, SESSION_KEY))

    def save_settings(self, settings: LicenseSettings) -> None:
        save_extra(self.app_presenter, SESSION_KEY, settings.to_mapping())

    def build(self, components: Sequence[Mapping[str, Any]], plan: PlanningInput, settings: LicenseSettings) -> LicenseView:
        sources = tuple(collect_license_sources(components, license_period_months=plan.software_months))
        horizon = max(1, int(plan.horizon_years))
        return LicenseView(sources, build_license_calendar(sources, settings.start, horizon), settings, horizon)

    @staticmethod
    def source_rows(view: LicenseView) -> list[dict[str, Any]]:
        upcoming = {source.key: dates for source, dates in next_renewals(view.calendar, view.sources, view.settings.start)}
        rows = []
        for source in view.sources:
            payment = source.amount * source.period_months if source.kind == "subscription" else source.amount
            dates = upcoming.get(source.key, [])
            rows.append(
                {
                    "name": source.name,
                    "kind": "Лицензия" if source.kind == "license" else "Подписка",
                    "period": period_label(source.period_months),
                    "amount": money(payment),
                    "next": ", ".join(ru_date(value) for value in dates) or "—",
                }
            )
        return rows

    @staticmethod
    def year_rows(view: LicenseView) -> list[dict[str, Any]]:
        return [
            {
                "year": str(year.year),
                "licenses": money(year.licenses),
                "subscriptions": money(year.subscriptions),
                "total": money(year.total),
            }
            for year in view.calendar.by_year
        ]

    @staticmethod
    def summary(view: LicenseView) -> str:
        if not view.sources:
            return (
                "Нет лицензий и подписок в выбранной конфигурации ПО: "
                "добавьте компоненты категорий «licenses» или «subscription_licenses» и выполните Hybrid."
            )
        cal = view.calendar
        return (
            f"За {view.horizon_years} г. с {ru_date(view.settings.start)}: продления лицензий {money(cal.renewals_total)}, "
            f"подписки {money(cal.subscriptions_total)}, всего {money(cal.total)}. "
            "Первая покупка лицензий уже учтена в CAPEX."
        )

    def ics_text(self, view: LicenseView) -> str:
        return licenses_to_ics(view.sources, view.settings.start, view.horizon_years, remind_days_before=view.settings.remind_days)

    def export_ics(self, view: LicenseView, path: Path | None = None) -> Path:
        target = path or self.app_presenter.generated_reports_dir() / "licenses.ics"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.ics_text(view).encode("utf-8"))
        self.app_presenter._record_export_paths("licenses_ics", {"ics": target})
        return target

    def report_section(self, view: LicenseView) -> dict[str, Any]:
        return {
            "paragraphs": [self.summary(view)],
            "tables": [
                {
                    "caption": "Таблица – Лицензии и подписки",
                    "header": ["Позиция", "Вид", "Период", "Платёж", "Ближайшие даты"],
                    "rows": [[row["name"], row["kind"], row["period"], row["amount"], row["next"]] for row in self.source_rows(view)],
                },
                {
                    "caption": "Таблица – Платежи за лицензии по годам",
                    "header": ["Год", "Продления", "Подписки", "Итого"],
                    "rows": [[row["year"], row["licenses"], row["subscriptions"], row["total"]] for row in self.year_rows(view)],
                },
            ],
        }
