"""Presenter for the «Финансирование» tab of the «Планы» screen."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping, Sequence

from domain.finance.financing import (
    DeviceCounts,
    LeaseBuyInput,
    LeaseBuyResult,
    LocalCloudInput,
    LocalCloudResult,
    StaffInput,
    StaffResult,
    compare_financing,
    compare_local_cloud,
    count_devices,
    estimate_it_staff,
)
from ui_qt.presenters.app_presenter import QtAppPresenter
from ui_qt.presenters.planning_presenter import PlanningInput, money
from ui_qt.presenters.session_extras import load_extra, save_extra

SESSION_KEY = "financing_settings"


@dataclass(frozen=True)
class FinancingSettings:
    """Inputs of the tab; shares and rates in percent as typed by the user."""

    term_years: int = 3
    residual_percent: float = 15
    maintenance_percent: float = 5
    lease_advance_percent: float = 20
    lease_markup_percent: float = 10
    rent_monthly_percent: float = 3.5
    tax_percent: float = 0
    units_per_specialist: float = 60
    salary_monthly: float = 90000
    contributions_percent: float = 30.2
    outsourcing_per_workstation: float = 1500
    outsourcing_per_server: float = 6000
    cloud_per_seat_monthly: float = 2500
    cloud_one_time: float = 0
    cloud_growth_percent: float = 5

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "FinancingSettings":
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


FINANCING_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("term_years", "Срок, лет", "Срок сравнения покупки, лизинга и аренды, 1–10 лет."),
    ("residual_percent", "Остаточная стоимость, %", "Сколько стоит техника в конце срока."),
    ("maintenance_percent", "Обслуживание, %/год", "Обслуживание при покупке и лизинге, доля цены в год."),
    ("lease_advance_percent", "Аванс лизинга, %", "Первый взнос по лизингу."),
    ("lease_markup_percent", "Удорожание лизинга, %/год", "Переплата лизинговой компании за год."),
    ("rent_monthly_percent", "Аренда, %/мес", "Месячная плата за аренду, доля цены."),
    ("tax_percent", "Налог на прибыль, %", "0 – не учитывать налоговый щит."),
    ("units_per_specialist", "Единиц на специалиста", "40 – высокие требования, 60 – обычный офис, 80 – простой офис."),
    ("salary_monthly", "Зарплата, ₽/мес", "Оклад одного ИТ-специалиста."),
    ("contributions_percent", "Взносы, %", "Страховые взносы с зарплаты."),
    ("outsourcing_per_workstation", "Аутсорс за ПК, ₽/мес", "Цена обслуживания рабочего места."),
    ("outsourcing_per_server", "Аутсорс за сервер, ₽/мес", "Цена обслуживания сервера."),
    ("cloud_per_seat_monthly", "Облако за место, ₽/мес", "Подписка облака (VDI/SaaS/IaaS) на одно рабочее место."),
    ("cloud_one_time", "Переезд в облако, ₽", "Разовые затраты на миграцию."),
    ("cloud_growth_percent", "Рост цены облака, %/год", "Ежегодная индексация подписки."),
)


@dataclass(frozen=True)
class FinancingView:
    equipment_cost: float
    lease_buy: LeaseBuyResult
    devices: DeviceCounts
    staff: StaffResult
    local_cloud: LocalCloudResult


class FinancingPresenter:
    def __init__(self, app_presenter: QtAppPresenter) -> None:
        self.app_presenter = app_presenter

    def saved_settings(self) -> FinancingSettings:
        return FinancingSettings.from_mapping(load_extra(self.app_presenter, SESSION_KEY))

    def save_settings(self, settings: FinancingSettings) -> None:
        save_extra(self.app_presenter, SESSION_KEY, asdict(settings))

    @staticmethod
    def equipment_cost(components: Sequence[Mapping[str, Any]], categories: Sequence[str] = ("server", "client", "network")) -> float:
        total = 0.0
        for component in components:
            category = str(component.get("source_category") or component.get("category") or "")
            if category not in categories:
                continue
            try:
                total += max(float(component.get("price") or 0), 0.0) * max(float(component.get("quantity") or 1), 0.0)
            except (TypeError, ValueError):
                continue
        return round(total, 2)

    def build(self, components: Sequence[Mapping[str, Any]], plan: PlanningInput, settings: FinancingSettings) -> FinancingView:
        rate = plan.discount_rate_percent / 100
        cost = self.equipment_cost(components)
        lease_buy = compare_financing(
            LeaseBuyInput(
                cost=cost,
                years=settings.term_years,
                rate=rate,
                residual_share=settings.residual_percent / 100,
                maintenance_share=settings.maintenance_percent / 100,
                lease_advance_share=settings.lease_advance_percent / 100,
                lease_markup_per_year=settings.lease_markup_percent / 100,
                rent_monthly_share=settings.rent_monthly_percent / 100,
                tax_rate=settings.tax_percent / 100,
            )
        )
        devices = count_devices(components)
        staff = estimate_it_staff(
            StaffInput(
                devices=devices,
                units_per_specialist=settings.units_per_specialist,
                salary_monthly=settings.salary_monthly,
                contributions_rate=settings.contributions_percent / 100,
                outsourcing_per_workstation=settings.outsourcing_per_workstation,
                outsourcing_per_server=settings.outsourcing_per_server,
            )
        )
        local_cloud = compare_local_cloud(
            LocalCloudInput(
                local_capex=plan.one_time,
                local_monthly=(plan.opex_annual + plan.electricity_annual) / 12,
                seats=max(1, int(devices.workstations)),
                cloud_per_seat_monthly=settings.cloud_per_seat_monthly,
                cloud_one_time=settings.cloud_one_time,
                years=plan.horizon_years,
                rate=rate,
                cloud_growth=settings.cloud_growth_percent / 100,
            )
        )
        return FinancingView(cost, lease_buy, devices, staff, local_cloud)

    # --- rows ----------------------------------------------------------------------------------

    @staticmethod
    def lease_rows(view: FinancingView) -> list[dict[str, Any]]:
        return [
            {
                "title": option.title + (" ✓" if option.key == view.lease_buy.best else ""),
                "upfront": money(option.upfront),
                "monthly": money(option.monthly),
                "total": money(option.total_paid),
                "pv": money(option.pv_cost),
            }
            for option in view.lease_buy.options
        ]

    @staticmethod
    def lease_text(view: FinancingView) -> str:
        if view.equipment_cost <= 0:
            return "Нет стоимости техники: выберите конфигурацию ТО (Hybrid)."
        best = next(option for option in view.lease_buy.options if option.key == view.lease_buy.best)
        share = f"{view.lease_buy.rent_break_even_share * 100:.2f}".replace(".", ",")
        return (
            f"Техника на {money(view.equipment_cost)}. Выгоднее всего: {best.title.lower()}, "
            f"экономия {money(view.lease_buy.advantage)} в приведённых ценах против следующего варианта. "
            f"Аренда сравняется с лучшим из покупки и лизинга при ставке {share} % цены в месяц."
        )

    @staticmethod
    def staff_rows(view: FinancingView) -> list[dict[str, Any]]:
        devices, staff = view.devices, view.staff
        return [
            {"name": "Рабочие места / серверы / сеть / принтеры",
             "value": f"{devices.workstations:g} / {devices.servers:g} / {devices.network_devices:g} / {devices.printers:g}"},
            {"name": "Единиц обслуживания", "value": f"{staff.units:g}"},
            {"name": "Потребность, ставок", "value": f"{staff.fte:g} → {staff.staff:g}".replace(".", ",")},
            {"name": "Свой штат, в месяц", "value": money(staff.staff_monthly)},
            {"name": "Аутсорсинг, в месяц", "value": money(staff.outsourcing_monthly)},
        ]

    @staticmethod
    def staff_text(view: FinancingView) -> str:
        staff = view.staff
        if staff.units <= 0:
            return "Нет устройств для обслуживания: выберите конфигурацию ТО (Hybrid)."
        cheaper = "свой штат" if staff.cheaper == "staff" else "аутсорсинг"
        return f"Дешевле {cheaper}: разница {money(staff.difference)} в месяц ({money(staff.difference * 12)} в год)."

    @staticmethod
    def local_cloud_rows(view: FinancingView) -> list[dict[str, Any]]:
        result = view.local_cloud
        return [
            {"year": str(year), "local": money(local), "cloud": money(cloud)}
            for year, (local, cloud) in enumerate(zip(result.local_curve, result.cloud_curve))
        ]

    @staticmethod
    def local_cloud_text(view: FinancingView) -> str:
        result = view.local_cloud
        cheaper = "локальная инфраструктура" if result.cheaper == "local" else "облако"
        tail = (
            f" Локальный вариант окупает вложения на {result.break_even_year}-й год."
            if result.break_even_year
            else ""
        )
        return (
            f"За горизонт дешевле {cheaper}: локально {money(result.local_pv)}, облако {money(result.cloud_pv)} "
            f"(приведённо; облако {money(result.cloud_monthly)} в месяц).{tail}"
        )

    def report_section(self, view: FinancingView) -> dict[str, Any]:
        return {
            "paragraphs": [self.lease_text(view), self.staff_text(view), self.local_cloud_text(view)],
            "tables": [
                {
                    "caption": "Таблица – Покупка, лизинг или аренда",
                    "header": ["Вариант", "Сразу", "В месяц", "Всего", "Приведённо"],
                    "rows": [[row["title"], row["upfront"], row["monthly"], row["total"], row["pv"]] for row in self.lease_rows(view)],
                },
                {
                    "caption": "Таблица – Свой ИТ-штат или аутсорсинг",
                    "header": ["Показатель", "Значение"],
                    "rows": [[row["name"], row["value"]] for row in self.staff_rows(view)],
                },
                {
                    "caption": "Таблица – Локально или облако (накопленно, приведённо)",
                    "header": ["Год", "Локально", "Облако"],
                    "rows": [[row["year"], row["local"], row["cloud"]] for row in self.local_cloud_rows(view)],
                },
            ],
        }
