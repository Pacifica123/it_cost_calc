"""How to pay for and run the chosen configuration.

* Покупка / лизинг / аренда техники: present value of all costs over the same term.
* Свой ИТ-штат или аутсорсинг: staff need by service units and the monthly cost of both.
* Локальная инфраструктура или облако: cumulative cost by years and the break-even year.

Pure functions without Qt; money in rubles, rates and shares as fractions.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, Mapping

# --- Buy / lease / rent ------------------------------------------------------------------------


@dataclass(frozen=True)
class LeaseBuyInput:
    cost: float
    years: int = 3
    rate: float = 0.1
    residual_share: float = 0.15
    maintenance_share: float = 0.05
    lease_advance_share: float = 0.2
    lease_markup_per_year: float = 0.1
    rent_monthly_share: float = 0.035
    tax_rate: float = 0.0


@dataclass(frozen=True)
class FinancingOption:
    key: str
    title: str
    upfront: float
    monthly: float
    total_paid: float
    tax_saving: float
    pv_cost: float
    curve: tuple[float, ...]


@dataclass(frozen=True)
class LeaseBuyResult:
    options: tuple[FinancingOption, ...]
    best: str
    advantage: float
    rent_break_even_share: float


def _evaluate(key: str, title: str, flows: list[tuple[int, float]], months: int, monthly_rate: float,
              upfront: float, monthly: float, tax_saving: float) -> FinancingOption:
    def discount(month: int) -> float:
        return 1 / (1 + monthly_rate) ** month

    pv = sum(amount * discount(month) for month, amount in flows)
    total_paid = sum(amount for _month, amount in flows)
    years = math.ceil(months / 12)
    curve = tuple(
        round(sum(amount * discount(month) for month, amount in flows if month <= year * 12))
        for year in range(years + 1)
    )
    return FinancingOption(
        key=key,
        title=title,
        upfront=round(upfront),
        monthly=round(monthly),
        total_paid=round(total_paid + tax_saving),
        tax_saving=round(tax_saving),
        pv_cost=round(pv),
        curve=curve,
    )


def compare_financing(data: LeaseBuyInput) -> LeaseBuyResult:
    """Cash flows are monthly and discounted at r_m = (1 + r)^(1/12) - 1.

    Покупка: -C at once, maintenance m*C/12 a month, residual +R*C at the end.
    Лизинг: advance a*C, the rest of C*(1 + u*T) in equal payments, maintenance, residual at the end.
    Аренда: k*C a month, maintenance included, nothing left at the end.
    With a profit tax rate, amortization, lease/rent payments and maintenance reduce the tax.
    """
    cost = max(0.0, float(data.cost))
    years = max(1, min(10, int(round(data.years)) or 1))
    months = years * 12
    monthly_rate = (1 + max(0.0, data.rate)) ** (1 / 12) - 1
    tax = max(0.0, min(0.5, data.tax_rate))
    maintenance_monthly = cost * max(0.0, data.maintenance_share) / 12
    residual = cost * max(0.0, min(1.0, data.residual_share))
    amort_monthly = cost / months

    buy: list[tuple[int, float]] = [(0, cost)]
    lease_advance = cost * max(0.0, min(1.0, data.lease_advance_share))
    lease_total = cost * (1 + max(0.0, data.lease_markup_per_year) * years)
    lease_monthly = max(0.0, lease_total - lease_advance) / months
    lease: list[tuple[int, float]] = [(0, lease_advance)]
    rent_monthly = cost * max(0.0, data.rent_monthly_share)
    rent: list[tuple[int, float]] = []
    buy_tax = lease_tax = rent_tax = 0.0
    for month in range(1, months + 1):
        buy.append((month, maintenance_monthly))
        lease.append((month, lease_monthly + maintenance_monthly))
        rent.append((month, rent_monthly))
        if tax > 0:
            buy_shield = tax * (amort_monthly + maintenance_monthly)
            lease_shield = tax * (lease_monthly + lease_advance / months + maintenance_monthly)
            rent_shield = tax * rent_monthly
            buy.append((month, -buy_shield))
            lease.append((month, -lease_shield))
            rent.append((month, -rent_shield))
            buy_tax += buy_shield
            lease_tax += lease_shield
            rent_tax += rent_shield
    buy.append((months, -residual))
    lease.append((months, -residual))

    options = (
        _evaluate("buy", "Покупка", buy, months, monthly_rate, cost, maintenance_monthly, buy_tax),
        _evaluate("lease", "Лизинг", lease, months, monthly_rate, lease_advance, lease_monthly, lease_tax),
        _evaluate("rent", "Аренда", rent, months, monthly_rate, 0.0, rent_monthly, rent_tax),
    )
    ordered = sorted(options, key=lambda option: option.pv_cost)
    target = min(options[0].pv_cost, options[1].pv_cost)
    annuity = sum(1 / (1 + monthly_rate) ** (index + 1) for index in range(months))
    rent_break_even = target / (cost * annuity * (1 - tax)) if cost > 0 else 0.0
    return LeaseBuyResult(
        options=options,
        best=ordered[0].key,
        advantage=ordered[1].pv_cost - ordered[0].pv_cost,
        rent_break_even_share=rent_break_even,
    )


# --- Own staff or outsourcing ------------------------------------------------------------------

UNIT_WEIGHTS = {"workstations": 1.0, "servers": 5.0, "network_devices": 0.5, "printers": 0.5}


@dataclass(frozen=True)
class DeviceCounts:
    workstations: float = 0
    servers: float = 0
    network_devices: float = 0
    printers: float = 0


@dataclass(frozen=True)
class StaffInput:
    devices: DeviceCounts
    units_per_specialist: float = 60
    salary_monthly: float = 90000
    contributions_rate: float = 0.302
    outsourcing_per_workstation: float = 1500
    outsourcing_per_server: float = 6000


@dataclass(frozen=True)
class StaffResult:
    units: float
    fte: float
    staff: float
    staff_monthly: float
    outsourcing_monthly: float
    cheaper: str
    difference: float


def _non_negative(value: object) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, number) if math.isfinite(number) else 0.0


def estimate_it_staff(data: StaffInput) -> StaffResult:
    """U = ПК*1 + серверы*5 + сеть*0,5 + принтеры*0,5; штат = U / нагрузка, вверх до 0,5 ставки."""
    devices = data.devices
    units = (
        _non_negative(devices.workstations) * UNIT_WEIGHTS["workstations"]
        + _non_negative(devices.servers) * UNIT_WEIGHTS["servers"]
        + _non_negative(devices.network_devices) * UNIT_WEIGHTS["network_devices"]
        + _non_negative(devices.printers) * UNIT_WEIGHTS["printers"]
    )
    per_specialist = max(1.0, _non_negative(data.units_per_specialist))
    fte = units / per_specialist
    staff = max(0.5, math.ceil(fte * 2) / 2) if units > 0 else 0.0
    staff_monthly = round(staff * _non_negative(data.salary_monthly) * (1 + _non_negative(data.contributions_rate)))
    outsourcing_monthly = round(
        _non_negative(devices.workstations) * _non_negative(data.outsourcing_per_workstation)
        + _non_negative(devices.servers) * _non_negative(data.outsourcing_per_server)
    )
    return StaffResult(
        units=round(units, 1),
        fte=round(fte, 2),
        staff=staff,
        staff_monthly=staff_monthly,
        outsourcing_monthly=outsourcing_monthly,
        cheaper="staff" if staff_monthly <= outsourcing_monthly else "outsourcing",
        difference=abs(staff_monthly - outsourcing_monthly),
    )


_PRINTER = re.compile(r"принтер|мфу|сканер|плоттер|printer")
_NETWORK = re.compile(r"коммутатор|маршрутиз|роутер|switch|router|точка доступа|wi-?fi|межсетев")
_SERVER = re.compile(r"сервер|server|схд|nas")


def count_devices(components: Iterable[Mapping[str, object]]) -> DeviceCounts:
    """Device counts of the selected configuration by category, refined by the name."""
    counts = {"workstations": 0.0, "servers": 0.0, "network_devices": 0.0, "printers": 0.0}
    for component in components:
        category = str(component.get("source_category") or component.get("category") or "")
        if category not in ("server", "client", "network"):
            continue
        quantity = _non_negative(component.get("quantity", 1))
        name = str(component.get("name") or "").lower()
        if _PRINTER.search(name):
            counts["printers"] += quantity
        elif category == "network" or _NETWORK.search(name):
            counts["network_devices"] += quantity
        elif category == "server" or _SERVER.search(name):
            counts["servers"] += quantity
        else:
            counts["workstations"] += quantity
    return DeviceCounts(**counts)


# --- Local or cloud ----------------------------------------------------------------------------


@dataclass(frozen=True)
class LocalCloudInput:
    local_capex: float
    local_monthly: float
    seats: int
    cloud_per_seat_monthly: float = 2500
    cloud_one_time: float = 0
    years: int = 5
    rate: float = 0.1
    cloud_growth: float = 0.0


@dataclass(frozen=True)
class LocalCloudResult:
    local_curve: tuple[float, ...]
    cloud_curve: tuple[float, ...]
    local_pv: float
    cloud_pv: float
    cheaper: str
    break_even_year: int | None
    cloud_monthly: float


def compare_local_cloud(data: LocalCloudInput) -> LocalCloudResult:
    """Cumulative discounted cost by years: local = CAPEX + running costs, cloud = subscription per seat."""
    years = max(1, min(15, int(data.years)))
    rate = max(0.0, data.rate)
    cloud_monthly = max(0, int(data.seats)) * max(0.0, data.cloud_per_seat_monthly)
    local = [max(0.0, data.local_capex)]
    cloud = [max(0.0, data.cloud_one_time)]
    for year in range(1, years + 1):
        factor = 1 / (1 + rate) ** year
        local.append(local[-1] + max(0.0, data.local_monthly) * 12 * factor)
        cloud.append(cloud[-1] + cloud_monthly * 12 * (1 + data.cloud_growth) ** (year - 1) * factor)
    break_even = None
    if local[0] > cloud[0]:
        break_even = next((year for year in range(1, years + 1) if local[year] <= cloud[year]), None)
    return LocalCloudResult(
        local_curve=tuple(round(value) for value in local),
        cloud_curve=tuple(round(value) for value in cloud),
        local_pv=round(local[-1]),
        cloud_pv=round(cloud[-1]),
        cheaper="local" if local[-1] <= cloud[-1] else "cloud",
        break_even_year=break_even,
        cloud_monthly=round(cloud_monthly),
    )
