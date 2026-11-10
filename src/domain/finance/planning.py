"""Multi-year planning: TCO by years with price growth, amortization and a replacement plan.

Pure functions without Qt. Money is in rubles, growth and rates are fractions (0.06 = 6 % a year).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

# --- TCO by years ------------------------------------------------------------------------------


@dataclass(frozen=True)
class TcoYear:
    year: int
    opex: float
    electricity: float
    total: float
    discounted: float
    cumulative: float
    cumulative_discounted: float


@dataclass(frozen=True)
class TcoByYears:
    one_time: float
    years: tuple[TcoYear, ...]
    total: float
    total_discounted: float
    total_flat: float
    growth_extra: float

    @property
    def cumulative_line(self) -> list[float]:
        return [self.one_time] + [year.cumulative for year in self.years]

    @property
    def cumulative_discounted_line(self) -> list[float]:
        return [self.one_time] + [year.cumulative_discounted for year in self.years]


def build_tco_by_years(
    *,
    one_time: float,
    opex_annual: float,
    electricity_annual: float,
    horizon_years: int,
    opex_growth: float = 0.0,
    electricity_growth: float = 0.0,
    discount_rate: float = 0.0,
) -> TcoByYears:
    """OPEX_t = OPEX * (1 + g)^(t-1); one-time costs (CAPEX + разовые) are paid at t = 0.

    The discounted value divides each year by (1 + r)^t.
    """
    horizon = max(1, min(30, int(round(horizon_years or 1))))
    rate = max(0.0, float(discount_rate))
    opex_growth = max(-0.5, float(opex_growth))
    electricity_growth = max(-0.5, float(electricity_growth))
    one_time = float(one_time)
    cumulative = one_time
    cumulative_discounted = one_time
    years: list[TcoYear] = []
    for year in range(1, horizon + 1):
        opex = float(opex_annual) * (1 + opex_growth) ** (year - 1)
        electricity = float(electricity_annual) * (1 + electricity_growth) ** (year - 1)
        total = opex + electricity
        discounted = total / (1 + rate) ** year
        cumulative += total
        cumulative_discounted += discounted
        years.append(
            TcoYear(
                year=year,
                opex=round(opex, 2),
                electricity=round(electricity, 2),
                total=round(total, 2),
                discounted=round(discounted, 2),
                cumulative=round(cumulative, 2),
                cumulative_discounted=round(cumulative_discounted, 2),
            )
        )
    total_flat = one_time + (float(opex_annual) + float(electricity_annual)) * horizon
    return TcoByYears(
        one_time=round(one_time, 2),
        years=tuple(years),
        total=round(cumulative, 2),
        total_discounted=round(cumulative_discounted, 2),
        total_flat=round(total_flat, 2),
        growth_extra=round(cumulative - total_flat, 2),
    )


# --- Amortization ------------------------------------------------------------------------------


@dataclass(frozen=True)
class ServiceLives:
    """Service life in months by kind of equipment."""

    server_months: int = 48
    hardware_months: int = 36
    software_months: int = 12


@dataclass(frozen=True)
class AmortizationRow:
    key: str
    name: str
    kind: str  # "ТО" or "ПО"
    category: str
    quantity: float
    total: float
    months: int
    monthly: float


SERVER_LIKE = ("server", "network")
SOFTWARE_LIKE = ("licenses",)


def service_life_months(name: str, category: str, lives: ServiceLives) -> int:
    text = f"{name} {category}".lower()
    if category in SOFTWARE_LIKE or "лиценз" in text:
        return lives.software_months
    if category in SERVER_LIKE or "сервер" in text or "коммутатор" in text or "маршрутиз" in text:
        return lives.server_months
    return lives.hardware_months


def _number(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def build_amortization_rows(
    components: Iterable[Mapping[str, object]],
    lives: ServiceLives | None = None,
    *,
    capital_categories: Sequence[str] = ("server", "client", "network", "licenses"),
) -> list[AmortizationRow]:
    """Straight-line amortization for every capital component of the selected configuration."""
    lives = lives or ServiceLives()
    rows: list[AmortizationRow] = []
    for index, component in enumerate(components):
        category = str(component.get("source_category") or component.get("category") or "")
        if category not in capital_categories:
            continue
        quantity = max(_number(component.get("quantity"), 1.0), 0.0)
        price = max(_number(component.get("price"), 0.0), 0.0)
        total = round(price * quantity, 2)
        if total <= 0:
            continue
        name = str(component.get("name") or f"Позиция {index + 1}")
        months = service_life_months(name, category, lives)
        rows.append(
            AmortizationRow(
                key=f"{index}:{name}",
                name=name,
                kind="ПО" if category in SOFTWARE_LIKE else "ТО",
                category=category,
                quantity=quantity,
                total=total,
                months=months,
                monthly=round(total / months, 2) if months > 0 else total,
            )
        )
    return rows


@dataclass(frozen=True)
class AmortizationSummary:
    total: float
    monthly: float
    annual: float
    hardware_monthly: float
    software_monthly: float


def amortization_summary(rows: Sequence[AmortizationRow]) -> AmortizationSummary:
    monthly = sum(row.monthly for row in rows)
    return AmortizationSummary(
        total=round(sum(row.total for row in rows), 2),
        monthly=round(monthly, 2),
        annual=round(monthly * 12, 2),
        hardware_monthly=round(sum(row.monthly for row in rows if row.kind == "ТО"), 2),
        software_monthly=round(sum(row.monthly for row in rows if row.kind == "ПО"), 2),
    )


# --- Replacement plan --------------------------------------------------------------------------


@dataclass
class ReplacementPurchase:
    key: str
    name: str
    quantity: float
    cost: float


@dataclass
class ReplacementYear:
    year: int
    purchases: list[ReplacementPurchase] = field(default_factory=list)
    total: float = 0.0


@dataclass(frozen=True)
class ReplacementPlan:
    years: tuple[ReplacementYear, ...]
    total: float
    average_per_year: float
    peak_year: int | None


def build_replacement_plan(rows: Sequence[AmortizationRow], horizon_years: int, price_growth: float = 0.0) -> ReplacementPlan:
    """A line with a service life of m months is bought again at months m, 2m, 3m...

    A purchase at month t falls into year ceil(t / 12); cost in year y = total * (1 + g)^y.
    Year 0 (the first purchase) is not part of the plan.
    """
    horizon = max(1, int(horizon_years))
    years = [ReplacementYear(year=index + 1) for index in range(horizon)]
    for row in rows:
        if row.total <= 0 or row.months <= 0:
            continue
        month = row.months
        while month <= horizon * 12:
            year = math.ceil(month / 12)
            cost = round(row.total * (1 + price_growth) ** year, 2)
            bucket = years[year - 1]
            existing = next((item for item in bucket.purchases if item.key == row.key), None)
            if existing is not None:
                existing.quantity += row.quantity
                existing.cost = round(existing.cost + cost, 2)
            else:
                bucket.purchases.append(ReplacementPurchase(row.key, row.name, row.quantity, cost))
            bucket.total = round(bucket.total + cost, 2)
            month += row.months
    for bucket in years:
        bucket.purchases.sort(key=lambda item: item.cost, reverse=True)
    total = round(sum(bucket.total for bucket in years), 2)
    peak = max((bucket for bucket in years if bucket.total > 0), key=lambda bucket: bucket.total, default=None)
    return ReplacementPlan(
        years=tuple(years),
        total=total,
        average_per_year=round(total / horizon, 2),
        peak_year=peak.year if peak is not None else None,
    )
