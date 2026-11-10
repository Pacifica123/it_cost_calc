"""License renewal calendar and iCalendar (.ics) export.

Capital licenses (category ``licenses``) are bought for a term and paid again when it ends;
subscriptions (``subscription_licenses``) are paid every period. The calendar lists the
payments over the horizon, sums renewals by year and exports reminders as an .ics file
that Outlook, Google Calendar and phone calendars import.
"""

from __future__ import annotations

import calendar
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Iterable, Mapping, Sequence

CAPITAL_LICENSE_CATEGORIES = ("licenses",)
SUBSCRIPTION_CATEGORIES = ("subscription_licenses",)


@dataclass(frozen=True)
class LicenseSource:
    key: str
    name: str
    amount: float  # one renewal of a license, or the monthly price of a subscription
    quantity: float
    kind: str  # "license" | "subscription"
    period_months: int


@dataclass(frozen=True)
class LicensePayment:
    key: str
    name: str
    when: date
    amount: float
    kind: str
    renewal: int  # 0 = first purchase


@dataclass(frozen=True)
class LicenseYear:
    year: int
    licenses: float
    subscriptions: float
    total: float


@dataclass(frozen=True)
class LicenseCalendar:
    payments: tuple[LicensePayment, ...]
    by_year: tuple[LicenseYear, ...]
    renewals_total: float
    subscriptions_total: float
    total: float


def add_months(start: date, months: int) -> date:
    """Adds months keeping the day; the 31st becomes the last day of a shorter month."""
    index = start.month - 1 + months
    year, month = start.year + index // 12, index % 12 + 1
    return date(year, month, min(start.day, calendar.monthrange(year, month)[1]))


def period_label(months: int) -> str:
    if months == 1:
        return "каждый месяц"
    if months % 12 == 0:
        years = months // 12
        return "раз в год" if years == 1 else f"раз в {years} {'года' if years < 5 else 'лет'}"
    return f"раз в {months} мес."


def _number(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def collect_license_sources(
    components: Iterable[Mapping[str, object]],
    *,
    license_period_months: int = 12,
    subscription_period_months: int = 1,
) -> list[LicenseSource]:
    sources: list[LicenseSource] = []
    for index, component in enumerate(components):
        category = str(component.get("source_category") or component.get("category") or "")
        name = str(component.get("name") or f"Позиция {index + 1}")
        quantity = max(_number(component.get("quantity"), 1.0), 0.0) or 1.0
        if category in CAPITAL_LICENSE_CATEGORIES:
            price = max(_number(component.get("price")), 0.0)
            if price <= 0:
                continue
            sources.append(
                LicenseSource(f"{index}:{name}", name, round(price * quantity, 2), quantity, "license", max(1, int(license_period_months)))
            )
        elif category in SUBSCRIPTION_CATEGORIES:
            monthly = max(_number(component.get("monthly_cost", component.get("price"))), 0.0)
            if monthly <= 0:
                continue
            per_unit = "monthly_cost" not in component
            amount = monthly * quantity if per_unit else monthly
            sources.append(
                LicenseSource(f"{index}:{name}", name, round(amount, 2), quantity, "subscription", max(1, int(subscription_period_months)))
            )
    return sources


def build_license_calendar(sources: Sequence[LicenseSource], start: date, horizon_years: int) -> LicenseCalendar:
    end = add_months(start, max(1, int(horizon_years)) * 12)
    payments: list[LicensePayment] = []
    for source in sources:
        per_payment = source.amount * source.period_months if source.kind == "subscription" else source.amount
        renewal = 0
        while True:
            when = add_months(start, renewal * source.period_months)
            if when >= end:
                break
            payments.append(LicensePayment(source.key, source.name, when, round(per_payment, 2), source.kind, renewal))
            renewal += 1
    payments.sort(key=lambda payment: (payment.when, payment.name))
    years: dict[int, dict[str, float]] = {}
    for payment in payments:
        entry = years.setdefault(payment.when.year, {"licenses": 0.0, "subscriptions": 0.0})
        # The first purchase of a capital license is already in CAPEX.
        if payment.kind == "license" and payment.renewal == 0:
            continue
        entry["licenses" if payment.kind == "license" else "subscriptions"] += payment.amount
    by_year = tuple(
        LicenseYear(year, round(values["licenses"], 2), round(values["subscriptions"], 2), round(values["licenses"] + values["subscriptions"], 2))
        for year, values in sorted(years.items())
    )
    renewals = sum(payment.amount for payment in payments if payment.kind == "license" and payment.renewal > 0)
    subscriptions = sum(payment.amount for payment in payments if payment.kind == "subscription")
    return LicenseCalendar(tuple(payments), by_year, round(renewals, 2), round(subscriptions, 2), round(renewals + subscriptions, 2))


def next_renewals(cal: LicenseCalendar, sources: Sequence[LicenseSource], after: date, limit: int = 3) -> list[tuple[LicenseSource, list[date]]]:
    result = []
    for source in sources:
        dates = [payment.when for payment in cal.payments if payment.key == source.key and payment.when > after][:limit]
        if dates:
            result.append((source, dates))
    result.sort(key=lambda item: item[1][0])
    return result


# --- iCalendar ---------------------------------------------------------------------------------


def _ics_date(value: date) -> str:
    return value.strftime("%Y%m%d")


def _ics_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\r\n", "\\n").replace("\n", "\\n")


def _fold(line: str) -> str:
    """Folds lines longer than 75 octets (RFC 5545); Cyrillic letters take 2 octets."""
    parts: list[str] = []
    current = ""
    octets = 0
    for char in line:
        size = len(char.encode("utf-8"))
        if octets + size > (74 if parts else 75):
            parts.append(current)
            current, octets = "", 0
        current += char
        octets += size
    parts.append(current)
    return "\r\n ".join(parts)


def _money(value: float) -> str:
    return f"{value:,.0f}".replace(",", " ") + " ₽"


def licenses_to_ics(
    sources: Sequence[LicenseSource],
    start: date,
    horizon_years: int,
    *,
    remind_days_before: int = 14,
    now: datetime | None = None,
) -> str:
    """One all-day event per source repeating with its period (RRULE) and a reminder ahead."""
    now = now or datetime.now(timezone.utc)
    stamp = now.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//IT Cost Calc//Licenses//RU",
        "CALSCALE:GREGORIAN",
        "X-WR-CALNAME:Продление лицензий",
    ]
    horizon_months = max(1, int(horizon_years)) * 12
    for index, source in enumerate(sources):
        offset = source.period_months if source.kind == "license" else 0
        first = add_months(start, offset)
        count = (horizon_months - offset - 1) // source.period_months + 1
        if count < 1:
            continue
        payment = source.amount * source.period_months if source.kind == "subscription" else source.amount
        summary = f"Продлить лицензию: {source.name}" if source.kind == "license" else f"Оплата подписки: {source.name}"
        description = f"Сумма: {_money(payment)}. Период: {period_label(source.period_months)}."
        if source.quantity > 1:
            description += f" Количество: {source.quantity:g}."
        lines.extend(
            [
                "BEGIN:VEVENT",
                f"UID:itcost-license-{index}-{_ics_date(start)}@itcost.local",
                f"DTSTAMP:{stamp}",
                f"DTSTART;VALUE=DATE:{_ics_date(first)}",
                f"DTEND;VALUE=DATE:{_ics_date(first + timedelta(days=1))}",
                f"RRULE:FREQ=MONTHLY;INTERVAL={source.period_months};COUNT={count}",
                f"SUMMARY:{_ics_text(summary)}",
                f"DESCRIPTION:{_ics_text(description)}",
                "BEGIN:VALARM",
                "ACTION:DISPLAY",
                f"DESCRIPTION:{_ics_text(summary)}",
                f"TRIGGER:-P{max(0, int(remind_days_before))}D",
                "END:VALARM",
                "END:VEVENT",
            ]
        )
    lines.append("END:VCALENDAR")
    return "\r\n".join(_fold(line) for line in lines) + "\r\n"
