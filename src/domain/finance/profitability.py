"""Profitability and risk measures that go with NPV.

IRR, profitability index, simple and discounted payback, Monte Carlo risk analysis
and a tornado (one-factor sensitivity) diagram. Pure functions without Qt, so the
same logic is used by the NPV screen, reports and tests.

Conventions: ``investment`` is the initial outlay at year 0 (positive number),
``cash_flows`` are the net flows of years 1..n, ``rate`` is a fraction (0.1 = 10 %).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

IRR_LOW = -0.99
IRR_HIGH = 10.0


def npv_at_rate(investment: float, cash_flows: Sequence[float], rate: float) -> float:
    """NPV = -I0 + sum(CFt / (1 + r)^t)."""
    total = -float(investment)
    for index, cash_flow in enumerate(cash_flows):
        total += float(cash_flow) / (1.0 + rate) ** (index + 1)
    return total


def calculate_irr(investment: float, cash_flows: Sequence[float]) -> float | None:
    """Rate at which NPV = 0 (bisection on -99 %..1000 %). None when NPV keeps one sign."""
    flows = [float(value) for value in cash_flows]
    if not flows or investment <= 0:
        return None
    low, high = IRR_LOW, IRR_HIGH
    npv_low = npv_at_rate(investment, flows, low)
    npv_high = npv_at_rate(investment, flows, high)
    if npv_low == 0:
        return low
    if math.copysign(1, npv_low) == math.copysign(1, npv_high):
        return None
    for _ in range(200):
        if high - low <= 1e-9:
            break
        middle = (low + high) / 2
        npv_middle = npv_at_rate(investment, flows, middle)
        if math.copysign(1, npv_middle) == math.copysign(1, npv_low):
            low, npv_low = middle, npv_middle
        else:
            high = middle
    return (low + high) / 2


def profitability_index(investment: float, cash_flows: Sequence[float], rate: float) -> float | None:
    """PI = PV of flows / I0. Above 1 means the project returns more than it costs."""
    if investment <= 0:
        return None
    return (npv_at_rate(investment, cash_flows, rate) + investment) / investment


def _payback(investment: float, flows: Sequence[float]) -> float | None:
    """Fractional year at which the cumulative value crosses zero, or None."""
    accumulated = -float(investment)
    if accumulated >= 0:
        return 0.0
    for index, value in enumerate(flows):
        previous = accumulated
        accumulated += value
        if accumulated >= 0:
            share = (-previous / value) if value > 0 else 1.0
            return round(index + min(max(share, 0.0), 1.0), 2)
    return None


def simple_payback_years(investment: float, cash_flows: Sequence[float]) -> float | None:
    return _payback(investment, [float(value) for value in cash_flows])


def discounted_payback_years(investment: float, cash_flows: Sequence[float], rate: float) -> float | None:
    discounted = [float(value) / (1.0 + rate) ** (index + 1) for index, value in enumerate(cash_flows)]
    return _payback(investment, discounted)


def npv_rate_curve(
    investment: float,
    cash_flows: Sequence[float],
    rate: float,
    irr: float | None,
    points: int = 9,
) -> list[tuple[float, float]]:
    """NPV at evenly spaced rates from 0 % to a bit past IRR: [(rate_percent, npv), ...]."""
    irr_percent = 0.0 if irr is None else irr * 100
    top = min(100.0, max(20.0, math.ceil(max(irr_percent, rate * 100) * 1.5 / 5) * 5))
    step = next((candidate for candidate in (1, 2, 2.5, 5, 10, 20) if top / candidate <= points - 1), 20)
    result: list[tuple[float, float]] = []
    value = 0.0
    while value <= top + 1e-9:
        result.append((value, round(npv_at_rate(investment, cash_flows, value / 100), 2)))
        value += step
    return result


@dataclass(frozen=True)
class ProfitabilityMetrics:
    npv: float
    irr: float | None
    profitability_index: float | None
    simple_payback: float | None
    discounted_payback: float | None
    horizon_years: int


def profitability_metrics(investment: float, cash_flows: Sequence[float], rate: float) -> ProfitabilityMetrics:
    flows = [float(value) for value in cash_flows]
    return ProfitabilityMetrics(
        npv=round(npv_at_rate(investment, flows, rate), 2),
        irr=calculate_irr(investment, flows),
        profitability_index=profitability_index(investment, flows, rate),
        simple_payback=simple_payback_years(investment, flows),
        discounted_payback=discounted_payback_years(investment, flows, rate),
        horizon_years=len(flows),
    )


# --- Monte Carlo -----------------------------------------------------------------------------


def _seeded_random(seed: int):
    """Linear congruential generator, same as in the mobile app: a fixed seed gives the same result."""
    state = int(seed) or 1

    def next_value() -> float:
        nonlocal state
        state = (state * 1664525 + 1013904223) % 0x100000000
        return state / 0x100000000

    return next_value


def triangular(low: float, mode: float, high: float, u: float) -> float:
    """Triangular distribution on [low, high] with the peak at ``mode``."""
    if high <= low:
        return mode
    split = (mode - low) / (high - low)
    if u < split:
        return low + math.sqrt(u * (high - low) * (mode - low))
    return high - math.sqrt((1 - u) * (high - low) * (high - mode))


def _around(value: float, spread: float, u: float) -> float:
    delta = abs(value) * spread
    return triangular(value - delta, value, value + delta, u)


def _percentile(sorted_values: Sequence[float], share: float) -> float:
    position = (len(sorted_values) - 1) * share
    lower = math.floor(position)
    upper = math.ceil(position)
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (position - lower)


@dataclass(frozen=True)
class HistogramBin:
    low: float
    high: float
    count: int


@dataclass(frozen=True)
class MonteCarloResult:
    runs: int
    mean: float
    std_dev: float
    p10: float
    p50: float
    p90: float
    minimum: float
    maximum: float
    loss_probability: float
    bins: tuple[HistogramBin, ...] = field(default_factory=tuple)


DEFAULT_MONTE_CARLO = {"investment_spread": 0.1, "cash_flow_spread": 0.2, "rate_spread": 0.02, "runs": 2000, "seed": 7}
HISTOGRAM_BINS = 12


def run_monte_carlo(
    investment: float,
    cash_flows: Sequence[float],
    rate: float,
    *,
    investment_spread: float = 0.1,
    cash_flow_spread: float = 0.2,
    rate_spread: float = 0.02,
    runs: int = 2000,
    seed: int = 7,
) -> MonteCarloResult:
    """Each run draws investment, every flow and the rate from a triangular distribution around the entered value."""
    runs = max(100, min(20000, int(round(runs))))
    random = _seeded_random(seed)
    flows = [float(value) for value in cash_flows]
    values: list[float] = []
    for _ in range(runs):
        drawn_investment = _around(float(investment), investment_spread, random())
        drawn_flows = [_around(value, cash_flow_spread, random()) for value in flows]
        drawn_rate = max(0.0, triangular(rate - rate_spread, rate, rate + rate_spread, random()))
        values.append(npv_at_rate(drawn_investment, drawn_flows, drawn_rate))

    ordered = sorted(values)
    mean = sum(values) / runs
    variance = sum((value - mean) ** 2 for value in values) / runs
    minimum, maximum = ordered[0], ordered[-1]
    width = (maximum - minimum) / HISTOGRAM_BINS or 1.0
    counts = [0] * HISTOGRAM_BINS
    for value in values:
        counts[min(HISTOGRAM_BINS - 1, int((value - minimum) // width))] += 1
    bins = tuple(
        HistogramBin(round(minimum + index * width, 2), round(minimum + (index + 1) * width, 2), counts[index])
        for index in range(HISTOGRAM_BINS)
    )
    return MonteCarloResult(
        runs=runs,
        mean=round(mean, 2),
        std_dev=round(math.sqrt(variance), 2),
        p10=round(_percentile(ordered, 0.1), 2),
        p50=round(_percentile(ordered, 0.5), 2),
        p90=round(_percentile(ordered, 0.9), 2),
        minimum=round(minimum, 2),
        maximum=round(maximum, 2),
        loss_probability=sum(1 for value in values if value < 0) / runs,
        bins=bins,
    )


# --- Tornado -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TornadoBar:
    key: str
    label: str
    low_case: str
    high_case: str
    low: float
    high: float
    range: float
    break_even: str


@dataclass(frozen=True)
class TornadoResult:
    base_npv: float
    bars: tuple[TornadoBar, ...]
    leader: str | None


def _pct(fraction: float) -> str:
    return f"{fraction * 100:.1f}".rstrip("0").rstrip(".").replace(".", ",") + " %"


def _points(fraction: float) -> str:
    return f"{fraction * 100:.1f}".rstrip("0").rstrip(".").replace(".", ",") + " п. п."


def years_word(count: float) -> str:
    whole = int(count)
    if whole != count:
        return "года"
    mod10, mod100 = whole % 10, whole % 100
    if mod10 == 1 and mod100 != 11:
        return "год"
    if 2 <= mod10 <= 4 and not 12 <= mod100 <= 14:
        return "года"
    return "лет"


def build_tornado(
    investment: float,
    cash_flows: Sequence[float],
    rate: float,
    *,
    share: float = 0.2,
    rate_points: float = 0.02,
) -> TornadoResult:
    """Moves every factor to its pessimistic and optimistic value while the others stay as entered."""
    flows = [float(value) for value in cash_flows]
    share = max(0.0, share)
    rate_points = max(0.0, rate_points)
    base = npv_at_rate(investment, flows, rate)
    pv_flows = npv_at_rate(0.0, flows, rate)
    raw: list[dict] = []

    # Investment: more expensive is pessimistic.
    if investment > 0:
        if base >= 0:
            text = f"NPV станет 0, если вложения вырастут на {_pct(base / investment)}"
        else:
            text = f"NPV станет 0, если вложения снизятся на {_pct(min(1.0, -base / investment))}"
    else:
        text = "Вложений нет"
    raw.append(
        dict(
            key="investment",
            label="Вложения",
            low_case=f"+{_pct(share)}",
            high_case=f"−{_pct(share)}",
            low=npv_at_rate(investment * (1 + share), flows, rate),
            high=npv_at_rate(investment * (1 - share), flows, rate),
            break_even=text,
        )
    )

    # Cash flows, all years together.
    if pv_flows <= 0:
        text = "Потоки не возвращают вложения ни при каком росте"
    elif base >= 0:
        text = f"NPV станет 0, если потоки упадут на {_pct(base / pv_flows)}"
    else:
        text = f"NPV станет 0, если потоки вырастут на {_pct(-base / pv_flows)}"
    raw.append(
        dict(
            key="cash_flows",
            label="Денежные потоки",
            low_case=f"−{_pct(share)}",
            high_case=f"+{_pct(share)}",
            low=npv_at_rate(investment, [value * (1 - share) for value in flows], rate),
            high=npv_at_rate(investment, [value * (1 + share) for value in flows], rate),
            break_even=text,
        )
    )

    # Discount rate.
    up = npv_at_rate(investment, flows, rate + rate_points)
    down = npv_at_rate(investment, flows, max(0.0, rate - rate_points))
    irr = calculate_irr(investment, flows)
    if irr is None:
        text = "IRR нет: NPV одного знака при любой ставке"
    elif irr >= rate:
        text = f"NPV станет 0 при ставке {_pct(irr)} (IRR), запас {_points(irr - rate)}"
    else:
        text = f"NPV станет 0 при ставке {_pct(irr)} (IRR), нужно снизить на {_points(rate - irr)}"
    up_is_worse = up <= down
    raw.append(
        dict(
            key="rate",
            label="Ставка дисконтирования",
            low_case=("+" if up_is_worse else "−") + _points(rate_points),
            high_case=("−" if up_is_worse else "+") + _points(rate_points),
            low=min(up, down),
            high=max(up, down),
            break_even=text,
        )
    )

    # Horizon: one year less, or one more year repeating the last flow.
    if flows:
        shorter = npv_at_rate(investment, flows[:-1], rate)
        longer = npv_at_rate(investment, flows + [flows[-1]], rate)
        payback = discounted_payback_years(investment, flows, rate)
        if payback is None:
            text = "За весь срок проект не окупается с учётом дисконтирования"
        elif payback >= len(flows) - 1e-9:
            text = "Окупается только в последний год: запаса по сроку нет"
        else:
            spare = round(len(flows) - payback, 2)
            text = (
                f"Окупается за {str(payback).replace('.', ',')} {years_word(payback)} с учётом дисконтирования, "
                f"запас по сроку {str(spare).replace('.', ',')} {years_word(spare)}"
            )
        shorter_is_worse = shorter <= longer
        raw.append(
            dict(
                key="horizon",
                label="Срок проекта",
                low_case="−1 год" if shorter_is_worse else "+1 год",
                high_case="+1 год" if shorter_is_worse else "−1 год",
                low=min(shorter, longer),
                high=max(shorter, longer),
                break_even=text,
            )
        )

        # Late start: every flow comes one year later.
        delayed = npv_at_rate(investment, [0.0] + flows[:-1], rate)
        delay_is_worse = delayed <= base
        raw.append(
            dict(
                key="delay",
                label="Задержка запуска",
                low_case="на 1 год" if delay_is_worse else "по плану",
                high_case="по плану" if delay_is_worse else "на 1 год",
                low=min(delayed, base),
                high=max(delayed, base),
                break_even="Задержку на год проект выдерживает" if delayed >= 0 else "Задержка на год уводит NPV в минус",
            )
        )

    bars = sorted(
        (
            TornadoBar(
                key=item["key"],
                label=item["label"],
                low_case=item["low_case"],
                high_case=item["high_case"],
                low=round(item["low"], 2),
                high=round(item["high"], 2),
                range=round(item["high"] - item["low"], 2),
                break_even=item["break_even"],
            )
            for item in raw
        ),
        key=lambda bar: bar.range,
        reverse=True,
    )
    leader = bars[0].key if bars and bars[0].range > 0 else None
    return TornadoResult(base_npv=round(base, 2), bars=tuple(bars), leader=leader)
