"""Pessimistic, base and optimistic scenarios of costs and NPV."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from domain.finance.planning import build_tco_by_years
from domain.finance.profitability import calculate_irr, npv_at_rate


@dataclass(frozen=True)
class ScenarioShift:
    """Changes against the base, fractions: capex +0.15 = CAPEX 15 % higher."""

    key: str
    title: str
    capex: float = 0.0
    opex: float = 0.0
    effect: float = 0.0


DEFAULT_SHIFTS = (
    ScenarioShift("pessimistic", "Пессимистичный", capex=0.15, opex=0.2, effect=-0.2),
    ScenarioShift("base", "Базовый"),
    ScenarioShift("optimistic", "Оптимистичный", capex=-0.1, opex=-0.1, effect=0.1),
)


@dataclass(frozen=True)
class ScenarioResult:
    key: str
    title: str
    one_time: float
    annual_running: float
    tco: float
    tco_discounted: float
    npv: float | None
    irr: float | None


def build_scenarios(
    *,
    one_time: float,
    opex_annual: float,
    electricity_annual: float,
    horizon_years: int,
    opex_growth: float,
    electricity_growth: float,
    discount_rate: float,
    npv_investment: float | None = None,
    npv_cash_flows: Sequence[float] = (),
    shifts: Sequence[ScenarioShift] = DEFAULT_SHIFTS,
) -> list[ScenarioResult]:
    results = []
    for shift in shifts:
        capex_factor = max(0.0, 1 + shift.capex)
        opex_factor = max(0.0, 1 + shift.opex)
        tco = build_tco_by_years(
            one_time=one_time * capex_factor,
            opex_annual=opex_annual * opex_factor,
            electricity_annual=electricity_annual * opex_factor,
            horizon_years=horizon_years,
            opex_growth=opex_growth,
            electricity_growth=electricity_growth,
            discount_rate=discount_rate,
        )
        npv = irr = None
        if npv_investment is not None and npv_cash_flows:
            investment = npv_investment * capex_factor
            flows = [float(value) * (1 + shift.effect) for value in npv_cash_flows]
            npv = round(npv_at_rate(investment, flows, discount_rate), 2)
            irr = calculate_irr(investment, flows)
        results.append(
            ScenarioResult(
                key=shift.key,
                title=shift.title,
                one_time=round(one_time * capex_factor, 2),
                annual_running=round((opex_annual + electricity_annual) * opex_factor, 2),
                tco=tco.total,
                tco_discounted=tco.total_discounted,
                npv=npv,
                irr=irr,
            )
        )
    return results
