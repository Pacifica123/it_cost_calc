"""Presenter for the «Доходность и риск» part of the NPV screen (no QWidget dependencies)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from domain.finance.profitability import (
    MonteCarloResult,
    ProfitabilityMetrics,
    TornadoResult,
    build_tornado,
    profitability_metrics,
    run_monte_carlo,
    years_word,
)


@dataclass(frozen=True)
class RiskSettings:
    """Spreads for Monte Carlo and the tornado. Shares are fractions, rate spread is in fractions too (0.02 = 2 п. п.)."""

    investment_spread: float = 0.1
    cash_flow_spread: float = 0.2
    rate_spread: float = 0.02
    runs: int = 2000
    seed: int = 7


@dataclass(frozen=True)
class RiskView:
    metrics: ProfitabilityMetrics
    monte_carlo: MonteCarloResult | None
    tornado: TornadoResult | None


def money(value: float) -> str:
    return f"{value:,.2f}".replace(",", " ").replace(".", ",") + " ₽"


def percent(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.2f}".replace(".", ",") + " %"


def years_text(value: float | None, horizon: int) -> str:
    if value is None:
        return f"не окупается за {horizon} {years_word(horizon)}"
    return f"{str(value).replace('.', ',')} {years_word(value)}"


class NpvRiskPresenter:
    """Builds profitability, Monte Carlo and tornado rows for the NPV screen and reports."""

    def build(self, investment: float, cash_flows: Sequence[float], rate: float, settings: RiskSettings | None = None) -> RiskView:
        settings = settings or RiskSettings()
        flows = [float(value) for value in cash_flows]
        metrics = profitability_metrics(investment, flows, rate)
        if not flows:
            return RiskView(metrics=metrics, monte_carlo=None, tornado=None)
        monte_carlo = run_monte_carlo(
            investment,
            flows,
            rate,
            investment_spread=settings.investment_spread,
            cash_flow_spread=settings.cash_flow_spread,
            rate_spread=settings.rate_spread,
            runs=settings.runs,
            seed=settings.seed,
        )
        tornado = build_tornado(
            investment,
            flows,
            rate,
            share=max(settings.investment_spread, settings.cash_flow_spread),
            rate_points=settings.rate_spread,
        )
        return RiskView(metrics=metrics, monte_carlo=monte_carlo, tornado=tornado)

    @staticmethod
    def metric_rows(view: RiskView, rate: float) -> list[dict[str, Any]]:
        metrics = view.metrics
        irr = metrics.irr
        if irr is None:
            irr_note = "Нет: NPV одного знака при любой ставке"
        elif irr >= rate:
            irr_note = f"Выше ставки {percent(rate)}: проект выгоден"
        else:
            irr_note = f"Ниже ставки {percent(rate)}: проект не выгоден"
        pi = metrics.profitability_index
        if pi is None:
            pi_note = "Нет вложений"
        else:
            pi_note = "≥ 1: вложения возвращаются с запасом" if pi >= 1 else "< 1: вложения не возвращаются"
        return [
            {"name": "NPV", "value": money(metrics.npv), "note": "≥ 0: проект окупается" if metrics.npv >= 0 else "< 0: не окупается"},
            {"name": "IRR (внутренняя норма доходности)", "value": percent(irr), "note": irr_note},
            {"name": "PI (индекс доходности)", "value": "—" if pi is None else f"{pi:.3f}".replace(".", ","), "note": pi_note},
            {"name": "Простой срок окупаемости", "value": years_text(metrics.simple_payback, metrics.horizon_years), "note": "Без дисконтирования"},
            {"name": "Дисконтированный срок окупаемости", "value": years_text(metrics.discounted_payback, metrics.horizon_years), "note": f"По ставке {percent(rate)}"},
        ]

    @staticmethod
    def monte_carlo_rows(view: RiskView) -> list[dict[str, Any]]:
        result = view.monte_carlo
        if result is None:
            return []
        return [
            {"name": "Прогонов", "value": str(result.runs)},
            {"name": "Средний NPV", "value": money(result.mean)},
            {"name": "Стандартное отклонение", "value": money(result.std_dev)},
            {"name": "P10 (пессимистично)", "value": money(result.p10)},
            {"name": "P50 (медиана)", "value": money(result.p50)},
            {"name": "P90 (оптимистично)", "value": money(result.p90)},
            {"name": "Минимум / максимум", "value": f"{money(result.minimum)} / {money(result.maximum)}"},
            {"name": "Вероятность убытка (NPV < 0)", "value": percent(result.loss_probability)},
        ]

    @staticmethod
    def tornado_rows(view: RiskView) -> list[dict[str, Any]]:
        result = view.tornado
        if result is None:
            return []
        return [
            {
                "factor": bar.label,
                "low": f"{money(bar.low)} ({bar.low_case})",
                "high": f"{money(bar.high)} ({bar.high_case})",
                "range": money(bar.range),
                "break_even": bar.break_even,
            }
            for bar in result.bars
        ]

    @staticmethod
    def verdict(view: RiskView) -> str:
        result = view.monte_carlo
        if result is None:
            return "Нет потоков"
        leader = ""
        if view.tornado is not None and view.tornado.leader:
            leader_bar = next(bar for bar in view.tornado.bars if bar.key == view.tornado.leader)
            leader = f" Сильнее всего влияет: {leader_bar.label.lower()}."
        loss = result.loss_probability
        if loss < 0.1:
            level = "Риск низкий"
        elif loss < 0.3:
            level = "Риск умеренный"
        else:
            level = "Риск высокий"
        return f"{level}: убыток в {percent(loss)} прогонов.{leader}"

    @staticmethod
    def report_section(view: RiskView, rate: float) -> dict[str, Any]:
        """Plain dict for exporters (JSON/Markdown/DOCX)."""
        return {
            "metrics": NpvRiskPresenter.metric_rows(view, rate),
            "monte_carlo": NpvRiskPresenter.monte_carlo_rows(view),
            "tornado": NpvRiskPresenter.tornado_rows(view),
            "verdict": NpvRiskPresenter.verdict(view),
        }
