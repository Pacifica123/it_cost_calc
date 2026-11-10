"""Presenter for the «Внедрение» and «Риски и сценарии» tabs of the «Планы» screen."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Mapping, Sequence

from domain.planning.risks import LEVEL_TITLES, RiskInputs, RiskRegister, build_risk_register, risk_matrix
from domain.planning.scenarios import ScenarioResult, ScenarioShift, build_scenarios
from domain.planning.schedule import (
    PlanStep,
    ProjectFacts,
    Schedule,
    build_gantt_tasks,
    build_plan_steps,
    facts_from_components,
    schedule_tasks,
    task_dates,
)
from shared.constants import ANALYSIS_SCOPE_SOFTWARE, ANALYSIS_SCOPE_TECHNICAL
from ui_qt.presenters.app_presenter import QtAppPresenter
from ui_qt.presenters.licenses_presenter import parse_date, ru_date
from ui_qt.presenters.planning_presenter import PlanningInput, money
from ui_qt.presenters.session_extras import load_extra, save_extra

SESSION_KEY = "implementation_settings"

SHIFT_FIELDS: tuple[tuple[str, str], ...] = (
    ("pessimistic_capex", "Пессим.: CAPEX, %"),
    ("pessimistic_opex", "Пессим.: OPEX, %"),
    ("pessimistic_effect", "Пессим.: эффект, %"),
    ("optimistic_capex", "Оптим.: CAPEX, %"),
    ("optimistic_opex", "Оптим.: OPEX, %"),
    ("optimistic_effect", "Оптим.: эффект, %"),
)
DEFAULT_SHIFT_PERCENTS = {
    "pessimistic_capex": 15.0,
    "pessimistic_opex": 20.0,
    "pessimistic_effect": -20.0,
    "optimistic_capex": -10.0,
    "optimistic_opex": -10.0,
    "optimistic_effect": 10.0,
}


@dataclass(frozen=True)
class ImplementationSettings:
    start: date
    durations: dict[str, int] = field(default_factory=dict)
    shifts: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SHIFT_PERCENTS))
    budget: float = 0.0

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], *, today: date | None = None) -> "ImplementationSettings":
        start = parse_date(str(payload.get("start", ""))) or (today or date.today())
        durations: dict[str, int] = {}
        raw = payload.get("durations", {})
        if isinstance(raw, Mapping):
            for key, value in raw.items():
                try:
                    durations[str(key)] = max(1, min(120, int(round(float(value)))))
                except (TypeError, ValueError):
                    continue
        shifts = dict(DEFAULT_SHIFT_PERCENTS)
        raw_shifts = payload.get("shifts", {})
        if isinstance(raw_shifts, Mapping):
            for key in shifts:
                try:
                    shifts[key] = max(-90.0, min(300.0, float(raw_shifts[key])))
                except (KeyError, TypeError, ValueError):
                    continue
        try:
            budget = max(0.0, float(payload.get("budget", 0) or 0))
        except (TypeError, ValueError):
            budget = 0.0
        return cls(start=start, durations=durations, shifts=shifts, budget=budget)

    def to_mapping(self) -> dict[str, Any]:
        return {"start": self.start.isoformat(), "durations": dict(self.durations), "shifts": dict(self.shifts), "budget": self.budget}

    def scenario_shifts(self) -> tuple[ScenarioShift, ...]:
        s = self.shifts
        return (
            ScenarioShift("pessimistic", "Пессимистичный", s["pessimistic_capex"] / 100, s["pessimistic_opex"] / 100, s["pessimistic_effect"] / 100),
            ScenarioShift("base", "Базовый"),
            ScenarioShift("optimistic", "Оптимистичный", s["optimistic_capex"] / 100, s["optimistic_opex"] / 100, s["optimistic_effect"] / 100),
        )


@dataclass(frozen=True)
class ImplementationView:
    facts: ProjectFacts
    steps: tuple[PlanStep, ...]
    schedule: Schedule
    settings: ImplementationSettings
    register: RiskRegister
    scenarios: tuple[ScenarioResult, ...]


class ImplementationPresenter:
    def __init__(self, app_presenter: QtAppPresenter) -> None:
        self.app_presenter = app_presenter

    def saved_settings(self) -> ImplementationSettings:
        return ImplementationSettings.from_mapping(load_extra(self.app_presenter, SESSION_KEY))

    def save_settings(self, settings: ImplementationSettings) -> None:
        save_extra(self.app_presenter, SESSION_KEY, settings.to_mapping())

    def _npv_context(self) -> tuple[float | None, list[float], float | None, float | None, float | None]:
        """Investment, flows, NPV, IRR and loss probability of the last NPV calculation."""
        if not self.app_presenter.get_npv_report():
            return None, [], None, None, None
        inputs = self.app_presenter.get_npv_inputs()
        flows = [float(value) for value in inputs.get("cash_flows", []) if value is not None]
        if not flows:
            return None, [], None, None, None
        from ui_qt.presenters.npv_risk_presenter import NpvRiskPresenter, RiskSettings

        investment = float(inputs.get("investment", 0.0) or 0.0)
        rate = float(inputs.get("discount_rate", 0.1) or 0.0)
        view = NpvRiskPresenter().build(investment, flows, rate, RiskSettings(runs=1000))
        loss = view.monte_carlo.loss_probability if view.monte_carlo else None
        return investment, flows, view.metrics.npv, view.metrics.irr, loss

    def build(self, components: Sequence[Mapping[str, Any]], plan: PlanningInput, settings: ImplementationSettings) -> ImplementationView:
        facts = facts_from_components(components, self.app_presenter.entities_snapshot())
        steps = tuple(build_plan_steps(facts))
        schedule = schedule_tasks(build_gantt_tasks(steps, facts, settings.durations))
        investment, flows, npv, irr, loss = self._npv_context()
        register = build_risk_register(
            RiskInputs(
                has_technical_selection=self.app_presenter.selected_hybrid_candidate(ANALYSIS_SCOPE_TECHNICAL) is not None,
                has_software_selection=self.app_presenter.selected_hybrid_candidate(ANALYSIS_SCOPE_SOFTWARE) is not None,
                workstations=facts.workstations,
                servers=facts.servers,
                network_devices=facts.network_devices,
                software_lines=facts.software_lines,
                has_backup=facts.has_backup,
                one_time=plan.one_time,
                annual_running=plan.opex_annual + plan.electricity_annual,
                npv=npv,
                irr=irr,
                discount_rate=plan.discount_rate_percent / 100,
                loss_probability=loss,
                implementation_days=schedule.total_days,
                budget=settings.budget,
            )
        )
        scenarios = tuple(
            build_scenarios(
                one_time=plan.one_time,
                opex_annual=plan.opex_annual,
                electricity_annual=plan.electricity_annual,
                horizon_years=plan.horizon_years,
                opex_growth=plan.opex_growth_percent / 100,
                electricity_growth=plan.electricity_growth_percent / 100,
                discount_rate=plan.discount_rate_percent / 100,
                npv_investment=investment,
                npv_cash_flows=flows,
                shifts=settings.scenario_shifts(),
            )
        )
        return ImplementationView(facts, steps, schedule, settings, register, scenarios)

    # --- rows ----------------------------------------------------------------------------------

    @staticmethod
    def schedule_rows(view: ImplementationView) -> list[dict[str, Any]]:
        titles = {step.key: step for step in view.steps}
        rows = []
        for task in view.schedule.tasks:
            first, last = task_dates(view.settings.start, task)
            rows.append(
                {
                    "title": task.title,
                    "section": titles[task.key].section,
                    "duration": str(task.duration),
                    "dates": f"{ru_date(first)} – {ru_date(last)}",
                    "slack": "критический путь" if task.critical else f"резерв {task.slack} дн.",
                }
            )
        return rows

    @staticmethod
    def schedule_text(view: ImplementationView) -> str:
        if not view.schedule.tasks:
            return "Нет этапов."
        _first, finish = task_dates(view.settings.start, max(view.schedule.tasks, key=lambda task: task.ef))
        titles = {task.key: task.title[:1].lower() + task.title[1:] for task in view.schedule.tasks}
        path = " → ".join(titles[key] for key in view.schedule.critical_path)
        return (
            f"Внедрение займёт {view.schedule.total_days} рабочих дней: с {ru_date(view.settings.start)} по {ru_date(finish)}. "
            f"Критический путь: {path}."
        )

    @staticmethod
    def risk_rows(view: ImplementationView) -> list[dict[str, Any]]:
        return [
            {
                "title": risk.title,
                "category": risk.category,
                "pi": f"{risk.probability} × {risk.impact} = {risk.score}",
                "level": LEVEL_TITLES[risk.level],
                "response": risk.response,
            }
            for risk in view.register.risks
        ]

    @staticmethod
    def risk_text(view: ImplementationView) -> str:
        register = view.register
        top = [risk.title[:1].lower() + risk.title[1:] for risk in register.risks if risk.level in ("critical", "high")][:3]
        tail = f" Главное: {', '.join(top)}." if top else ""
        return f"Индекс риска проекта {register.index}/100 — {register.label}. Рисков в реестре: {len(register.risks)}.{tail}"

    @staticmethod
    def matrix(view: ImplementationView) -> list[list[int]]:
        return risk_matrix(view.register.risks)

    @staticmethod
    def scenario_rows(view: ImplementationView) -> list[dict[str, Any]]:
        rows = []
        for item in view.scenarios:
            rows.append(
                {
                    "title": item.title,
                    "one_time": money(item.one_time),
                    "running": money(item.annual_running),
                    "tco": money(item.tco),
                    "npv": "—" if item.npv is None else money(item.npv),
                    "irr": "—" if item.irr is None else f"{item.irr * 100:.1f}".replace(".", ",") + " %",
                }
            )
        return rows

    @staticmethod
    def scenario_text(view: ImplementationView) -> str:
        pessimistic, base, optimistic = view.scenarios[0], view.scenarios[1], view.scenarios[-1]
        text = f"TCO от {money(optimistic.tco)} до {money(pessimistic.tco)} (база {money(base.tco)})."
        if pessimistic.npv is not None:
            verdict = "проект окупается даже в пессимистичном сценарии" if pessimistic.npv >= 0 else "в пессимистичном сценарии проект не окупается"
            text += f" NPV от {money(pessimistic.npv)} до {money(optimistic.npv or 0)}: {verdict}."
        else:
            text += " NPV появится после расчёта на экране NPV."
        return text

    def report_sections(self, view: ImplementationView) -> list[tuple[str, dict[str, Any]]]:
        schedule_rows = self.schedule_rows(view)
        return [
            (
                "План внедрения",
                {
                    "paragraphs": [self.schedule_text(view)],
                    "tables": [
                        {
                            "caption": "Таблица – Календарный план внедрения (метод критического пути)",
                            "header": ["Этап", "Раздел", "Дней", "Сроки", "Резерв"],
                            "rows": [[r["title"], r["section"], r["duration"], r["dates"], r["slack"]] for r in schedule_rows],
                        }
                    ],
                },
            ),
            (
                "Реестр рисков и сценарии",
                {
                    "paragraphs": [self.risk_text(view), self.scenario_text(view)],
                    "tables": [
                        {
                            "caption": "Таблица – Реестр рисков (вероятность × влияние)",
                            "header": ["Риск", "Область", "P × I", "Уровень", "Мера"],
                            "rows": [[r["title"], r["category"], r["pi"], r["level"], r["response"]] for r in self.risk_rows(view)],
                        },
                        {
                            "caption": "Таблица – Сценарии затрат и окупаемости",
                            "header": ["Сценарий", "Разово", "В год", "TCO", "NPV", "IRR"],
                            "rows": [[r["title"], r["one_time"], r["running"], r["tco"], r["npv"], r["irr"]] for r in self.scenario_rows(view)],
                        },
                    ],
                },
            ),
        ]
