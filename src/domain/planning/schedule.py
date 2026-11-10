"""Implementation plan and its Gantt schedule by the critical path method (CPM).

Durations are in working days (Mon–Fri). A task starts when all its predecessors are finished.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Iterable, Mapping, Sequence

MAX_TASK_DAYS = 120


@dataclass(frozen=True)
class PlanStep:
    key: str
    title: str
    section: str
    description: str


@dataclass(frozen=True)
class ProjectFacts:
    """What the plan depends on, taken from the selected configuration and the entities."""

    workstations: int = 0
    servers: int = 0
    network_devices: int = 0
    software_lines: int = 0
    capital_lines: int = 0
    has_backup: bool = False
    has_migration: bool = False


STANDARD_DEPS: dict[str, tuple[str, ...]] = {
    "requirements": (),
    "procurement": ("requirements",),
    "hardware": ("procurement",),
    "software": ("hardware",),
    "network": ("hardware",),
    "backup": ("network", "software"),
    "migration": ("software", "network"),
    "testing": ("hardware", "software", "network", "backup", "migration"),
    "training": ("testing",),
    "handover": ("testing", "training"),
}


def build_plan_steps(facts: ProjectFacts) -> list[PlanStep]:
    steps = [
        PlanStep("requirements", "Уточнить требования", "Подготовка",
                 f"Зафиксировать бюджет, состав и ограничения. Рабочих мест: {facts.workstations}."),
        PlanStep("procurement", "Закупить оборудование и ПО", "Закупка",
                 f"Конкурентная закупка и поставка по {facts.capital_lines} позициям."),
    ]
    if facts.workstations or facts.servers or facts.network_devices:
        steps.append(PlanStep("hardware", "Развернуть оборудование", "Внедрение",
                              "Рабочие места, серверная часть, сетевое оборудование и периферия."))
    if facts.software_lines:
        steps.append(PlanStep("software", "Установить и активировать ПО", "Внедрение",
                              "ОС, офисные пакеты, прикладное и защитное ПО, активация лицензий."))
    if facts.servers or facts.network_devices:
        steps.append(PlanStep("network", "Настроить сеть и доступы", "Настройка",
                              "Адресация, VLAN, маршрутизация, учётные записи и права."))
    if facts.has_backup:
        steps.append(PlanStep("backup", "Настроить резервное копирование", "Надёжность",
                              "Расписание копий и тестовое восстановление."))
    if facts.has_migration:
        steps.append(PlanStep("migration", "Перенести данные", "Внедрение",
                              "Миграция пользовательских данных, профилей и баз."))
    steps.extend(
        [
            PlanStep("testing", "Провести приёмочное тестирование", "Контроль",
                     "Рабочие места, сервисы, печать, сеть, сценарии отказа."),
            PlanStep("training", "Обучить пользователей", "Контроль",
                     "Инструктаж по новым рабочим местам и сервисам."),
            PlanStep("handover", "Передать в эксплуатацию", "Завершение",
                     "Итоговая конфигурация, регламенты сопровождения, ответственные."),
        ]
    )
    return steps


def clamp_days(value: float) -> int:
    try:
        number = round(float(value))
    except (TypeError, ValueError):
        number = 1
    return max(1, min(MAX_TASK_DAYS, int(number) or 1))


def default_duration(key: str, facts: ProjectFacts) -> int:
    seats = max(1, facts.workstations)
    if key == "requirements":
        return 5 if seats > 30 else 3
    if key == "procurement":
        return clamp_days(10 + -(-facts.capital_lines // 10))
    if key == "hardware":
        return clamp_days(-(-seats // 8) + 1 + facts.servers)
    if key == "software":
        return clamp_days(-(-seats // 10) + 1)
    if key == "network":
        return 5 if seats > 30 or facts.servers > 2 else 3
    if key == "backup":
        return 2
    if key == "migration":
        return clamp_days(2 + -(-seats // 15))
    if key == "testing":
        return 5 if seats > 30 else 3
    if key == "training":
        return 1 if seats <= 15 else 2
    if key == "handover":
        return 2
    return 3


@dataclass(frozen=True)
class GanttTask:
    key: str
    title: str
    duration: int
    deps: tuple[str, ...]


@dataclass
class ScheduledTask:
    key: str
    title: str
    duration: int
    deps: tuple[str, ...]
    es: int = 0
    ef: int = 0
    ls: int = 0
    lf: int = 0
    slack: int = 0
    critical: bool = False


@dataclass(frozen=True)
class Schedule:
    tasks: tuple[ScheduledTask, ...]
    total_days: int
    critical_path: tuple[str, ...] = field(default_factory=tuple)


def build_gantt_tasks(steps: Sequence[PlanStep], facts: ProjectFacts, overrides: Mapping[str, float] | None = None) -> list[GanttTask]:
    """Standard predecessors; a missing step passes its own predecessors on, so the chain never breaks."""
    overrides = overrides or {}
    present = {step.key for step in steps}

    def resolve(key: str, seen: set[str]) -> list[str]:
        result: list[str] = []
        for dep in STANDARD_DEPS.get(key, ()):
            if dep in seen:
                continue
            seen.add(dep)
            if dep in present:
                result.append(dep)
            else:
                result.extend(resolve(dep, seen))
        return list(dict.fromkeys(result))

    ancestors: dict[str, set[str]] = {}
    tasks: list[GanttTask] = []
    for index, step in enumerate(steps):
        if step.key in STANDARD_DEPS:
            resolved = resolve(step.key, set())
        else:
            resolved = [steps[index - 1].key] if index else []
        deps = [dep for dep in resolved if not any(other != dep and dep in ancestors.get(other, set()) for other in resolved)]
        everything = set(deps)
        for dep in deps:
            everything |= ancestors.get(dep, set())
        ancestors[step.key] = everything
        duration = clamp_days(overrides[step.key]) if step.key in overrides else default_duration(step.key, facts)
        tasks.append(GanttTask(step.key, step.title, duration, tuple(deps)))
    return tasks


def schedule_tasks(tasks: Sequence[GanttTask]) -> Schedule:
    """Forward and backward pass. Predecessors must come first in ``tasks``."""
    by_key: dict[str, ScheduledTask] = {}
    for task in tasks:
        es = max((by_key[dep].ef for dep in task.deps if dep in by_key), default=0)
        by_key[task.key] = ScheduledTask(task.key, task.title, task.duration, task.deps, es=es, ef=es + task.duration)
    total = max((item.ef for item in by_key.values()), default=0)
    successors: dict[str, list[str]] = {}
    for task in tasks:
        for dep in task.deps:
            successors.setdefault(dep, []).append(task.key)
    for task in reversed(tasks):
        current = by_key[task.key]
        following = successors.get(task.key, [])
        current.lf = min((by_key[key].ls for key in following), default=total)
        current.ls = current.lf - current.duration
        current.slack = current.ls - current.es
        current.critical = current.slack == 0
    scheduled = tuple(by_key[task.key] for task in tasks)
    critical = tuple(item.key for item in sorted((t for t in scheduled if t.critical), key=lambda t: t.es))
    return Schedule(scheduled, total, critical)


def next_working_day(value: date) -> date:
    while value.weekday() >= 5:
        value += timedelta(days=1)
    return value


def add_working_days(start: date, offset: int) -> date:
    """Date of working day number ``offset`` (0 = the start day itself)."""
    result = next_working_day(start)
    left = max(0, int(offset))
    while left > 0:
        result += timedelta(days=1)
        if result.weekday() < 5:
            left -= 1
    return result


def task_dates(start: date, task: ScheduledTask) -> tuple[date, date]:
    """First and last working day of a task (inclusive)."""
    return add_working_days(start, task.es), add_working_days(start, max(task.es, task.ef - 1))


def facts_from_components(components: Iterable[Mapping[str, object]], entities: Mapping[str, object] | None = None) -> ProjectFacts:
    def qty(item: Mapping[str, object]) -> int:
        try:
            return max(0, int(round(float(item.get("quantity", 1) or 1))))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 1

    workstations = servers = network = software = capital = 0
    for item in components:
        category = str(item.get("source_category") or item.get("category") or "")
        if category == "client":
            workstations += qty(item)
            capital += 1
        elif category == "server":
            servers += qty(item)
            capital += 1
        elif category == "network":
            network += qty(item)
            capital += 1
        elif category == "licenses":
            software += 1
            capital += 1
        elif category == "subscription_licenses":
            software += 1
    entities = entities or {}

    def non_empty(name: str) -> bool:
        value = entities.get(name)
        return isinstance(value, list) and len(value) > 0

    return ProjectFacts(
        workstations=workstations,
        servers=servers,
        network_devices=network,
        software_lines=software,
        capital_lines=capital,
        has_backup=non_empty("backup"),
        has_migration=non_empty("migration"),
    )

