"""Risk register of the IT project: rules over the selected configuration and the calculations.

Every risk has a probability and an impact on a 1–5 scale; the score is P × I (1–25), which
places it in the classic risk matrix. The project index is 100 minus the weights of the levels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

LEVELS = ("critical", "high", "medium", "low")
LEVEL_TITLES = {"critical": "Критичный", "high": "Высокий", "medium": "Средний", "low": "Низкий"}
LEVEL_WEIGHTS = {"critical": 28, "high": 16, "medium": 9, "low": 4}


@dataclass(frozen=True)
class RiskInputs:
    has_technical_selection: bool = False
    has_software_selection: bool = False
    workstations: int = 0
    servers: int = 0
    network_devices: int = 0
    software_lines: int = 0
    has_backup: bool = False
    one_time: float = 0.0
    annual_running: float = 0.0
    npv: float | None = None
    irr: float | None = None
    discount_rate: float = 0.1
    loss_probability: float | None = None
    implementation_days: int = 0
    budget: float = 0.0


@dataclass(frozen=True)
class Risk:
    key: str
    title: str
    category: str
    probability: int
    impact: int
    description: str
    response: str

    @property
    def score(self) -> int:
        return self.probability * self.impact

    @property
    def level(self) -> str:
        if self.score >= 16:
            return "critical"
        if self.score >= 10:
            return "high"
        if self.score >= 5:
            return "medium"
        return "low"


@dataclass(frozen=True)
class RiskRegister:
    risks: tuple[Risk, ...]
    index: int
    label: str


def _label(index: int) -> str:
    if index >= 85:
        return "низкий риск"
    if index >= 65:
        return "умеренный риск"
    if index >= 40:
        return "повышенный риск"
    return "критичный риск"


def build_risk_register(data: RiskInputs) -> RiskRegister:
    risks: list[Risk] = []

    def add(key: str, title: str, category: str, probability: int, impact: int, description: str, response: str) -> None:
        risks.append(Risk(key, title, category, probability, impact, description, response))

    if not data.has_technical_selection and not data.has_software_selection:
        add("no-selection", "Нет итогового выбора конфигурации", "Управление", 5, 4,
            "Hybrid-выбор ТО и ПО не выполнен: расчёты затрат и планы пустые.",
            "Пройдите GA → AHP → Pareto → Hybrid в разделах ПО и ТО.")
    elif not data.has_software_selection:
        add("no-software-selection", "Не выбрано ПО", "Управление", 4, 3,
            "Есть выбор ТО, но нет итогового выбора ПО.", "Выполните Hybrid в разделе ПО.")
    elif not data.has_technical_selection:
        add("no-technical-selection", "Не выбрано ТО", "Управление", 4, 3,
            "Есть выбор ПО, но нет итогового выбора ТО.", "Выполните Hybrid в разделе ТО.")

    if data.servers and not data.network_devices:
        add("server-without-network", "Сервер без сетевого оборудования", "Техника", 4, 4,
            "В конфигурации есть сервер, но нет коммутаторов и маршрутизаторов.",
            "Добавьте сетевое оборудование или подтвердите, что сеть уже есть.")
    if data.servers == 1:
        add("single-server", "Нет резервирования сервера", "Надёжность", 2, 4,
            "Один сервер — единая точка отказа для сервисов.",
            "Предусмотрите резервный сервер, кластер или облачную реплику.")
    if (data.servers or data.workstations) and not data.has_backup:
        add("no-backup", "Нет резервного копирования", "Надёжность", 3, 5,
            "В исходных данных нет услуг или средств резервного копирования.",
            "Добавьте позицию в категорию backup и проверяйте восстановление.")
    if data.workstations and not data.software_lines:
        add("clients-without-software", "Рабочие места без лицензий ПО", "Лицензии", 4, 3,
            "Выбраны рабочие места, но нет лицензий и подписок.",
            "Добавьте ОС, офисный пакет и защитное ПО в раздел ПО.")
    if data.one_time > 0 and data.annual_running > data.one_time * 0.7:
        add("high-running-costs", "Высокая доля текущих затрат", "Финансы", 3, 3,
            "Годовой OPEX с энергией превышает 70 % разовых вложений.",
            "Пересмотрите подписки, аренду и сопровождение; сравните с покупкой.")
    if data.budget > 0 and data.one_time > data.budget:
        add("over-budget", "Превышение бюджета", "Финансы", 5, 4,
            "Разовые вложения больше заданного бюджета.", "Снизьте состав или ищите альтернативы дешевле.")
    if data.npv is not None and data.npv < 0:
        add("negative-npv", "Проект не окупается (NPV < 0)", "Финансы", 4, 5,
            "При заданных потоках и ставке приведённый эффект меньше вложений.",
            "Уточните эффект, снизьте вложения или увеличьте горизонт.")
    if data.irr is not None and data.npv is not None and data.npv >= 0 and data.irr - data.discount_rate < 0.02:
        add("thin-irr-margin", "Малый запас по ставке", "Финансы", 3, 3,
            "IRR превышает ставку дисконтирования меньше чем на 2 п. п.",
            "Проверьте устойчивость торнадо-анализом, заложите резерв.")
    if data.loss_probability is not None:
        if data.loss_probability >= 0.3:
            add("monte-carlo-loss", "Высокая вероятность убытка", "Финансы", 4, 4,
                f"По Монте-Карло NPV < 0 в {data.loss_probability:.0%} прогонов.",
                "Сократите неопределённость: зафиксируйте цены, уточните эффект.")
        elif data.loss_probability >= 0.1:
            add("monte-carlo-loss", "Заметная вероятность убытка", "Финансы", 3, 3,
                f"По Монте-Карло NPV < 0 в {data.loss_probability:.0%} прогонов.",
                "Держите резерв бюджета, следите за ключевым фактором торнадо.")
    if data.implementation_days > 60:
        add("long-implementation", "Долгое внедрение", "Сроки", 3, 3,
            f"План внедрения занимает {data.implementation_days} рабочих дней.",
            "Разбейте внедрение на этапы, распараллельте работы вне критического пути.")
    add("supply-delay", "Задержка поставки оборудования", "Сроки", 3, 2,
        "Сроки поставки зависят от поставщика и импорта.",
        "Закладывайте резерв в закупку, держите альтернативного поставщика.")
    add("price-growth", "Рост цен до закупки", "Финансы", 3, 2,
        "Цены каталога могут вырасти между расчётом и закупкой.",
        "Обновите цены перед закупкой, учитывайте рост цен в «Планах».")

    risks.sort(key=lambda risk: (-risk.score, risk.title))
    penalty = sum(LEVEL_WEIGHTS[risk.level] for risk in risks if risk.key not in ("supply-delay", "price-growth"))
    index = max(0, min(100, 100 - penalty))
    return RiskRegister(tuple(risks), index, _label(index))


def risk_matrix(risks: Sequence[Risk]) -> list[list[int]]:
    """5 × 5 counts: rows = impact 5..1, columns = probability 1..5."""
    matrix = [[0] * 5 for _ in range(5)]
    for risk in risks:
        matrix[5 - risk.impact][risk.probability - 1] += 1
    return matrix
