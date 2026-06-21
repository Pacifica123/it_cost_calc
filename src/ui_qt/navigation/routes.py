from __future__ import annotations

from dataclasses import dataclass

from ui_qt.text_rules import assert_short_text


@dataclass(frozen=True)
class RootRoute:
    """Navigation item for the new mono-screen Qt shell."""

    route_id: str
    label: str
    title: str
    status: str
    details: str

    def __post_init__(self) -> None:
        assert_short_text(self.label, field="RootRoute.label")
        assert_short_text(self.title, field="RootRoute.title")
        assert_short_text(self.status, field="RootRoute.status")


ROOT_ROUTES: tuple[RootRoute, ...] = (
    RootRoute(
        route_id="software",
        label="ПО",
        title="Программное обеспечение",
        status="Подготовка данных",
        details=(
            "Будущий маршрут раздела: данные, GA, AHP, Pareto и гибридный итог. "
            "Расчётная логика остаётся в application/domain слоях."
        ),
    ),
    RootRoute(
        route_id="hardware",
        label="ТО",
        title="Техническое обеспечение",
        status="Подготовка данных",
        details=(
            "Будущий маршрут раздела повторяет ПО, но использует профиль технического "
            "обеспечения и не смешивает области анализа."
        ),
    ),
    RootRoute(
        route_id="catalog",
        label="Каталог",
        title="Каталог оборудования",
        status="Импорт и проверка",
        details=(
            "Внешние JSON, CSV и XLSX сначала проходят staging-проверку. "
            "Только подтверждённые готовые устройства переносятся в варианты ТО."
        ),
    ),
    RootRoute(
        route_id="components",
        label="Компоненты",
        title="Редактор компонентов",
        status="CRUD-контур",
        details=(
            "Здесь будет компактный редактор компонентов с выровненными полями, "
            "короткими действиями и скрытыми необязательными параметрами."
        ),
    ),
    RootRoute(
        route_id="energy",
        label="Энергия",
        title="Электроэнергия",
        status="Расчёт затрат",
        details=(
            "Раздел будет показывать только ключевые вводы и итог, а детальные "
            "параметры останутся в раскрываемом блоке."
        ),
    ),
    RootRoute(
        route_id="npv",
        label="NPV",
        title="NPV-анализ",
        status="Финансы",
        details=(
            "Будущий экран покажет ключевые финансовые метрики карточками, "
            "а подробный денежный поток вынесет ниже или в раскрытие."
        ),
    ),
    RootRoute(
        route_id="export",
        label="Экспорт",
        title="Экспорт",
        status="Отчёты",
        details=(
            "Раздел будет собирать отчёты одним главным действием, без длинных "
            "логов и лишних постоянных панелей."
        ),
    ),
)

ROOT_ROUTE_BY_ID: dict[str, RootRoute] = {route.route_id: route for route in ROOT_ROUTES}
DEFAULT_ROOT_ROUTE_ID = ROOT_ROUTES[0].route_id


def require_root_route(route_id: str) -> RootRoute:
    try:
        return ROOT_ROUTE_BY_ID[route_id]
    except KeyError as exc:
        raise ValueError(f"Unknown root route: {route_id!r}") from exc
