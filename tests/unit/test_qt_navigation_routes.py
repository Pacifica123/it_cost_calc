from ui_qt.navigation.routes import DEFAULT_ROOT_ROUTE_ID, ROOT_ROUTES, require_root_route
from ui_qt.text_rules import count_words


def test_root_routes_keep_expected_sections_and_short_labels():
    labels = [route.label for route in ROOT_ROUTES]
    titles = [route.title for route in ROOT_ROUTES]

    assert labels == ["ПО", "ТО", "Каталог", "Компоненты", "Энергия", "NPV", "Экспорт"]
    assert titles == [
        "Программное обеспечение",
        "Техническое обеспечение",
        "Каталог оборудования",
        "Редактор компонентов",
        "Электроэнергия",
        "NPV-анализ",
        "Экспорт",
    ]
    assert all(count_words(route.label) <= 10 for route in ROOT_ROUTES)
    assert all(count_words(route.title) <= 10 for route in ROOT_ROUTES)
    assert all(count_words(route.status) <= 10 for route in ROOT_ROUTES)


def test_default_route_is_first_software_workspace():
    route = require_root_route(DEFAULT_ROOT_ROUTE_ID)

    assert route.route_id == "software"
    assert route.label == "ПО"
