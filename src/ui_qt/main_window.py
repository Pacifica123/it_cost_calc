from __future__ import annotations

from collections.abc import Callable

from PySide6.QtWidgets import QApplication, QHBoxLayout, QStackedWidget, QVBoxLayout, QWidget
from PySide6.QtWidgets import QMainWindow

from ui_qt.design import ThemeManager
from ui_qt.navigation import DEFAULT_ROOT_ROUTE_ID, ROOT_ROUTES, require_root_route
from ui_qt.navigation.root_menu import RootMenu
from ui_qt.presenters import QtAppPresenter
from ui_qt.screens import ComponentEditorScreen, EnergyScreen, NpvScreen, PlaceholderScreen
from ui_qt.widgets import ActionBar, SettingsPanel, StatusStrip


class QtMainWindow(QMainWindow):
    """Qt shell with root navigation and lazy central screens."""

    def __init__(self, presenter: QtAppPresenter | None = None) -> None:
        super().__init__()
        self.presenter = presenter or QtAppPresenter()
        self._theme_manager = ThemeManager("light")
        self._screen_factories: dict[str, Callable[[], QWidget]] = {}
        self._screen_indexes: dict[str, int] = {}
        self._last_root_route_id = DEFAULT_ROOT_ROUTE_ID

        app = QApplication.instance()
        if app is not None:
            self._theme_manager.apply_to_application(app)

        self.setWindowTitle("IT Cost Calc")
        self.resize(1100, 720)
        self.setMinimumSize(860, 560)
        self.setCentralWidget(self._build_shell())
        self._register_root_screens()
        self.open_root_route(DEFAULT_ROOT_ROUTE_ID)

    def _build_shell(self) -> QWidget:
        root = QWidget(self)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        self.action_bar = ActionBar("IT Cost Calc", root, status=self._data_status())
        self.action_bar.add_primary_action("Демо", self._load_demo_dataset)
        self.action_bar.add_secondary_action("Настройки", self.open_settings)
        self.action_bar.add_secondary_action("Светлая", lambda: self._apply_theme("light"))
        self.action_bar.add_secondary_action("Тёмная", lambda: self._apply_theme("dark"))
        self.action_bar.add_secondary_action("Выход", self.close)

        body = QWidget(root)
        body.setObjectName("contentArea")
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(12)

        self.root_menu = RootMenu(body)
        self.root_menu.route_changed.connect(self.open_root_route)
        self.screen_stack = QStackedWidget(body)

        body_layout.addWidget(self.root_menu, 0)
        body_layout.addWidget(self.screen_stack, 1)

        self.status_strip = StatusStrip("Готово", root)

        layout.addWidget(self.action_bar, 0)
        layout.addWidget(body, 1)
        layout.addWidget(self.status_strip, 0)
        return root

    def _register_root_screens(self) -> None:
        for route in ROOT_ROUTES:
            if route.route_id == "components":
                self._screen_factories[route.route_id] = (
                    lambda: ComponentEditorScreen(self.presenter)
                )
            elif route.route_id == "energy":
                self._screen_factories[route.route_id] = lambda: EnergyScreen(self.presenter)
            elif route.route_id == "npv":
                self._screen_factories[route.route_id] = lambda: NpvScreen(self.presenter)
            else:
                self._screen_factories[route.route_id] = lambda value=route: PlaceholderScreen(value)

    def open_root_route(self, route_id: str) -> None:
        route = require_root_route(route_id)
        self._last_root_route_id = route_id
        self.root_menu.set_active(route_id, emit=False)
        self._open_screen(route_id)
        self.action_bar.title_label.setText(route.title)
        self.action_bar.set_status(route.status)
        self.status_strip.set_message(f"Раздел: {route.label}")

    def open_settings(self) -> None:
        self.root_menu.clear_active()
        self._open_screen("__settings__")
        self.action_bar.title_label.setText("Настройки")
        self.action_bar.set_status("Тема")
        self.status_strip.set_message("Настройки")

    def _open_screen(self, screen_id: str) -> None:
        if screen_id not in self._screen_indexes:
            factory = self._screen_factories.get(screen_id, self._create_settings_screen)
            widget = factory()
            self._screen_indexes[screen_id] = self.screen_stack.addWidget(widget)
        self.screen_stack.setCurrentIndex(self._screen_indexes[screen_id])

    def _create_settings_screen(self) -> QWidget:
        panel = SettingsPanel(
            self,
            current_theme=self._theme_manager.theme_name,
            on_theme_changed=self._apply_theme,
        )
        return panel

    def _load_demo_dataset(self) -> None:
        try:
            self.presenter.load_demo_dataset()
        except Exception as exc:  # pragma: no cover - GUI status fallback
            self.action_bar.set_status("Ошибка демо")
            self.status_strip.set_message("Ошибка демо")
            self.status_strip.setToolTip(str(exc))
            return
        self.action_bar.set_status(self._data_status())
        current = self.screen_stack.currentWidget()
        if hasattr(current, "refresh_data"):
            current.refresh_data()
        self.status_strip.set_message("Демо загружено")

    def _data_status(self) -> str:
        total = self.presenter.total_rows()
        return f"Данные: {total}" if total else "Нет данных"

    def _apply_theme(self, theme_name: str) -> None:
        app = QApplication.instance()
        if app is None:
            return
        applied = self._theme_manager.apply_to_application(app, theme_name)
        self.status_strip.set_message("Тема: тёмная" if applied == "dark" else "Тема: светлая")
