from __future__ import annotations

from collections.abc import Callable
from time import perf_counter

from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QApplication, QHBoxLayout, QStackedWidget, QVBoxLayout, QWidget
from PySide6.QtWidgets import QMainWindow

from ui_qt.design import ThemeManager
from ui_qt.navigation import DEFAULT_ROOT_ROUTE_ID, ROOT_ROUTES, require_root_route
from ui_qt.navigation.root_menu import RootMenu
from ui_qt.presenters import QtAppPresenter, QtSettingsPresenter, QtUiSettings
from ui_qt.screens import (
    CatalogStagingScreen,
    ComponentEditorScreen,
    EnergyScreen,
    ExportScreen,
    NpvScreen,
    PlaceholderScreen,
    SoftwareWorkspaceScreen,
    TechnicalWorkspaceScreen,
)
from ui_qt.widgets import ActionBar, SettingsPanel, StatusStrip


class QtMainWindow(QMainWindow):
    """Qt shell with root navigation, settings and lazy central screens."""

    def __init__(self, presenter: QtAppPresenter | None = None) -> None:
        start_time = perf_counter()
        super().__init__()
        self.presenter = presenter or QtAppPresenter()
        self._settings_presenter = QtSettingsPresenter(repo_root=self.presenter.paths.repo_root)
        self._settings = self._settings_presenter.settings
        self._theme_manager = ThemeManager(self._settings.theme)
        self._screen_factories: dict[str, Callable[[], QWidget]] = {}
        self._screen_indexes: dict[str, int] = {}
        self._last_root_route_id = self._settings.start_route or DEFAULT_ROOT_ROUTE_ID
        self._shortcut_refs: list[QShortcut] = []
        self._compact_layout = False
        self._startup_ms = 0.0

        app = QApplication.instance()
        if app is not None:
            self._theme_manager.apply_to_application(app)
            self._apply_scale(self._settings.ui_scale, persist=False)

        self.setWindowTitle("IT Cost Calc")
        self.resize(1100, 720)
        self.setMinimumSize(860, 560)
        self.setCentralWidget(self._build_shell())
        self._register_root_screens()
        self._install_shortcuts()
        self.open_root_route(self._settings.start_route)
        self._apply_responsive_rules()
        self._startup_ms = (perf_counter() - start_time) * 1000.0

    def _build_shell(self) -> QWidget:
        root = QWidget(self)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        self.action_bar = ActionBar("IT Cost Calc", root, status=self._data_status())
        self.action_bar.add_primary_action("Демо", self._load_demo_dataset)
        self.action_bar.add_secondary_action(
            "Демо: обычный",
            lambda: self._load_demo_profile("default"),
        )
        self.action_bar.add_secondary_action(
            "Демо: расчётный",
            lambda: self._load_demo_profile("calculation_control"),
        )
        self.action_bar.add_secondary_action(
            "Демо: нагрузочный ТО",
            lambda: self._load_demo_profile("technical_load_1000"),
        )
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
            if route.route_id == "software":
                self._screen_factories[route.route_id] = (
                    lambda: SoftwareWorkspaceScreen(self.presenter)
                )
            elif route.route_id == "hardware":
                self._screen_factories[route.route_id] = (
                    lambda: TechnicalWorkspaceScreen(self.presenter)
                )
            elif route.route_id == "catalog":
                self._screen_factories[route.route_id] = (
                    lambda: CatalogStagingScreen(self.presenter)
                )
            elif route.route_id == "components":
                self._screen_factories[route.route_id] = (
                    lambda: ComponentEditorScreen(self.presenter)
                )
            elif route.route_id == "energy":
                self._screen_factories[route.route_id] = lambda: EnergyScreen(self.presenter)
            elif route.route_id == "npv":
                self._screen_factories[route.route_id] = lambda: NpvScreen(self.presenter)
            elif route.route_id == "export":
                self._screen_factories[route.route_id] = lambda: ExportScreen(self.presenter)
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
        self.action_bar.set_status("Интерфейс")
        self.status_strip.set_message("Настройки")

    def _open_screen(self, screen_id: str) -> None:
        if screen_id not in self._screen_indexes:
            factory = self._screen_factories.get(screen_id, self._create_settings_screen)
            widget = factory()
            self._screen_indexes[screen_id] = self.screen_stack.addWidget(widget)
        self.screen_stack.setCurrentIndex(self._screen_indexes[screen_id])
        current = self.screen_stack.currentWidget()
        if hasattr(current, "refresh_data"):
            current.refresh_data()

    def _create_settings_screen(self) -> QWidget:
        settings = self._settings
        panel = SettingsPanel(
            self,
            current_theme=self._theme_manager.theme_name,
            current_scale=settings.ui_scale,
            current_start_route=settings.start_route,
            show_advanced=settings.show_advanced,
            on_theme_changed=self._apply_theme,
            on_scale_changed=self._apply_scale,
            on_start_route_changed=self._set_start_route,
            on_advanced_changed=self._set_show_advanced,
        )
        return panel

    def _load_demo_dataset(self) -> None:
        self._load_demo_profile("default")

    def _load_demo_profile(self, profile_id: str) -> None:
        try:
            profiles = {
                str(profile.get("id")): profile
                for profile in self.presenter.list_demo_profiles()
            }
            if profile_id == "default":
                self.presenter.load_demo_dataset()
            else:
                self.presenter.load_demo_profile(profile_id)
            title = str(profiles.get(profile_id, {}).get("title") or "Демо")
            message = f"{title} загружен"
        except Exception as exc:  # pragma: no cover - GUI status fallback
            self.action_bar.set_status("Ошибка демо")
            self.status_strip.set_message("Ошибка демо")
            self.status_strip.setToolTip(str(exc))
            return
        self.action_bar.set_status(self._data_status())
        current = self.screen_stack.currentWidget()
        if hasattr(current, "refresh_data"):
            current.refresh_data()
        self.status_strip.set_message(message)

    def _data_status(self) -> str:
        total = self.presenter.total_rows()
        return f"Данные: {total}" if total else "Нет данных"

    def _apply_theme(self, theme_name: str) -> None:
        app = QApplication.instance()
        if app is None:
            return
        applied = self._theme_manager.apply_to_application(app, theme_name)
        self._settings = self._settings_presenter.update(theme=applied)
        self.status_strip.set_message("Тема: тёмная" if applied == "dark" else "Тема: светлая")

    def _apply_scale(self, scale: float, *, persist: bool = True) -> None:
        app = QApplication.instance()
        if app is None:
            return
        safe_scale = QtUiSettings.from_mapping({"ui_scale": scale}).ui_scale
        font = app.font()
        font.setPointSizeF(10.0 * safe_scale)
        app.setFont(font)
        if persist:
            self._settings = self._settings_presenter.update(ui_scale=safe_scale)
            self.status_strip.set_message(f"Масштаб: {int(safe_scale * 100)}%")

    def _set_start_route(self, route_id: str) -> None:
        self._settings = self._settings_presenter.update(start_route=route_id)
        route = require_root_route(self._settings.start_route)
        self.status_strip.set_message(f"Старт: {route.label}")

    def _set_show_advanced(self, enabled: bool) -> None:
        self._settings = self._settings_presenter.update(show_advanced=enabled)
        self.status_strip.set_message("Расширенные: да" if enabled else "Расширенные: нет")

    def _install_shortcuts(self) -> None:
        shortcuts: tuple[tuple[str, Callable[[], None]], ...] = (
            ("Ctrl+L", self._load_demo_dataset),
            ("Ctrl+Return", self._trigger_primary_action),
            ("Alt+Right", self._next_workflow_step),
            ("Alt+Left", self._previous_workflow_step),
            ("F1", self._show_current_help),
            ("Esc", self._close_overlay),
        )
        for key, callback in shortcuts:
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(callback)
            self._shortcut_refs.append(shortcut)

    def _trigger_primary_action(self) -> None:
        current = self.screen_stack.currentWidget()
        if hasattr(current, "run_primary_action"):
            current.run_primary_action()
            self.status_strip.set_message("Действие выполнено")

    def _next_workflow_step(self) -> None:
        current = self.screen_stack.currentWidget()
        if hasattr(current, "next_workflow_step"):
            current.next_workflow_step()

    def _previous_workflow_step(self) -> None:
        current = self.screen_stack.currentWidget()
        if hasattr(current, "previous_workflow_step"):
            current.previous_workflow_step()

    def _show_current_help(self) -> None:
        current = self.screen_stack.currentWidget()
        current_help = current.toolTip() if current is not None else ""
        self.status_strip.set_message("F1: помощь")
        if current_help:
            self.status_strip.setToolTip(current_help)

    def _close_overlay(self) -> None:
        if self.action_bar.title_label.text() == "Настройки":
            self.open_root_route(self._last_root_route_id)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._apply_responsive_rules()

    def _apply_responsive_rules(self) -> None:
        compact = self.width() < 980
        if compact == self._compact_layout:
            return
        self._compact_layout = compact
        self.root_menu.set_compact(compact)
        self.action_bar.set_compact(compact)
        root_layout = self.centralWidget().layout() if self.centralWidget() else None
        if root_layout is not None:
            margin = 12 if compact else 18
            root_layout.setContentsMargins(margin, margin, margin, margin)

    @property
    def startup_ms(self) -> float:
        return self._startup_ms
