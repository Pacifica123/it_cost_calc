from __future__ import annotations

from collections.abc import Callable, Iterable

from PySide6.QtWidgets import QButtonGroup, QCheckBox, QComboBox, QFormLayout, QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from ui_qt.design.tokens import ThemeName
from ui_qt.navigation.routes import ROOT_ROUTES
from ui_qt.presenters.settings_presenter import allowed_scales
from ui_qt.widgets.compact_label import CompactLabel


class SettingsPanel(QWidget):
    """Small settings surface for theme, scale and start screen."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        current_theme: ThemeName = "light",
        current_scale: float = 1.0,
        current_start_route: str = "software",
        show_advanced: bool = False,
        on_theme_changed: Callable[[ThemeName], None] | None = None,
        on_scale_changed: Callable[[float], None] | None = None,
        on_start_route_changed: Callable[[str], None] | None = None,
        on_advanced_changed: Callable[[bool], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("surface")
        self._on_theme_changed = on_theme_changed
        self._on_scale_changed = on_scale_changed
        self._on_start_route_changed = on_start_route_changed
        self._on_advanced_changed = on_advanced_changed

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        layout.addWidget(CompactLabel("Настройки", self, object_name="pageTitle"))
        layout.addWidget(CompactLabel("Интерфейс", self, object_name="pageStatus"))

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(10)
        form.addRow("Тема", self._build_theme_row(current_theme))
        form.addRow("Масштаб", self._build_scale_combo(current_scale))
        form.addRow("Старт", self._build_start_combo(current_start_route))

        self.advanced_check = QCheckBox("Расширенные", self)
        self.advanced_check.setChecked(bool(show_advanced))
        self.advanced_check.setToolTip("Открывать дополнительные параметры по умолчанию.")
        self.advanced_check.toggled.connect(self._emit_advanced)
        form.addRow("Вид", self.advanced_check)

        layout.addLayout(form)
        layout.addStretch(1)

    def _build_theme_row(self, current_theme: ThemeName) -> QWidget:
        row = QWidget(self)
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(8)
        self._theme_group = QButtonGroup(row)
        self._theme_group.setExclusive(True)
        for name, text in (("light", "Светлая"), ("dark", "Тёмная")):
            button = QPushButton(text, row)
            button.setCheckable(True)
            button.setChecked(name == current_theme)
            button.clicked.connect(lambda _checked=False, value=name: self._emit_theme(value))
            self._theme_group.addButton(button)
            row_layout.addWidget(button, 0)
        row_layout.addStretch(1)
        return row

    def _build_scale_combo(self, current_scale: float) -> QComboBox:
        combo = QComboBox(self)
        for scale in allowed_scales():
            combo.addItem(_scale_label(scale), float(scale))
        combo.setCurrentIndex(_closest_index(allowed_scales(), current_scale))
        combo.currentIndexChanged.connect(lambda _index: self._emit_scale(combo.currentData()))
        return combo

    def _build_start_combo(self, current_start_route: str) -> QComboBox:
        combo = QComboBox(self)
        for route in ROOT_ROUTES:
            combo.addItem(route.label, route.route_id)
        route_ids = [route.route_id for route in ROOT_ROUTES]
        if current_start_route in route_ids:
            combo.setCurrentIndex(route_ids.index(current_start_route))
        combo.currentIndexChanged.connect(lambda _index: self._emit_start_route(combo.currentData()))
        return combo

    def _emit_theme(self, value: str) -> None:
        if self._on_theme_changed:
            self._on_theme_changed("dark" if value == "dark" else "light")

    def _emit_scale(self, value: object) -> None:
        if self._on_scale_changed:
            self._on_scale_changed(float(value or 1.0))

    def _emit_start_route(self, value: object) -> None:
        if self._on_start_route_changed:
            self._on_start_route_changed(str(value or "software"))

    def _emit_advanced(self, enabled: bool) -> None:
        if self._on_advanced_changed:
            self._on_advanced_changed(bool(enabled))


def _scale_label(scale: float) -> str:
    return f"{int(round(scale * 100))}%"


def _closest_index(values: Iterable[float], target: float) -> int:
    items = tuple(values)
    if not items:
        return 0
    closest = min(items, key=lambda item: abs(item - float(target)))
    return items.index(closest)
