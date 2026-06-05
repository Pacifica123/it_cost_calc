from __future__ import annotations

from collections.abc import Callable

from PySide6.QtWidgets import QButtonGroup, QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from ui_qt.design.tokens import ThemeName
from ui_qt.widgets.compact_label import CompactLabel


class SettingsPanel(QWidget):
    """Small settings surface for theme and density preferences."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        current_theme: ThemeName = "light",
        on_theme_changed: Callable[[ThemeName], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("surface")
        self._on_theme_changed = on_theme_changed

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        layout.addWidget(CompactLabel("Настройки", self, object_name="pageTitle"))

        row = QWidget(self)
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(8)
        row_layout.addWidget(CompactLabel("Тема", row))

        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        for name, text in (("light", "Светлая"), ("dark", "Тёмная")):
            button = QPushButton(text, row)
            button.setCheckable(True)
            button.setChecked(name == current_theme)
            button.clicked.connect(lambda _checked=False, value=name: self._emit_theme(value))
            self._group.addButton(button)
            row_layout.addWidget(button)

        row_layout.addStretch(1)
        layout.addWidget(row)
        layout.addStretch(1)

    def _emit_theme(self, value: str) -> None:
        if self._on_theme_changed:
            self._on_theme_changed("dark" if value == "dark" else "light")
