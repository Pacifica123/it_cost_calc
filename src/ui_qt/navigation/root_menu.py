from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QButtonGroup, QFrame, QPushButton, QVBoxLayout, QWidget

from ui_qt.navigation.routes import DEFAULT_ROOT_ROUTE_ID, ROOT_ROUTES


class RootMenu(QFrame):
    """Compact root navigation for one-screen Qt workflows."""

    route_changed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("rootNav")
        self._buttons: dict[str, QPushButton] = {}
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        for route in ROOT_ROUTES:
            button = QPushButton(route.label, self)
            button.setObjectName("rootMenuButton")
            button.setCheckable(True)
            button.setToolTip(route.title)
            button.clicked.connect(lambda _checked=False, value=route.route_id: self.set_active(value))
            self._group.addButton(button)
            self._buttons[route.route_id] = button
            layout.addWidget(button)

        layout.addStretch(1)
        self.set_active(DEFAULT_ROOT_ROUTE_ID, emit=False)

    def set_active(self, route_id: str, *, emit: bool = True) -> None:
        button = self._buttons.get(route_id)
        if button is None:
            raise ValueError(f"Unknown root route: {route_id!r}")
        button.setChecked(True)
        if emit:
            self.route_changed.emit(route_id)

    def clear_active(self) -> None:
        self._group.setExclusive(False)
        for button in self._buttons.values():
            button.setChecked(False)
        self._group.setExclusive(True)
