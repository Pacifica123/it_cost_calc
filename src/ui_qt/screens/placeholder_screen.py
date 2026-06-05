from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QVBoxLayout, QWidget

from ui_qt.navigation.routes import RootRoute
from ui_qt.widgets import EmptyState, InfoHint


class PlaceholderScreen(QWidget):
    """Short placeholder for root sections not migrated yet."""

    def __init__(self, route: RootRoute, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.route = route
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        state = EmptyState("Экран готов", self, status=route.status, details=route.details)
        hint = InfoHint(route.details, self)
        hint.setToolTip(route.details)

        layout.addWidget(state, 1)
        layout.addWidget(hint, 0, Qt.AlignmentFlag.AlignRight)
