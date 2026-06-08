from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QToolButton, QWidget


class InfoHint(QToolButton):
    """Small tooltip-only hint for long explanations."""

    def __init__(self, tooltip: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("infoHint")
        self.setText("ⓘ")
        self.setToolTip(tooltip)
        self.setCursor(Qt.CursorShape.WhatsThisCursor)
        self.setAutoRaise(True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setAccessibleName("Подсказка")
        self.setFixedSize(22, 22)
