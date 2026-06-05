from __future__ import annotations

from PySide6.QtWidgets import QHBoxLayout, QWidget

from ui_qt.widgets.compact_label import CompactLabel


class StatusStrip(QWidget):
    """Thin status line for short events and errors."""

    def __init__(self, message: str = "Готово", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("surface")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(8)
        self.message_label = CompactLabel(message, self, object_name="statusText")
        layout.addWidget(self.message_label, 1)

    def set_message(self, message: str) -> None:
        self.message_label.setText(message)
