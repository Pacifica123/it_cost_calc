from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QVBoxLayout, QPushButton, QWidget

from ui_qt.widgets.compact_label import CompactLabel
from ui_qt.widgets.info_hint import InfoHint
from ui_qt.widgets.text_rules import assert_short_text


class EmptyState(QWidget):
    """Compact empty or placeholder state with one suggested action."""

    def __init__(
        self,
        title: str,
        parent: QWidget | None = None,
        *,
        status: str = "Готово",
        details: str | None = None,
        action_text: str | None = None,
        action_callback: Callable[[], None] | None = None,
        compact: bool = False,
    ) -> None:
        assert_short_text(title, field="EmptyState.title")
        assert_short_text(status, field="EmptyState.status")
        if action_text:
            assert_short_text(action_text, field="EmptyState.action")
        super().__init__(parent)
        self.setObjectName("card")
        layout = QVBoxLayout(self)
        if compact:
            layout.setContentsMargins(10, 8, 10, 8)
            layout.setSpacing(3)
        else:
            layout.setContentsMargins(16, 16, 16, 16)
            layout.setSpacing(6)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title_label = CompactLabel(
            title,
            self,
            object_name="pageStatus" if compact else "pageTitle",
            alignment=Qt.AlignmentFlag.AlignCenter,
        )
        status_label = CompactLabel(
            status,
            self,
            object_name="pageStatus",
            alignment=Qt.AlignmentFlag.AlignCenter,
        )
        layout.addWidget(title_label)
        layout.addWidget(status_label)

        if details:
            hint = InfoHint(details, self)
            layout.addWidget(hint, 0, Qt.AlignmentFlag.AlignCenter)

        if action_text:
            button = QPushButton(action_text, self)
            button.setProperty("role", "primary")
            if action_callback:
                button.clicked.connect(action_callback)
            layout.addWidget(button, 0, Qt.AlignmentFlag.AlignCenter)
