from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QMenu, QPushButton, QToolButton, QWidget

from ui_qt.widgets.compact_label import CompactLabel
from ui_qt.widgets.text_rules import assert_short_text


class ActionBar(QWidget):
    """Compact top or bottom bar with one clear primary action."""

    def __init__(
        self,
        title: str,
        parent: QWidget | None = None,
        *,
        status: str = "Готово",
    ) -> None:
        assert_short_text(title, field="ActionBar.title")
        assert_short_text(status, field="ActionBar.status")
        super().__init__(parent)
        self.setObjectName("surface")
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(12, 8, 12, 8)
        self._layout.setSpacing(8)

        self.title_label = CompactLabel(title, self, object_name="pageTitle")
        self.status_label = CompactLabel(status, self, object_name="pageStatus")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self.more_button = QToolButton(self)
        self.more_button.setObjectName("moreButton")
        self.more_button.setText("Ещё ▾")
        self.more_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.more_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.more_menu = QMenu(self.more_button)
        self.more_button.setMenu(self.more_menu)
        self.more_button.hide()

        self._layout.addWidget(self.title_label, 0)
        self._layout.addStretch(1)
        self._layout.addWidget(self.status_label, 0)
        self._layout.addWidget(self.more_button, 0)

    def set_status(self, status: str) -> None:
        self.status_label.setText(status)

    def add_primary_action(self, text: str, callback: Callable[[], None]) -> QPushButton:
        assert_short_text(text, field="primary action")
        button = QPushButton(text, self)
        button.setProperty("role", "primary")
        button.clicked.connect(lambda _checked=False: callback())
        self._layout.insertWidget(max(self._layout.count() - 2, 0), button, 0)
        return button

    def add_secondary_action(self, text: str, callback: Callable[[], None]) -> None:
        assert_short_text(text, field="secondary action")
        action = self.more_menu.addAction(text)
        action.triggered.connect(lambda _checked=False: callback())
        self.more_button.show()
