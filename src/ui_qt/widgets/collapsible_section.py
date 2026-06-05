from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QToolButton, QVBoxLayout, QWidget

from ui_qt.widgets.info_hint import InfoHint
from ui_qt.widgets.text_rules import assert_short_text


class CollapsibleSection(QFrame):
    """Optional content block hidden until the user asks for details."""

    def __init__(
        self,
        title: str,
        content: QWidget,
        parent: QWidget | None = None,
        *,
        tooltip: str | None = None,
        expanded: bool = False,
    ) -> None:
        assert_short_text(title, field="CollapsibleSection.title")
        super().__init__(parent)
        self.setObjectName("surface")
        self._content = content
        self._toggle_callbacks: list[Callable[[bool], None]] = []
        self._last_expanded = bool(expanded)

        self._toggle = QToolButton(self)
        self._toggle.setText(title)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(expanded)
        self._toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self._toggle.clicked.connect(lambda: self._sync_state(notify=True))

        header = QWidget(self)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(6)
        header_layout.addWidget(self._toggle, 0)
        if tooltip:
            header_layout.addWidget(InfoHint(tooltip, header), 0)
        header_layout.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(8)
        layout.addWidget(header)
        layout.addWidget(content)
        self._sync_state()

    def is_expanded(self) -> bool:
        return bool(self._toggle.isChecked())

    def set_expanded(self, expanded: bool) -> None:
        self._toggle.setChecked(expanded)
        self._sync_state()

    def add_toggle_callback(self, callback: Callable[[bool], None]) -> None:
        """Register a lightweight callback for user-driven expansion changes."""
        self._toggle_callbacks.append(callback)

    def _sync_state(self, *, notify: bool = False) -> None:
        expanded = self._toggle.isChecked()
        self._toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self._content.setVisible(expanded)
        if expanded:
            self.setMaximumHeight(16777215)
        else:
            self.setMaximumHeight(self.sizeHint().height())
        self.updateGeometry()
        state_changed = expanded != self._last_expanded
        self._last_expanded = bool(expanded)
        if notify and state_changed:
            for callback in self._toggle_callbacks:
                callback(bool(expanded))
