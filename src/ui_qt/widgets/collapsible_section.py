from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QToolButton, QVBoxLayout, QWidget

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

        self._toggle = QToolButton(self)
        self._toggle.setText(title)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(expanded)
        self._toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self._toggle.clicked.connect(self._sync_state)

        header = QWidget(self)
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(0)
        header_layout.addWidget(self._toggle)
        if tooltip:
            header_layout.addWidget(InfoHint(tooltip, header), 0, Qt.AlignmentFlag.AlignLeft)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(8)
        layout.addWidget(header)
        layout.addWidget(content)
        self._sync_state()

    def _sync_state(self) -> None:
        expanded = self._toggle.isChecked()
        self._toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self._content.setVisible(expanded)
