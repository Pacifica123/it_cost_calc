from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QWidget

from ui_qt.widgets.text_rules import assert_short_text


class CompactLabel(QLabel):
    """QLabel wrapper that enforces the migration text-density rule."""

    def __init__(
        self,
        text: str = "",
        parent: QWidget | None = None,
        *,
        object_name: str | None = None,
        tooltip: str | None = None,
        alignment: Qt.AlignmentFlag | Qt.Alignment = Qt.AlignmentFlag.AlignLeft,
    ) -> None:
        assert_short_text(text, field="CompactLabel.text")
        super().__init__(text, parent)
        self.setAlignment(alignment)
        if object_name:
            self.setObjectName(object_name)
        if tooltip:
            self.setToolTip(tooltip)

    def setText(self, text: str) -> None:  # noqa: N802 - Qt naming
        assert_short_text(text, field="CompactLabel.text")
        super().setText(text)
