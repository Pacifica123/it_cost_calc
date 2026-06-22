from __future__ import annotations

from ui_qt.models import RowTableModel
from ui_qt.presenters.configuration_details_presenter import ConfigurationDetails
from ui_qt.widgets import CompactLabel, SmartTable

try:
    from PySide6.QtWidgets import QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget
except ModuleNotFoundError as exc:
    if exc.name != "PySide6":
        raise
    QDialog = QDialogButtonBox = QHBoxLayout = QLabel = QVBoxLayout = None  # type: ignore[assignment]
    QWidget = object  # type: ignore[assignment,misc]

_COLUMNS = ("category", "name", "quantity", "cost")
_HEADERS = {
    "category": "Категория",
    "name": "Компонент",
    "quantity": "Кол.",
    "cost": "Стоимость",
}


class ConfigurationDetailsDialog(QDialog):  # type: ignore[misc,valid-type]
    """Minimal read-only composition dialog for a decision candidate."""

    def __init__(
        self,
        details: ConfigurationDetails,
        parent: QWidget | None = None,  # type: ignore[valid-type]
    ) -> None:
        if QVBoxLayout is None:
            raise RuntimeError("PySide6 is required to create ConfigurationDetailsDialog")
        super().__init__(parent)
        self.setWindowTitle("Состав конфигурации")
        self.setModal(True)
        self.resize(760, 430)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 12)
        layout.setSpacing(10)
        title = QLabel(details.name, self)
        title.setObjectName("pageTitle")
        title.setWordWrap(True)
        layout.addWidget(title, 0)

        summary = QHBoxLayout()
        summary.setContentsMargins(0, 0, 0, 0)
        summary.addWidget(CompactLabel(f"Источник: {details.source}", self, object_name="pageStatus"), 0)
        summary.addWidget(CompactLabel(f"Позиций: {details.component_count}", self, object_name="pageStatus"), 0)
        summary.addStretch(1)
        summary.addWidget(CompactLabel(f"Стоимость: {details.total_cost}", self, object_name="pageStatus"), 0)
        layout.addLayout(summary)

        self.model = RowTableModel(details.components, columns=_COLUMNS, headers=_HEADERS)
        self.table = SmartTable(
            self.model,
            self,
            empty_title="Состав не указан",
            empty_status="В конфигурации нет компонентов",
            show_actions=False,
        )
        layout.addWidget(self.table, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons, 0)
