from __future__ import annotations

from typing import Any

from ui_qt.models import RowTableModel
from ui_qt.presenters import CatalogStagingPresenter, QtAppPresenter
from ui_qt.widgets import CompactLabel, InfoHint, SmartTable

try:
    from PySide6.QtWidgets import (
        QFileDialog,
        QFrame,
        QHBoxLayout,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )
except ModuleNotFoundError as exc:
    if exc.name != "PySide6":
        raise
    QFileDialog = QFrame = QHBoxLayout = QMessageBox = QPlainTextEdit = QPushButton = None  # type: ignore[assignment]
    QVBoxLayout = QWidget = object  # type: ignore[assignment,misc]

_COLUMNS = ("status", "name", "source", "category", "target", "price", "issues")
_HEADERS = {
    "status": "Статус",
    "name": "Название",
    "source": "Источник",
    "category": "Категория",
    "target": "В ТО",
    "price": "Цена",
    "issues": "Замеч.",
}


class CatalogStagingScreen(QWidget):  # type: ignore[misc,valid-type]
    """Review external catalog rows before they become technical alternatives."""

    def __init__(
        self,
        app_presenter: QtAppPresenter | None = None,
        parent: QWidget | None = None,  # type: ignore[valid-type]
    ) -> None:
        if QVBoxLayout is None:
            raise RuntimeError("PySide6 is required to create CatalogStagingScreen")
        super().__init__(parent)
        self.presenter = CatalogStagingPresenter(app_presenter)
        self._build_ui()
        self.refresh_data()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addWidget(self._build_header(), 0)

        self.model = RowTableModel([], columns=_COLUMNS, headers=_HEADERS)
        self.table = SmartTable(
            self.model,
            self,
            empty_title="Каталог не загружен",
            empty_status="Откройте JSON, CSV или XLSX",
            on_select=self._show_details,
            show_actions=False,
        )
        layout.addWidget(self.table, 1)

        self.details = QPlainTextEdit(self)
        self.details.setObjectName("surface")
        self.details.setReadOnly(True)
        self.details.setFixedHeight(150)
        self.details.setPlainText("Выберите позицию для проверки.")
        layout.addWidget(self.details, 0)
        layout.addLayout(self._build_actions(), 0)

    def _build_header(self) -> QWidget:  # type: ignore[valid-type]
        header = QFrame(self)
        header.setObjectName("surface")
        row = QHBoxLayout(header)
        row.setContentsMargins(12, 8, 12, 8)
        row.setSpacing(8)
        row.addWidget(CompactLabel("Каталог → проверка → ТО", header), 0)
        row.addWidget(
            InfoHint(
                "В расчёт попадают только подтверждённые готовые устройства. Комплектующие остаются заблокированными.",
                header,
            ),
            0,
        )
        row.addStretch(1)
        self.status_label = CompactLabel("Нет данных", header)
        row.addWidget(self.status_label, 0)
        return header

    def _build_actions(self) -> QHBoxLayout:  # type: ignore[valid-type]
        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)

        open_button = QPushButton("Открыть каталог", self)
        open_button.setProperty("role", "primary")
        open_button.clicked.connect(self.open_catalog)
        approve_button = QPushButton("Подтвердить", self)
        approve_button.clicked.connect(self.approve_selected)
        reject_button = QPushButton("Отклонить", self)
        reject_button.clicked.connect(self.reject_selected)
        import_button = QPushButton("Импортировать в ТО", self)
        import_button.setProperty("role", "primary")
        import_button.clicked.connect(self.import_approved)

        actions.addWidget(open_button, 0)
        actions.addStretch(1)
        actions.addWidget(reject_button, 0)
        actions.addWidget(approve_button, 0)
        actions.addWidget(import_button, 0)
        return actions

    def refresh_data(self) -> None:
        self.model.replace_rows(self.presenter.table_rows())
        self.table.refresh_state()
        summary = self.presenter.summary()
        self.status_label.setText(
            f"Всего {summary.total} · подтверждено {summary.approved} · блокировано {summary.blocked}"
        )
        if summary.total == 0:
            self.details.setPlainText("Выберите «Открыть каталог».")

    def run_primary_action(self) -> None:
        self.open_catalog()

    def open_catalog(self) -> None:
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Открыть каталог оборудования",
            str(self.presenter.app_presenter.paths.repo_root),
            "Каталог (*.json *.csv *.xlsx)",
        )
        if not path:
            return
        try:
            self.presenter.stage_file(path)
        except Exception as exc:
            QMessageBox.warning(self, "Ошибка каталога", str(exc))
            return
        self.details.setPlainText("Каталог загружен. Выберите строку и проверьте замечания.")
        self.refresh_data()

    def approve_selected(self) -> None:
        index = self._selected_index()
        if index is None:
            return
        try:
            self.presenter.approve(index)
        except ValueError as exc:
            QMessageBox.warning(self, "Нельзя подтвердить", str(exc))
            return
        self.refresh_data()
        self._select_details(index)

    def reject_selected(self) -> None:
        index = self._selected_index()
        if index is None:
            return
        try:
            self.presenter.reject(index)
        except ValueError as exc:
            QMessageBox.warning(self, "Нельзя отклонить", str(exc))
            return
        self.refresh_data()
        self._select_details(index)

    def import_approved(self) -> None:
        result = self.presenter.import_approved()
        self.refresh_data()
        QMessageBox.information(
            self,
            "Импорт каталога",
            f"Добавлено в ТО: {result['imported']}. Уже существовало: {result['skipped']}.",
        )

    def _selected_index(self) -> int | None:
        index = self.table.selected_row_index()
        if index is None:
            QMessageBox.information(self, "Выберите строку", "Сначала выберите позицию каталога.")
        return index

    def _show_details(self, index: int, _row: dict[str, Any]) -> None:
        self._select_details(index)

    def _select_details(self, index: int) -> None:
        if 0 <= index < len(self.presenter.records()):
            self.details.setPlainText(self.presenter.details_text(index))


__all__ = ["CatalogStagingScreen"]
