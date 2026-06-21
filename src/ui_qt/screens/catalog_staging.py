from __future__ import annotations

from typing import Any

from ui_qt.models import RowTableModel
from ui_qt.presenters import CatalogStagingPresenter, QtAppPresenter
from ui_qt.dialogs.dns_catalog_import_dialog import DnsCatalogImportDialog
from ui_qt.widgets import CompactLabel, InfoHint, SmartTable

try:
    from PySide6.QtWidgets import (
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QFileDialog,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QLineEdit,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )
except ModuleNotFoundError as exc:
    if exc.name != "PySide6":
        raise
    QComboBox = QDialogButtonBox = QFileDialog = QFrame = QGridLayout = None  # type: ignore[assignment]
    QHBoxLayout = QLineEdit = QMessageBox = QPlainTextEdit = QPushButton = None  # type: ignore[assignment]
    QVBoxLayout = None  # type: ignore[assignment]
    QDialog = QWidget = object  # type: ignore[assignment,misc]

_COLUMNS = (
    "status",
    "readiness",
    "name",
    "source",
    "category",
    "target",
    "price",
    "issues",
)
_HEADERS = {
    "status": "Статус",
    "readiness": "Готовность",
    "name": "Название",
    "source": "Источник",
    "category": "Категория",
    "target": "В ТО",
    "price": "Цена",
    "issues": "Замеч.",
}
_METRIC_LABELS = {
    "ram_gb": "ОЗУ, ГБ",
    "cpu_cores": "CPU, ядра",
    "storage_gb": "Накопитель, ГБ",
    "max_power_watts": "Мощность, Вт",
    "lan_ports": "LAN-порты",
    "lan_speed_mbps": "LAN, Мбит/с",
    "wifi_total_mbps": "Wi-Fi, Мбит/с",
    "ipv6_support": "IPv6",
}
_SOURCE_CATEGORIES = (
    "server",
    "rack_server",
    "tower_server",
    "prebuilt_pc",
    "workstation",
    "desktop",
    "laptop",
    "peripheral",
    "printer",
    "monitor",
    "router",
    "switch",
    "access_point",
    "network_device",
    "cpu",
    "gpu",
    "motherboard",
    "ram",
    "ssd",
    "hdd",
)


class CatalogRecordDialog(QDialog):  # type: ignore[misc,valid-type]
    def __init__(
        self,
        presenter: CatalogStagingPresenter,
        values: dict[str, Any],
        parent: QWidget | None = None,  # type: ignore[valid-type]
    ) -> None:
        super().__init__(parent)
        self.presenter = presenter
        self._entries: dict[str, QLineEdit] = {}
        self._metric_entries: dict[str, QLineEdit] = {}
        self.setWindowTitle("Проверка позиции каталога")
        self.resize(680, 520)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)
        layout.addWidget(
            InfoHint(
                "Исправления сохраняются отдельно от исходных данных. После сохранения запись нужно подтвердить заново.",
                self,
            ),
            0,
        )

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        self._add_line(grid, "title", "Название", 0, 0, values)

        grid.addWidget(CompactLabel("Исходная категория", self), 1, 0)
        self.category_combo = QComboBox(self)
        self.category_combo.setEditable(True)
        self.category_combo.addItems(_SOURCE_CATEGORIES)
        self._set_combo_text(self.category_combo, str(values.get("category") or ""))
        grid.addWidget(self.category_combo, 1, 1)

        self._add_line(grid, "price", "Цена", 2, 0, values)
        self._add_line(grid, "currency", "Валюта", 3, 0, values)

        grid.addWidget(CompactLabel("Раздел ТО", self), 0, 2)
        self.target_combo = QComboBox(self)
        self.target_combo.addItem("Не выбрано", "")
        for value, label in presenter.target_category_options():
            self.target_combo.addItem(label, value)
        self._set_combo_data(self.target_combo, str(values.get("target_category") or ""))
        self.target_combo.currentIndexChanged.connect(self._rebuild_component_types)
        grid.addWidget(self.target_combo, 0, 3)

        grid.addWidget(CompactLabel("Тип компонента", self), 1, 2)
        self.type_combo = QComboBox(self)
        grid.addWidget(self.type_combo, 1, 3)

        self._add_line(grid, "quantity", "Количество", 2, 2, values)
        self._add_line(grid, "client_seats", "Рабочие места", 3, 2, values)
        for column in (1, 3):
            grid.setColumnStretch(column, 1)
        layout.addLayout(grid)

        metrics_frame = QFrame(self)
        metrics_frame.setObjectName("surface")
        metrics_grid = QGridLayout(metrics_frame)
        metrics_grid.setContentsMargins(10, 10, 10, 10)
        metrics_grid.setHorizontalSpacing(10)
        metrics_grid.setVerticalSpacing(8)
        metrics = dict(values.get("attributes") or {})
        for index, field in enumerate(presenter.metric_fields()):
            row = index // 2
            column = 0 if index % 2 == 0 else 2
            metrics_grid.addWidget(
                CompactLabel(_METRIC_LABELS.get(field, field), metrics_frame),
                row,
                column,
            )
            entry = QLineEdit(metrics_frame)
            raw_value = metrics.get(field)
            if isinstance(raw_value, bool):
                entry.setText("да" if raw_value else "нет")
            elif raw_value not in (None, ""):
                entry.setText(str(raw_value))
            entry.setPlaceholderText("нет данных")
            self._metric_entries[field] = entry
            metrics_grid.addWidget(entry, row, column + 1)
        for column in (1, 3):
            metrics_grid.setColumnStretch(column, 1)
        layout.addWidget(metrics_frame, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons, 0)

        self._rebuild_component_types()
        self._set_combo_data(self.type_combo, str(values.get("target_component_type") or ""))

    def values(self) -> dict[str, Any]:
        return {
            "title": self._entries["title"].text(),
            "category": self.category_combo.currentText(),
            "price": self._entries["price"].text(),
            "currency": self._entries["currency"].text(),
            "target_category": str(self.target_combo.currentData() or ""),
            "target_component_type": str(self.type_combo.currentData() or ""),
            "quantity": self._entries["quantity"].text(),
            "client_seats": self._entries["client_seats"].text(),
            "attributes": {
                field: entry.text() for field, entry in self._metric_entries.items()
            },
        }

    def _add_line(
        self,
        grid: QGridLayout,  # type: ignore[valid-type]
        name: str,
        label: str,
        row: int,
        column: int,
        values: dict[str, Any],
    ) -> None:
        grid.addWidget(CompactLabel(label, self), row, column)
        entry = QLineEdit(self)
        entry.setText(str(values.get(name) or ""))
        self._entries[name] = entry
        grid.addWidget(entry, row, column + 1)

    def _rebuild_component_types(self) -> None:
        current = str(self.type_combo.currentData() or "")
        self.type_combo.clear()
        self.type_combo.addItem("Не выбрано", "")
        target = str(self.target_combo.currentData() or "")
        for value, label in self.presenter.component_type_options(target):
            self.type_combo.addItem(label, value)
        self._set_combo_data(self.type_combo, current)

    @staticmethod
    def _set_combo_data(combo: QComboBox, value: str) -> None:  # type: ignore[valid-type]
        index = combo.findData(value)
        combo.setCurrentIndex(index if index >= 0 else 0)

    @staticmethod
    def _set_combo_text(combo: QComboBox, value: str) -> None:  # type: ignore[valid-type]
        index = combo.findText(value)
        if index >= 0:
            combo.setCurrentIndex(index)
        else:
            combo.setEditText(value)


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
            on_edit=self.open_editor,
            show_actions=False,
            multi_select=True,
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
        self.filter_combo = QComboBox(header)
        for option in self.presenter.filter_options():
            self.filter_combo.addItem(option.label, option.value)
        self.filter_combo.currentIndexChanged.connect(lambda _index: self.refresh_data())
        row.addWidget(self.filter_combo, 0)
        self.status_label = CompactLabel("Нет данных", header)
        row.addWidget(self.status_label, 0)
        return header

    def _build_actions(self) -> QHBoxLayout:  # type: ignore[valid-type]
        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)

        open_button = QPushButton("Открыть каталог", self)
        open_button.setProperty("role", "primary")
        open_button.clicked.connect(self.open_catalog)
        dns_button = QPushButton("Собрать из DNS", self)
        dns_button.clicked.connect(self.collect_dns_catalog)
        approve_button = QPushButton("Подтвердить", self)
        approve_button.clicked.connect(self.approve_selected)
        reject_button = QPushButton("Отклонить", self)
        reject_button.clicked.connect(self.reject_selected)
        edit_button = QPushButton("Изменить", self)
        edit_button.clicked.connect(self.open_editor)
        import_button = QPushButton("Импортировать в ТО", self)
        import_button.setProperty("role", "primary")
        import_button.clicked.connect(self.import_approved)

        actions.addWidget(open_button, 0)
        actions.addWidget(dns_button, 0)
        actions.addStretch(1)
        actions.addWidget(reject_button, 0)
        actions.addWidget(edit_button, 0)
        actions.addWidget(approve_button, 0)
        actions.addWidget(import_button, 0)
        return actions

    def refresh_data(self) -> None:
        status_filter = (
            str(self.filter_combo.currentData() or "all")
            if hasattr(self, "filter_combo")
            else "all"
        )
        self.model.replace_rows(self.presenter.table_rows(status_filter))
        self.table.refresh_state()
        summary = self.presenter.summary()
        self.status_label.setText(
            f"Всего {summary.total} · готово {summary.ready} · блокировано {summary.blocked}"
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
        staging_ids = self._selected_ids()
        if not staging_ids:
            return
        result = self.presenter.approve_ids(staging_ids)
        self.refresh_data()
        QMessageBox.information(
            self,
            "Подтверждение",
            f"Подтверждено: {result['updated']}. Заблокировано: {result['blocked']}. "
            f"Пропущено: {result['skipped']}.",
        )

    def collect_dns_catalog(self) -> None:
        dialog = DnsCatalogImportDialog(self.presenter, self)
        if dialog.exec() != QDialog.DialogCode.Accepted or dialog.catalog_path is None:
            return
        try:
            self.presenter.stage_file(dialog.catalog_path)
        except Exception as exc:
            QMessageBox.warning(self, "Ошибка каталога DNS", str(exc))
            return
        self.filter_combo.setCurrentIndex(0)
        self.details.setPlainText(
            "Каталог DNS собран и загружен. Проверьте warnings и подтвердите нужные позиции."
        )
        self.refresh_data()

    def reject_selected(self) -> None:
        staging_ids = self._selected_ids()
        if not staging_ids:
            return
        result = self.presenter.reject_ids(staging_ids)
        self.refresh_data()
        QMessageBox.information(
            self,
            "Отклонение",
            f"Отклонено: {result['updated']}. Пропущено: {result['skipped']}.",
        )

    def open_editor(
        self,
        _index: int | None = None,
        row: dict[str, Any] | None = None,
    ) -> None:
        staging_id = str((row or {}).get("staging_id") or "")
        if not staging_id:
            staging_ids = self._selected_ids(require_single=True)
            if not staging_ids:
                return
            staging_id = staging_ids[0]
        values = self.presenter.edit_values(staging_id)
        if values.get("locked"):
            QMessageBox.information(
                self,
                "Запись уже импортирована",
                "Импортированную позицию изменяйте в данных ТО.",
            )
            return
        dialog = CatalogRecordDialog(self.presenter, values, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            self.presenter.save_edits(staging_id, dialog.values())
        except ValueError as exc:
            QMessageBox.warning(self, "Ошибка исправления", str(exc))
            return
        self.refresh_data()
        self._select_details(staging_id)

    def import_approved(self) -> None:
        result = self.presenter.import_approved()
        self.refresh_data()
        QMessageBox.information(
            self,
            "Импорт каталога",
            f"Добавлено в ТО: {result['imported']}. Уже существовало: {result['skipped']}.",
        )

    def _selected_ids(self, *, require_single: bool = False) -> list[str]:
        staging_ids = [
            str(row.get("staging_id") or "")
            for row in self.table.selected_rows()
            if row.get("staging_id")
        ]
        if not staging_ids:
            QMessageBox.information(self, "Выберите строки", "Сначала выберите позиции каталога.")
            return []
        if require_single and len(staging_ids) != 1:
            QMessageBox.information(
                self,
                "Выберите одну строку",
                "Для редактирования нужна ровно одна позиция.",
            )
            return []
        return staging_ids

    def _show_details(self, _index: int, row: dict[str, Any]) -> None:
        self._select_details(str(row.get("staging_id") or ""))

    def _select_details(self, staging_id: str) -> None:
        if staging_id:
            self.details.setPlainText(self.presenter.details_text_by_id(staging_id))


__all__ = ["CatalogStagingScreen"]
