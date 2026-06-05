from __future__ import annotations

from ui_qt.models import RowTableModel
from ui_qt.presenters import ExportPresenter, QtAppPresenter
from ui_qt.widgets import CollapsibleSection, CompactLabel, EmptyState, InfoHint

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QComboBox,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QHeaderView,
        QPushButton,
        QStackedLayout,
        QTableView,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ModuleNotFoundError as exc:
    if exc.name != "PySide6":
        raise
    Qt = None  # type: ignore[assignment]
    QComboBox = QFrame = QGridLayout = QHBoxLayout = QHeaderView = QPushButton = None  # type: ignore[assignment]
    QStackedLayout = QTableView = QTextEdit = QVBoxLayout = None  # type: ignore[assignment]
    QWidget = object  # type: ignore[assignment,misc]


class ExportScreen(QWidget):  # type: ignore[misc,valid-type]
    """Compact Qt screen for reports and file export."""

    def __init__(
        self,
        app_presenter: QtAppPresenter | None = None,
        parent: QWidget | None = None,  # type: ignore[valid-type]
    ) -> None:
        if QComboBox is None:
            raise RuntimeError("PySide6 is required to create ExportScreen")
        super().__init__(parent)
        self.presenter = ExportPresenter(app_presenter)
        self.files_model = RowTableModel(
            columns=("kind", "path"),
            headers={"kind": "Тип", "path": "Файл"},
        )
        self._summary_labels: dict[str, CompactLabel] = {}
        self._mode_ids: list[str] = []
        self._build_ui()
        self.refresh_data()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addWidget(self._build_header(), 0)
        layout.addWidget(self._build_summary(), 0)
        layout.addWidget(self._build_controls(), 0)
        layout.addWidget(self._build_details(), 1)
        layout.addLayout(self._build_actions(), 0)

    def _build_header(self) -> QWidget:  # type: ignore[valid-type]
        header = QFrame(self)
        header.setObjectName("surface")
        row = QHBoxLayout(header)
        row.setContentsMargins(12, 8, 12, 8)
        row.setSpacing(8)
        row.addWidget(CompactLabel("Экспорт отчётов", header), 0)
        row.addWidget(
            InfoHint(
                "Экран собирает файлы в data/generated и не запускает анализ заново.",
                header,
            ),
            0,
        )
        row.addStretch(1)
        self.status_label = CompactLabel("Готово", header)
        row.addWidget(self.status_label, 0)
        return header

    def _build_summary(self) -> QWidget:  # type: ignore[valid-type]
        panel = QFrame(self)
        panel.setObjectName("surface")
        grid = QGridLayout(panel)
        grid.setContentsMargins(12, 10, 12, 10)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        for index, (key, title) in enumerate(
            (
                ("rows", "Данных"),
                ("npv", "NPV"),
                ("files", "Файлов"),
                ("folder", "Папка"),
            )
        ):
            card = QFrame(panel)
            card.setObjectName("card")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(12, 10, 12, 10)
            card_layout.setSpacing(4)
            card_layout.addWidget(CompactLabel(title, card, object_name="pageStatus"), 0)
            value = CompactLabel("—", card, object_name="pageTitle")
            card_layout.addWidget(value, 0)
            self._summary_labels[key] = value
            grid.addWidget(card, 0, index)
            grid.setColumnStretch(index, 1)
        return panel

    def _build_controls(self) -> QWidget:  # type: ignore[valid-type]
        panel = QFrame(self)
        panel.setObjectName("surface")
        row = QHBoxLayout(panel)
        row.setContentsMargins(12, 8, 12, 8)
        row.setSpacing(8)
        row.addWidget(CompactLabel("Формат", panel), 0)
        self.mode_combo = QComboBox(panel)
        for mode in self.presenter.modes():
            self._mode_ids.append(mode.mode_id)
            self.mode_combo.addItem(mode.label, mode.mode_id)
        self.mode_combo.currentIndexChanged.connect(self._sync_mode_hint)
        row.addWidget(self.mode_combo, 1)
        self.mode_hint = CompactLabel("", panel, object_name="pageStatus")
        row.addWidget(self.mode_hint, 2)
        return panel

    def _build_details(self) -> QWidget:  # type: ignore[valid-type]
        details = QWidget(self)
        details.setObjectName("contentArea")
        layout = QVBoxLayout(details)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self._build_dashboard_section(), 0)
        layout.addWidget(self._build_files_section(), 1)
        return details

    def _build_dashboard_section(self) -> CollapsibleSection:
        content = QFrame(self)
        content.setObjectName("card")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        self.dashboard_text = QTextEdit(content)
        self.dashboard_text.setReadOnly(True)
        self.dashboard_text.setMinimumHeight(160)
        layout.addWidget(self.dashboard_text)
        return CollapsibleSection(
            "Сводка",
            content,
            self,
            tooltip="Краткое состояние данных перед экспортом.",
            expanded=True,
        )

    def _build_files_section(self) -> CollapsibleSection:
        content = QFrame(self)
        content.setObjectName("card")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        self.files_stack = QStackedLayout()
        self.files_table = QTableView(content)
        self.files_table.setModel(self.files_model)
        self.files_table.verticalHeader().setVisible(False)
        self.files_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.files_table.setMinimumHeight(160)
        self.files_empty = EmptyState(
            "Нет файлов",
            content,
            status="Соберите отчёт",
        )
        self.files_empty.setMinimumHeight(150)
        self.files_stack.addWidget(self.files_table)
        self.files_stack.addWidget(self.files_empty)
        layout.addLayout(self.files_stack)
        return CollapsibleSection(
            "Файлы",
            content,
            self,
            tooltip="Последние созданные файлы отчёта.",
            expanded=True,
        )

    def _build_actions(self) -> QHBoxLayout:  # type: ignore[valid-type]
        row = QHBoxLayout()
        self.folder_button = QPushButton("Папка", self)
        self.folder_button.clicked.connect(self.open_folder)
        row.addWidget(self.folder_button, 0)
        self.refresh_button = QPushButton("Обновить", self)
        self.refresh_button.clicked.connect(self.refresh_data)
        row.addWidget(self.refresh_button, 0)
        row.addStretch(1)
        self.export_button = QPushButton("Собрать отчёт", self)
        self.export_button.setProperty("role", "primary")
        self.export_button.clicked.connect(self.export_selected)
        row.addWidget(self.export_button, 0)
        return row

    def refresh_data(self) -> None:
        self.dashboard_text.setPlainText(self.presenter.dashboard_text())
        self._sync_mode_hint()
        self._sync_files(self.presenter.last_paths())
        self._sync_summary()
        self.status_label.setText("Готово")

    def export_selected(self) -> None:
        mode_id = self.mode_combo.currentData()
        try:
            paths = self.presenter.export(str(mode_id))
        except Exception as exc:  # pragma: no cover - GUI status fallback
            self.status_label.setText("Ошибка")
            self.setToolTip(str(exc))
            return
        self.dashboard_text.setPlainText(self.presenter.dashboard_text())
        self._sync_files(paths)
        self._sync_summary()
        self.status_label.setText("Собрано")

    def open_folder(self) -> None:
        try:
            path = self.presenter.open_folder()
        except Exception as exc:  # pragma: no cover - GUI status fallback
            self.status_label.setText("Ошибка папки")
            self.setToolTip(str(exc))
            return
        self.status_label.setText("Папка открыта")
        self.status_label.setToolTip(str(path))

    def _sync_mode_hint(self) -> None:
        mode_id = str(self.mode_combo.currentData())
        mode = self.presenter.mode_by_id(mode_id)
        self.mode_hint.setText(mode.details)
        self.mode_hint.setToolTip(mode.details)

    def _sync_files(self, paths: dict | None = None) -> None:
        rows = self.presenter.path_rows(paths or {})
        self.files_model.replace_rows(rows)
        self.files_stack.setCurrentWidget(
            self.files_empty if self.files_model.is_empty() else self.files_table,
        )

    def _sync_summary(self) -> None:
        summary = self.presenter.summary()
        self._summary_labels["rows"].setText(str(summary.total_rows))
        self._summary_labels["npv"].setText("есть" if summary.npv_ready else "нет")
        self._summary_labels["files"].setText(str(summary.last_files))
        self._summary_labels["folder"].setText("готова")
        self._summary_labels["folder"].setToolTip(str(summary.report_dir))
