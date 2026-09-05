from __future__ import annotations

import codecs
from pathlib import Path
from typing import Any

from ui_qt.presenters.catalog_staging_presenter import CatalogFeedJobSpec, CatalogStagingPresenter
from ui_qt.widgets import CompactLabel, InfoHint

try:
    from PySide6.QtCore import QProcess
    from PySide6.QtGui import QTextCursor
    from PySide6.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDialog,
        QFileDialog,
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
    QProcess = None  # type: ignore[assignment]
    QCheckBox = QComboBox = QFileDialog = QGridLayout = QHBoxLayout = None  # type: ignore[assignment]
    QLineEdit = QMessageBox = QPlainTextEdit = QPushButton = QVBoxLayout = None  # type: ignore[assignment]
    QDialog = QWidget = object  # type: ignore[assignment,misc]
    QTextCursor = None  # type: ignore[assignment]


class CatalogFeedSourceDialog(QDialog):  # type: ignore[misc,valid-type]
    """Fetch a structured supplier feed without browser automation."""

    def __init__(
        self,
        presenter: CatalogStagingPresenter,
        parent: QWidget | None = None,  # type: ignore[valid-type]
    ) -> None:
        super().__init__(parent)
        self.presenter = presenter
        self.job: CatalogFeedJobSpec | None = None
        self._sources = presenter.catalog_sources()
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._read_output)
        self._process.started.connect(self._process_started)
        self._process.finished.connect(self._process_finished)
        self._process.errorOccurred.connect(self._process_error)

        self.setWindowTitle("Структурированный источник каталога")
        self.resize(760, 560)
        self._build_ui()
        self._fill_preset(0)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)
        layout.addWidget(
            InfoHint(
                "Источник должен отдавать XLSX, CSV или YML/XML. Браузер и Playwright не используются.",
                self,
            )
        )

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)

        grid.addWidget(CompactLabel("Шаблон", self), 0, 0)
        self.preset_combo = QComboBox(self)
        for source in self._sources:
            self.preset_combo.addItem(source["name"], source["id"])
        self.preset_combo.addItem("Новый источник", "__custom__")
        self.preset_combo.currentIndexChanged.connect(self._fill_preset)
        grid.addWidget(self.preset_combo, 0, 1, 1, 3)

        grid.addWidget(CompactLabel("Название", self), 1, 0)
        self.name_edit = QLineEdit(self)
        grid.addWidget(self.name_edit, 1, 1)
        grid.addWidget(CompactLabel("ID", self), 1, 2)
        self.id_edit = QLineEdit(self)
        grid.addWidget(self.id_edit, 1, 3)

        grid.addWidget(CompactLabel("URL или файл", self), 2, 0)
        self.location_edit = QLineEdit(self)
        self.location_edit.setPlaceholderText("https://.../price.xlsx или /path/catalog.yml")
        grid.addWidget(self.location_edit, 2, 1, 1, 2)
        self.browse_button = QPushButton("Файл", self)
        self.browse_button.clicked.connect(self._browse_file)
        grid.addWidget(self.browse_button, 2, 3)

        grid.addWidget(CompactLabel("Формат", self), 3, 0)
        self.format_combo = QComboBox(self)
        for value, label in (
            ("xlsx", "XLSX"),
            ("csv", "CSV"),
            ("yml", "YML"),
            ("xml", "XML"),
            ("auto", "Авто"),
        ):
            self.format_combo.addItem(label, value)
        grid.addWidget(self.format_combo, 3, 1)
        grid.addWidget(CompactLabel("Регион", self), 3, 2)
        self.region_edit = QLineEdit(self)
        grid.addWidget(self.region_edit, 3, 3)
        grid.addWidget(CompactLabel("Объём", self), 4, 0)
        self.limit_combo = QComboBox(self)
        for label, value in (
            ("Все строки", 0),
            ("1 000 строк", 1000),
            ("5 000 строк", 5000),
            ("10 000 строк", 10000),
            ("25 000 строк", 25000),
        ):
            self.limit_combo.addItem(label, value)
        self.limit_combo.setToolTip(
            "Лимит ускоряет пробную загрузку. Значение «Все строки» сохраняет полный набор альтернатив."
        )
        grid.addWidget(self.limit_combo, 4, 1)
        for column in (1, 3):
            grid.setColumnStretch(column, 1)
        layout.addLayout(grid)

        self.notes = InfoHint("Выберите источник или задайте свой URL.", self)
        layout.addWidget(self.notes)
        self.save_checkbox = QCheckBox("Сохранить пользовательский источник", self)
        layout.addWidget(self.save_checkbox)

        self.status = CompactLabel("Готов к загрузке", self)
        layout.addWidget(self.status)
        self.log = QPlainTextEdit(self)
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Здесь появится журнал загрузки feed.")
        layout.addWidget(self.log, 1)

        actions = QHBoxLayout()
        self.start_button = QPushButton("Загрузить feed", self)
        self.start_button.setProperty("role", "primary")
        self.start_button.clicked.connect(self.start_fetch)
        self.cancel_button = QPushButton("Остановить", self)
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._process.terminate)
        self.stage_button = QPushButton("Готово", self)
        self.stage_button.setProperty("role", "primary")
        self.stage_button.setEnabled(False)
        self.stage_button.clicked.connect(self.accept)
        close_button = QPushButton("Закрыть", self)
        close_button.clicked.connect(self.reject)
        actions.addWidget(self.start_button)
        actions.addWidget(self.cancel_button)
        actions.addStretch(1)
        actions.addWidget(close_button)
        actions.addWidget(self.stage_button)
        layout.addLayout(actions)

    def source_values(self) -> dict[str, Any]:
        current = self._current_preset()
        strategy = str(current.get("download_strategy") or "direct") if current else "direct"
        return {
            "id": self.id_edit.text(),
            "name": self.name_edit.text(),
            "location": self.location_edit.text(),
            "format": str(self.format_combo.currentData() or "auto"),
            "region": self.region_edit.text(),
            "price_kind": str(current.get("price_kind") or "supplier_price") if current else "supplier_price",
            "download_strategy": strategy,
            "homepage": str(current.get("homepage") or "") if current else "",
            "notes": str(current.get("notes") or "") if current else "",
            "preset": bool(current and current.get("preset")),
        }

    def start_fetch(self) -> None:
        try:
            values = self.source_values()
            self.job = self.presenter.build_feed_job(
                values,
                max_rows=int(self.limit_combo.currentData() or 0),
            )
            if self.save_checkbox.isChecked() and not values.get("preset"):
                self.presenter.save_catalog_source(values)
        except ValueError as exc:
            QMessageBox.warning(self, "Источник каталога", str(exc))
            return

        self.log.clear()
        self._decoder.reset()
        self.stage_button.setEnabled(False)
        self._set_running(True)
        self._process.setWorkingDirectory(str(self.job.working_directory))
        self._process.start(self.job.program, list(self.job.arguments))

    def _process_started(self) -> None:
        self.status.setText("Feed загружается")

    def _read_output(self, *, final: bool = False) -> None:
        output = bytes(self._process.readAllStandardOutput())
        text = self._decoder.decode(output, final=final)
        if not text:
            return
        self.log.moveCursor(QTextCursor.MoveOperation.End)
        self.log.insertPlainText(text)
        self.log.ensureCursorVisible()

    def _process_finished(self, exit_code: int, _exit_status) -> None:
        self._read_output(final=True)
        self._set_running(False)
        if exit_code == 0 and self.job is not None and self.job.output_path.is_file():
            self.status.setText("Feed и staging готовы")
            self.stage_button.setEnabled(True)
        else:
            self.status.setText("Feed не загружен")

    def _process_error(self, _error) -> None:
        self._set_running(False)
        details = self._process.errorString()
        if details:
            self.log.appendPlainText(f"Ошибка запуска: {details}")
        self.status.setText("Процесс не запущен")

    def _set_running(self, running: bool) -> None:
        self.start_button.setEnabled(not running)
        self.cancel_button.setEnabled(running)
        self.preset_combo.setEnabled(not running)
        self.name_edit.setEnabled(not running)
        self.id_edit.setEnabled(not running)
        self.location_edit.setEnabled(not running)
        self.browse_button.setEnabled(not running)
        self.format_combo.setEnabled(not running)
        self.region_edit.setEnabled(not running)
        self.limit_combo.setEnabled(not running)
        self.save_checkbox.setEnabled(not running)

    def _browse_file(self) -> None:
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Выберите структурированный каталог",
            str(self.presenter.app_presenter.paths.repo_root),
            "Feed (*.xlsx *.csv *.yml *.xml)",
        )
        if not path:
            return
        self.location_edit.setText(path)
        suffix = Path(path).suffix.lower().lstrip(".")
        index = self.format_combo.findData(suffix)
        if index >= 0:
            self.format_combo.setCurrentIndex(index)

    def _current_preset(self) -> dict[str, Any] | None:
        source_id = str(self.preset_combo.currentData() or "")
        if source_id == "__custom__":
            return None
        return next((source for source in self._sources if source["id"] == source_id), None)

    def _fill_preset(self, _index: int) -> None:
        source = self._current_preset()
        if source is None:
            self.name_edit.clear()
            self.id_edit.clear()
            self.location_edit.clear()
            self.region_edit.clear()
            self.format_combo.setCurrentIndex(0)
            self.notes.setToolTip("Задайте URL или локальный файл структурированного feed.")
            self.save_checkbox.setChecked(True)
            self.save_checkbox.setEnabled(True)
            return
        self.name_edit.setText(source["name"])
        self.id_edit.setText(source["id"])
        self.location_edit.setText(source["location"])
        self.region_edit.setText(source.get("region", ""))
        index = self.format_combo.findData(source.get("format", "auto"))
        self.format_combo.setCurrentIndex(index if index >= 0 else 0)
        self.notes.setToolTip(source.get("notes") or "Структурированный источник каталога.")
        self.save_checkbox.setChecked(False)
        self.save_checkbox.setEnabled(False)

    def reject(self) -> None:
        if self._process.state() != QProcess.ProcessState.NotRunning:
            self._process.terminate()
        super().reject()


__all__ = ["CatalogFeedSourceDialog"]
