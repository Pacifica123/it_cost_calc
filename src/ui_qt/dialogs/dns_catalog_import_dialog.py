from __future__ import annotations

from pathlib import Path

from ui_qt.presenters.catalog_staging_presenter import CatalogStagingPresenter
from ui_qt.widgets import CompactLabel, InfoHint

try:
    from PySide6.QtCore import QProcess, QTimer
    from PySide6.QtGui import QTextCursor
    from PySide6.QtWidgets import (
        QCheckBox,
        QDialog,
        QGridLayout,
        QHBoxLayout,
        QLineEdit,
        QMessageBox,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )
except ModuleNotFoundError as exc:
    if exc.name != "PySide6":
        raise
    QProcess = QTimer = None  # type: ignore[assignment]
    QTextCursor = None  # type: ignore[assignment]
    QCheckBox = QGridLayout = QHBoxLayout = QLineEdit = QMessageBox = None  # type: ignore[assignment]
    QPlainTextEdit = QProgressBar = QPushButton = QSpinBox = QVBoxLayout = None  # type: ignore[assignment]
    QDialog = QWidget = object  # type: ignore[assignment,misc]


class DnsCatalogImportDialog(QDialog):  # type: ignore[misc,valid-type]
    """Run the standalone DNS parser process and expose its log to the user."""

    def __init__(
        self,
        presenter: CatalogStagingPresenter,
        parent: QWidget | None = None,  # type: ignore[valid-type]
    ) -> None:
        if QProcess is None:
            raise RuntimeError("PySide6 is required to create DnsCatalogImportDialog")
        super().__init__(parent)
        self.presenter = presenter
        self.catalog_path: Path | None = None
        self._job = None
        self._close_after_stop = False
        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._read_output)
        self._process.started.connect(self._process_started)
        self._process.finished.connect(self._process_finished)
        self._process.errorOccurred.connect(self._process_error)
        self.setWindowTitle("Сбор каталога DNS")
        self.resize(760, 620)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)
        layout.addWidget(
            InfoHint(
                "Откроется отдельный Chromium. На первой странице можно выбрать регион, принять cookies "
                "или пройти проверку. Сбор ограничен лимитом и сохраняет исходные HTML.",
                self,
            )
        )

        grid = QGridLayout()
        grid.addWidget(CompactLabel("Категории", self), 0, 0)
        category_row = QHBoxLayout()
        self._category_checks: dict[str, QCheckBox] = {}
        for value, label in self.presenter.dns_category_options():
            checkbox = QCheckBox(label, self)
            checkbox.setChecked(True)
            self._category_checks[value] = checkbox
            category_row.addWidget(checkbox)
        category_row.addStretch(1)
        grid.addLayout(category_row, 0, 1, 1, 3)

        grid.addWidget(CompactLabel("Карточек на категорию", self), 1, 0)
        self.limit_spin = QSpinBox(self)
        self.limit_spin.setRange(1, 50)
        self.limit_spin.setValue(10)
        grid.addWidget(self.limit_spin, 1, 1)

        grid.addWidget(CompactLabel("Общий таймаут, сек", self), 1, 2)
        self.timeout_spin = QSpinBox(self)
        self.timeout_spin.setRange(30, 1800)
        self.timeout_spin.setValue(300)
        grid.addWidget(self.timeout_spin, 1, 3)

        grid.addWidget(CompactLabel("Регион цены", self), 2, 0)
        self.region_edit = QLineEdit(self)
        self.region_edit.setPlaceholderText("например, Москва")
        grid.addWidget(self.region_edit, 2, 1)
        self.visible_browser = QCheckBox("Показывать браузер", self)
        self.visible_browser.setChecked(True)
        grid.addWidget(self.visible_browser, 2, 2, 1, 2)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        layout.addLayout(grid)

        self.progress = QProgressBar(self)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        self.status = CompactLabel("Готов к запуску", self)
        layout.addWidget(self.status)
        self.log = QPlainTextEdit(self)
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Здесь появится журнал сбора.")
        layout.addWidget(self.log, 1)

        buttons = QHBoxLayout()
        self.start_button = QPushButton("Начать сбор", self)
        self.start_button.setProperty("role", "primary")
        self.start_button.clicked.connect(self.start_collection)
        self.cancel_button = QPushButton("Остановить", self)
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.stop_collection)
        self.load_button = QPushButton("Загрузить в staging", self)
        self.load_button.setProperty("role", "primary")
        self.load_button.setEnabled(False)
        self.load_button.clicked.connect(self.accept)
        close_button = QPushButton("Закрыть", self)
        close_button.clicked.connect(self.reject)
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.cancel_button)
        buttons.addStretch(1)
        buttons.addWidget(close_button)
        buttons.addWidget(self.load_button)
        layout.addLayout(buttons)

    def start_collection(self) -> None:
        categories = [
            value for value, checkbox in self._category_checks.items() if checkbox.isChecked()
        ]
        try:
            self._job = self.presenter.build_dns_job(
                categories=categories,
                per_category_limit=self.limit_spin.value(),
                time_limit_seconds=self.timeout_spin.value(),
                visible_browser=self.visible_browser.isChecked(),
                region=self.region_edit.text(),
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Параметры DNS", str(exc))
            return
        self.catalog_path = None
        self.load_button.setEnabled(False)
        self.log.clear()
        self._set_controls_running(True)
        self._process.setWorkingDirectory(str(self._job.working_directory))
        self._process.start(self._job.program, list(self._job.arguments))

    def stop_collection(self) -> None:
        if self._process.state() == QProcess.ProcessState.NotRunning:
            return
        self.status.setText("Остановка процесса…")
        self._process.terminate()
        QTimer.singleShot(3000, self._kill_if_running)

    def _kill_if_running(self) -> None:
        if self._process.state() != QProcess.ProcessState.NotRunning:
            self._process.kill()

    def _process_started(self) -> None:
        self.progress.setRange(0, 0)
        self.status.setText("Сбор выполняется")

    def _read_output(self) -> None:
        text = bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if text:
            self.log.moveCursor(QTextCursor.MoveOperation.End)
            self.log.insertPlainText(text)
            self.log.ensureCursorVisible()

    def _process_finished(self, exit_code: int, _exit_status) -> None:
        self._read_output()
        self._set_controls_running(False)
        self.progress.setRange(0, 1)
        self.progress.setValue(1 if exit_code == 0 else 0)
        output_path = self._job.output_path if self._job is not None else None
        if exit_code == 0 and output_path is not None and output_path.exists():
            self.catalog_path = output_path
            self.load_button.setEnabled(True)
            self.status.setText(f"Каталог готов: {output_path}")
        else:
            self.status.setText(f"Сбор завершился с ошибкой, код {exit_code}")
        if self._close_after_stop:
            self._close_after_stop = False
            QDialog.reject(self)

    def _process_error(self, _error) -> None:
        self._set_controls_running(False)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.status.setText(f"Не удалось запустить процесс: {self._process.errorString()}")

    def _set_controls_running(self, running: bool) -> None:
        self.start_button.setEnabled(not running)
        self.cancel_button.setEnabled(running)
        for checkbox in self._category_checks.values():
            checkbox.setEnabled(not running)
        self.limit_spin.setEnabled(not running)
        self.timeout_spin.setEnabled(not running)
        self.region_edit.setEnabled(not running)
        self.visible_browser.setEnabled(not running)

    def reject(self) -> None:
        if self._process.state() != QProcess.ProcessState.NotRunning:
            self._close_after_stop = True
            self.stop_collection()
            return
        super().reject()


__all__ = ["DnsCatalogImportDialog"]
