from __future__ import annotations

import codecs

from ui_qt.presenters.catalog_staging_presenter import LocalEnrichmentJobSpec, CatalogStagingPresenter
from ui_qt.widgets import CompactLabel, InfoHint

try:
    from PySide6.QtCore import QProcess
    from PySide6.QtGui import QTextCursor
    from PySide6.QtWidgets import (
        QCheckBox,
        QDialog,
        QHBoxLayout,
        QPlainTextEdit,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )
except ModuleNotFoundError as exc:
    if exc.name != "PySide6":
        raise
    QProcess = None  # type: ignore[assignment]
    QCheckBox = QHBoxLayout = QPlainTextEdit = QPushButton = QVBoxLayout = None  # type: ignore[assignment]
    QDialog = QWidget = object  # type: ignore[assignment,misc]
    QTextCursor = None  # type: ignore[assignment]


class CatalogLocalEnrichmentDialog(QDialog):  # type: ignore[misc,valid-type]
    """Account-free, explicitly low-confidence enrichment in a worker process."""

    def __init__(
        self,
        presenter: CatalogStagingPresenter,
        staging_ids: list[str] | tuple[str, ...],
        parent: QWidget | None = None,  # type: ignore[valid-type]
    ) -> None:
        super().__init__(parent)
        self.presenter = presenter
        self._selected_ids = tuple(str(value) for value in staging_ids if str(value))
        self.job: LocalEnrichmentJobSpec | None = None
        self.completed = False
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._read_output)
        self._process.started.connect(lambda: self.status.setText("Автообогащение идёт"))
        self._process.finished.connect(self._process_finished)
        self._process.errorOccurred.connect(self._process_error)
        self.setWindowTitle("Автономное обогащение")
        self.resize(720, 470)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)
        layout.addWidget(
            InfoHint(
                "Без аккаунтов и сети: извлекает явные признаки из названия/полей, затем может заполнить пробелы консервативными demo-оценками. Оценочные данные помечаются provenance.",
                self,
            )
        )
        self.selected_only = QCheckBox("Только выделенные позиции", self)
        self.selected_only.setChecked(bool(self._selected_ids))
        self.selected_only.setEnabled(bool(self._selected_ids))
        layout.addWidget(self.selected_only)
        self.defaults_checkbox = QCheckBox("Заполнять отсутствующие характеристики", self)
        self.defaults_checkbox.setChecked(True)
        layout.addWidget(self.defaults_checkbox)
        self.price_checkbox = QCheckBox("Оценивать отсутствующую цену", self)
        self.price_checkbox.setChecked(True)
        layout.addWidget(self.price_checkbox)
        self.status = CompactLabel("Готов к запуску", self)
        layout.addWidget(self.status)
        self.log = QPlainTextEdit(self)
        self.log.setReadOnly(True)
        layout.addWidget(self.log, 1)

        actions = QHBoxLayout()
        self.start_button = QPushButton("Запустить", self)
        self.start_button.setProperty("role", "primary")
        self.start_button.clicked.connect(self.start)
        self.cancel_button = QPushButton("Остановить", self)
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._process.terminate)
        self.done_button = QPushButton("Готово", self)
        self.done_button.setEnabled(False)
        self.done_button.clicked.connect(self.accept)
        close_button = QPushButton("Закрыть", self)
        close_button.clicked.connect(self.reject)
        actions.addWidget(self.start_button)
        actions.addWidget(self.cancel_button)
        actions.addStretch(1)
        actions.addWidget(close_button)
        actions.addWidget(self.done_button)
        layout.addLayout(actions)

    def start(self) -> None:
        ids = self._selected_ids if self.selected_only.isChecked() else ()
        self.job = self.presenter.build_local_enrichment_job(
            ids,
            fill_defaults=self.defaults_checkbox.isChecked(),
            estimate_missing_price=self.price_checkbox.isChecked(),
        )
        self.completed = False
        self.done_button.setEnabled(False)
        self.log.clear()
        self._decoder.reset()
        self._set_running(True)
        self._process.setWorkingDirectory(str(self.job.working_directory))
        self._process.start(self.job.program, list(self.job.arguments))

    def _read_output(self, *, final: bool = False) -> None:
        text = self._decoder.decode(bytes(self._process.readAllStandardOutput()), final=final)
        if not text:
            return
        self.log.moveCursor(QTextCursor.MoveOperation.End)
        self.log.insertPlainText(text)
        self.log.ensureCursorVisible()

    def _process_finished(self, exit_code: int, _exit_status) -> None:
        self._read_output(final=True)
        self._set_running(False)
        self.completed = exit_code == 0
        self.status.setText("Обогащение готово" if self.completed else "Обогащение не завершено")
        self.done_button.setEnabled(self.completed)

    def _process_error(self, _error) -> None:
        self._set_running(False)
        details = self._process.errorString()
        if details:
            self.log.appendPlainText(f"Ошибка запуска: {details}")
        self.status.setText("Процесс не запущен")

    def _set_running(self, running: bool) -> None:
        self.start_button.setEnabled(not running)
        self.cancel_button.setEnabled(running)
        self.selected_only.setEnabled(not running and bool(self._selected_ids))
        self.defaults_checkbox.setEnabled(not running)
        self.price_checkbox.setEnabled(not running)

    def reject(self) -> None:
        if self._process.state() != QProcess.ProcessState.NotRunning:
            self._process.terminate()
        super().reject()


__all__ = ["CatalogLocalEnrichmentDialog"]
