from __future__ import annotations

import codecs
from pathlib import Path

from ui_qt.presenters.catalog_staging_presenter import CatalogStageJobSpec, CatalogStagingPresenter
from ui_qt.widgets import CompactLabel, InfoHint

try:
    from PySide6.QtCore import QProcess
    from PySide6.QtGui import QTextCursor
    from PySide6.QtWidgets import (
        QComboBox,
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
    QComboBox = QHBoxLayout = QPlainTextEdit = QPushButton = QVBoxLayout = None  # type: ignore[assignment]
    QDialog = QWidget = object  # type: ignore[assignment,misc]
    QTextCursor = None  # type: ignore[assignment]


class CatalogStageFileDialog(QDialog):  # type: ignore[misc,valid-type]
    """Run heavy catalog staging outside the GUI process."""

    def __init__(
        self,
        presenter: CatalogStagingPresenter,
        source_path: str | Path,
        parent: QWidget | None = None,  # type: ignore[valid-type]
    ) -> None:
        super().__init__(parent)
        self.presenter = presenter
        self.source_path = Path(source_path)
        self.job: CatalogStageJobSpec | None = None
        self.completed = False
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._read_output)
        self._process.started.connect(lambda: self.status.setText("Staging выполняется"))
        self._process.finished.connect(self._process_finished)
        self._process.errorOccurred.connect(self._process_error)
        self.setWindowTitle("Импорт большого каталога")
        self.resize(700, 430)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)
        layout.addWidget(
            InfoHint(
                "Обработка выполняется отдельным процессом. Таблица GUI показывает страницы, а расчёт сохраняет весь каталог.",
                self,
            )
        )
        layout.addWidget(CompactLabel(self.source_path.name[:80] or "Каталог", self))
        self.limit_combo = QComboBox(self)
        for label, value in (
            ("Все строки", 0),
            ("1 000 строк", 1000),
            ("5 000 строк", 5000),
            ("10 000 строк", 10000),
            ("25 000 строк", 25000),
        ):
            self.limit_combo.addItem(label, value)
        self.limit_combo.setToolTip("Для полного расчёта оставьте «Все строки».")
        layout.addWidget(self.limit_combo)
        self.status = CompactLabel("Готов к импорту", self)
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
        self.job = self.presenter.build_stage_job(
            self.source_path,
            max_rows=int(self.limit_combo.currentData() or 0),
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
        self.status.setText("Staging готов" if self.completed else "Staging не завершён")
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
        self.limit_combo.setEnabled(not running)

    def reject(self) -> None:
        if self._process.state() != QProcess.ProcessState.NotRunning:
            self._process.terminate()
        super().reject()


__all__ = ["CatalogStageFileDialog"]
