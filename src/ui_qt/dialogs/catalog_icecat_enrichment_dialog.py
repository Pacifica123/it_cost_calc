from __future__ import annotations

import codecs
import os

from ui_qt.presenters.catalog_staging_presenter import (
    CatalogStagingPresenter,
    IcecatEnrichmentJobSpec,
)
from ui_qt.widgets import CompactLabel, InfoHint

try:
    from PySide6.QtCore import QProcess, QProcessEnvironment
    from PySide6.QtGui import QTextCursor
    from PySide6.QtWidgets import (
        QDialog,
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
    QProcess = QProcessEnvironment = None  # type: ignore[assignment]
    QGridLayout = QHBoxLayout = QLineEdit = QMessageBox = None  # type: ignore[assignment]
    QPlainTextEdit = QPushButton = QVBoxLayout = None  # type: ignore[assignment]
    QDialog = QWidget = object  # type: ignore[assignment,misc]
    QTextCursor = None  # type: ignore[assignment]


class CatalogIcecatEnrichmentDialog(QDialog):  # type: ignore[misc,valid-type]
    """Run P3 specification enrichment without persisting credentials."""

    def __init__(
        self,
        presenter: CatalogStagingPresenter,
        staging_ids: list[str] | None = None,
        parent: QWidget | None = None,  # type: ignore[valid-type]
    ) -> None:
        super().__init__(parent)
        self.presenter = presenter
        self.staging_ids = list(staging_ids or [])
        self.job: IcecatEnrichmentJobSpec | None = None
        self.completed = False
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._read_output)
        self._process.started.connect(self._process_started)
        self._process.finished.connect(self._process_finished)
        self._process.errorOccurred.connect(self._process_error)

        self.setWindowTitle("Обогащение характеристик Icecat")
        self.resize(760, 520)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)
        layout.addWidget(
            InfoHint(
                "Icecat заполняет только отсутствующие характеристики по GTIN или бренд+MPN. "
                "Цена остаётся из российских feed-источников. API-токен используется только "
                "дочерним процессом и не сохраняется в проекте или manifest.",
                self,
            )
        )

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        grid.addWidget(CompactLabel("Логин Icecat", self), 0, 0)
        self.username_edit = QLineEdit(self)
        self.username_edit.setText(os.environ.get("ICECAT_USERNAME", ""))
        self.username_edit.setPlaceholderText("логин Open Icecat")
        grid.addWidget(self.username_edit, 0, 1)

        grid.addWidget(CompactLabel("API-токен", self), 1, 0)
        self.token_edit = QLineEdit(self)
        self.token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.token_edit.setPlaceholderText("не сохраняется")
        grid.addWidget(self.token_edit, 1, 1)
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)

        scope = (
            f"Выбрано позиций: {len(self.staging_ids)}"
            if self.staging_ids
            else "Область: все доступные позиции"
        )
        self.scope_label = CompactLabel(scope, self)
        layout.addWidget(self.scope_label)

        self.status = CompactLabel("Готов к enrichment", self)
        layout.addWidget(self.status)
        self.log = QPlainTextEdit(self)
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Здесь появится журнал Icecat enrichment.")
        layout.addWidget(self.log, 1)

        actions = QHBoxLayout()
        self.start_button = QPushButton("Обогатить", self)
        self.start_button.setProperty("role", "primary")
        self.start_button.clicked.connect(self.start_enrichment)
        self.cancel_button = QPushButton("Остановить", self)
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._process.terminate)
        self.done_button = QPushButton("Готово", self)
        self.done_button.setProperty("role", "primary")
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

    def start_enrichment(self) -> None:
        try:
            self.job = self.presenter.build_icecat_enrichment_job(
                self.staging_ids,
                username=self.username_edit.text(),
                language="EN",
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Icecat", str(exc))
            return

        self.log.clear()
        self._decoder.reset()
        self.completed = False
        self.done_button.setEnabled(False)
        self._set_running(True)
        environment = QProcessEnvironment.systemEnvironment()
        token = self.token_edit.text().strip()
        if token:
            environment.insert("ICECAT_API_TOKEN", token)
        environment.insert("ICECAT_USERNAME", self.username_edit.text().strip())
        self._process.setProcessEnvironment(environment)
        self._process.setWorkingDirectory(str(self.job.working_directory))
        self._process.start(self.job.program, list(self.job.arguments))

    def _process_started(self) -> None:
        self.status.setText("Icecat выполняется")

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
        if exit_code == 0:
            self.completed = True
            self.status.setText("Enrichment завершён")
            self.done_button.setEnabled(True)
        else:
            self.status.setText("Enrichment не завершён")

    def _process_error(self, _error) -> None:
        self._set_running(False)
        details = self._process.errorString()
        if details:
            self.log.appendPlainText(f"Ошибка запуска: {details}")
        self.status.setText("Процесс не запущен")

    def _set_running(self, running: bool) -> None:
        self.start_button.setEnabled(not running)
        self.cancel_button.setEnabled(running)
        self.username_edit.setEnabled(not running)
        self.token_edit.setEnabled(not running)

    def reject(self) -> None:
        if self._process.state() != QProcess.ProcessState.NotRunning:
            self._process.terminate()
        super().reject()


__all__ = ["CatalogIcecatEnrichmentDialog"]
