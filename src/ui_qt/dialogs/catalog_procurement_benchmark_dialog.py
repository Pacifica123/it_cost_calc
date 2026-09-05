from __future__ import annotations

import codecs

from ui_qt.presenters.catalog_staging_presenter import (
    CatalogStagingPresenter,
    ProcurementBenchmarkJobSpec,
)
from ui_qt.widgets import CompactLabel, InfoHint

try:
    from PySide6.QtCore import QProcess
    from PySide6.QtGui import QTextCursor
    from PySide6.QtWidgets import (
        QDialog,
        QFileDialog,
        QGridLayout,
        QHBoxLayout,
        QLineEdit,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )
except ModuleNotFoundError as exc:
    if exc.name != "PySide6":
        raise
    QProcess = None  # type: ignore[assignment]
    QFileDialog = QGridLayout = QHBoxLayout = QLineEdit = QMessageBox = None  # type: ignore[assignment]
    QPlainTextEdit = QPushButton = QSpinBox = QVBoxLayout = None  # type: ignore[assignment]
    QDialog = QWidget = object  # type: ignore[assignment,misc]
    QTextCursor = None  # type: ignore[assignment]


class CatalogProcurementBenchmarkDialog(QDialog):  # type: ignore[misc,valid-type]
    """Apply EIS/open-procurement contract prices as independent statistics."""

    def __init__(
        self,
        presenter: CatalogStagingPresenter,
        staging_ids: list[str] | None = None,
        parent: QWidget | None = None,  # type: ignore[valid-type]
    ) -> None:
        super().__init__(parent)
        self.presenter = presenter
        self.staging_ids = list(staging_ids or [])
        self.job: ProcurementBenchmarkJobSpec | None = None
        self.completed = False
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._read_output)
        self._process.started.connect(self._process_started)
        self._process.finished.connect(self._process_finished)
        self._process.errorOccurred.connect(self._process_error)

        self.setWindowTitle("Бенчмарк закупок ЕИС")
        self.resize(780, 560)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)
        layout.addWidget(
            InfoHint(
                "ЕИС используется только как статистический benchmark заключённых закупок. "
                "Он не становится магазином и не заменяет текущую цену расчёта. "
                "Поддерживаются локальные XML/ZIP/JSON/CSV и прямые URL выгрузок.",
                self,
            )
        )

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        grid.addWidget(CompactLabel("Файл или URL", self), 0, 0)
        self.location_edit = QLineEdit(self)
        self.location_edit.setPlaceholderText("/path/contracts.zip или https://.../export.xml")
        grid.addWidget(self.location_edit, 0, 1)
        browse = QPushButton("Файл", self)
        browse.clicked.connect(self._browse)
        grid.addWidget(browse, 0, 2)

        grid.addWidget(CompactLabel("Регион", self), 1, 0)
        self.region_edit = QLineEdit(self)
        self.region_edit.setPlaceholderText("опционально")
        grid.addWidget(self.region_edit, 1, 1)

        grid.addWidget(CompactLabel("Лимит строк", self), 1, 2)
        self.limit_spin = QSpinBox(self)
        self.limit_spin.setRange(100, 200000)
        self.limit_spin.setValue(20000)
        self.limit_spin.setSingleStep(1000)
        grid.addWidget(self.limit_spin, 1, 3)
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)

        scope = (
            f"Выбрано позиций: {len(self.staging_ids)}"
            if self.staging_ids
            else "Область: весь staging"
        )
        layout.addWidget(CompactLabel(scope, self))
        self.status = CompactLabel("Готов к анализу", self)
        layout.addWidget(self.status)
        self.log = QPlainTextEdit(self)
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Здесь появится журнал ЕИС benchmark.")
        layout.addWidget(self.log, 1)

        actions = QHBoxLayout()
        self.start_button = QPushButton("Рассчитать benchmark", self)
        self.start_button.setProperty("role", "primary")
        self.start_button.clicked.connect(self.start_benchmark)
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

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Выгрузка ЕИС",
            str(self.presenter.app_presenter.paths.repo_root),
            "ЕИС (*.zip *.xml *.json *.csv)",
        )
        if path:
            self.location_edit.setText(path)

    def start_benchmark(self) -> None:
        try:
            self.job = self.presenter.build_procurement_benchmark_job(
                self.location_edit.text(),
                self.staging_ids,
                region=self.region_edit.text(),
                max_records=self.limit_spin.value(),
            )
        except ValueError as exc:
            QMessageBox.warning(self, "ЕИС benchmark", str(exc))
            return
        self.log.clear()
        self._decoder.reset()
        self.completed = False
        self.done_button.setEnabled(False)
        self._set_running(True)
        self._process.setWorkingDirectory(str(self.job.working_directory))
        self._process.start(self.job.program, list(self.job.arguments))

    def _process_started(self) -> None:
        self.status.setText("ЕИС анализируется")

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
            self.status.setText("Benchmark рассчитан")
            self.done_button.setEnabled(True)
        else:
            self.status.setText("Benchmark не рассчитан")

    def _process_error(self, _error) -> None:
        self._set_running(False)
        details = self._process.errorString()
        if details:
            self.log.appendPlainText(f"Ошибка запуска: {details}")
        self.status.setText("Процесс не запущен")

    def _set_running(self, running: bool) -> None:
        self.start_button.setEnabled(not running)
        self.cancel_button.setEnabled(running)
        self.location_edit.setEnabled(not running)
        self.region_edit.setEnabled(not running)
        self.limit_spin.setEnabled(not running)

    def reject(self) -> None:
        if self._process.state() != QProcess.ProcessState.NotRunning:
            self._process.terminate()
        super().reject()


__all__ = ["CatalogProcurementBenchmarkDialog"]
