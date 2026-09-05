from __future__ import annotations

from ui_qt.presenters.catalog_staging_presenter import CatalogStagingPresenter
from ui_qt.widgets import CompactLabel, InfoHint

try:
    from PySide6.QtWidgets import (
        QApplication,
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
    QApplication = QDialog = QFileDialog = QGridLayout = QHBoxLayout = None  # type: ignore[assignment]
    QLineEdit = QMessageBox = QPlainTextEdit = QPushButton = QVBoxLayout = None  # type: ignore[assignment]
    QWidget = object  # type: ignore[assignment,misc]


class CatalogBrowserCaptureDialog(QDialog):  # type: ignore[misc,valid-type]
    """Import one product from an ordinary user browser, without automation."""

    def __init__(
        self,
        presenter: CatalogStagingPresenter,
        parent: QWidget | None = None,  # type: ignore[valid-type]
    ) -> None:
        super().__init__(parent)
        self.presenter = presenter
        self.completed = False
        self._file_path = ""
        self.setWindowTitle("Единичный захват из браузера")
        self.resize(760, 520)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)
        layout.addWidget(
            InfoHint(
                "Откройте карточку товара в обычном браузере. Затем сохраните HTML/исходный код "
                "или скопируйте JSON-LD в буфер. Приложение ничего не запрашивает у сайта и не "
                "управляет браузером.",
                self,
            )
        )

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        grid.addWidget(CompactLabel("URL карточки", self), 0, 0)
        self.url_edit = QLineEdit(self)
        self.url_edit.setPlaceholderText("https://shop.example/product/... (не открывается приложением)")
        grid.addWidget(self.url_edit, 0, 1, 1, 3)
        grid.addWidget(CompactLabel("Категория", self), 1, 0)
        self.category_edit = QLineEdit(self)
        self.category_edit.setPlaceholderText("router / switch / workstation / server")
        grid.addWidget(self.category_edit, 1, 1)
        grid.addWidget(CompactLabel("Регион", self), 1, 2)
        self.region_edit = QLineEdit(self)
        grid.addWidget(self.region_edit, 1, 3)
        for column in (1, 3):
            grid.setColumnStretch(column, 1)
        layout.addLayout(grid)

        self.input_edit = QPlainTextEdit(self)
        self.input_edit.setPlaceholderText(
            "Вставьте JSON-LD Product или HTML. Либо выберите HTML-файл."
        )
        layout.addWidget(self.input_edit, 1)

        input_actions = QHBoxLayout()
        file_button = QPushButton("HTML-файл", self)
        file_button.clicked.connect(self._browse)
        clipboard_button = QPushButton("Из буфера", self)
        clipboard_button.clicked.connect(self._paste_clipboard)
        self.file_label = CompactLabel("Файл не выбран", self)
        input_actions.addWidget(file_button)
        input_actions.addWidget(clipboard_button)
        input_actions.addWidget(self.file_label)
        input_actions.addStretch(1)
        layout.addLayout(input_actions)

        actions = QHBoxLayout()
        actions.addStretch(1)
        close_button = QPushButton("Закрыть", self)
        close_button.clicked.connect(self.reject)
        capture_button = QPushButton("Добавить в staging", self)
        capture_button.setProperty("role", "primary")
        capture_button.clicked.connect(self._capture)
        actions.addWidget(close_button)
        actions.addWidget(capture_button)
        layout.addLayout(actions)

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Сохранённая карточка",
            str(self.presenter.app_presenter.paths.repo_root),
            "Захват (*.html *.htm *.json *.txt)",
        )
        if path:
            self._file_path = path
            self.file_label.setText("Файл выбран")

    def _paste_clipboard(self) -> None:
        text = QApplication.clipboard().text()
        self.input_edit.setPlainText(text)
        self._file_path = ""
        self.file_label.setText("Буфер выбран")

    def _capture(self) -> None:
        try:
            captured = self.presenter.stage_browser_capture(
                path=self._file_path or None,
                content=None if self._file_path else self.input_edit.toPlainText(),
                source_url=self.url_edit.text(),
                region=self.region_edit.text(),
                category_override=self.category_edit.text(),
            )
        except Exception as exc:
            QMessageBox.warning(self, "Browser capture", str(exc))
            return
        self.completed = True
        message = f"Добавлено: {captured.item.get('title')}."
        if captured.warnings:
            message += " Проверьте предупреждения в staging."
        QMessageBox.information(self, "Browser capture", message)
        self.accept()


__all__ = ["CatalogBrowserCaptureDialog"]
