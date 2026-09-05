from __future__ import annotations

from datetime import date

from ui_qt.presenters.catalog_staging_presenter import CatalogStagingPresenter
from ui_qt.widgets import CompactLabel, InfoHint

try:
    from PySide6.QtWidgets import (
        QCheckBox,
        QDialog,
        QFileDialog,
        QGridLayout,
        QHBoxLayout,
        QLineEdit,
        QMessageBox,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )
except ModuleNotFoundError as exc:
    if exc.name != "PySide6":
        raise
    QCheckBox = QDialog = QFileDialog = QGridLayout = QHBoxLayout = None  # type: ignore[assignment]
    QLineEdit = QMessageBox = QPushButton = QVBoxLayout = None  # type: ignore[assignment]
    QWidget = object  # type: ignore[assignment,misc]


class CatalogCommercialQuoteDialog(QDialog):  # type: ignore[misc,valid-type]
    """Import a supplier quote as ``commercial_quote`` price observations."""

    def __init__(
        self,
        presenter: CatalogStagingPresenter,
        parent: QWidget | None = None,  # type: ignore[valid-type]
    ) -> None:
        super().__init__(parent)
        self.presenter = presenter
        self.completed = False
        self.setWindowTitle("Импорт коммерческого предложения")
        self.resize(680, 360)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)
        layout.addWidget(
            InfoHint(
                "КП импортируется как самый доверенный тип ценового наблюдения. "
                "Поддерживается тот же XLSX/CSV/JSON/YML/XML mapping, что и staging; "
                "GTIN/MPN позволяют объединить КП с существующим товаром.",
                self,
            )
        )

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        grid.addWidget(CompactLabel("Файл КП", self), 0, 0)
        self.path_edit = QLineEdit(self)
        grid.addWidget(self.path_edit, 0, 1, 1, 2)
        browse = QPushButton("Файл", self)
        browse.clicked.connect(self._browse)
        grid.addWidget(browse, 0, 3)

        grid.addWidget(CompactLabel("Поставщик", self), 1, 0)
        self.supplier_edit = QLineEdit(self)
        self.supplier_edit.setPlaceholderText("ООО Поставщик")
        grid.addWidget(self.supplier_edit, 1, 1)
        grid.addWidget(CompactLabel("Номер КП", self), 1, 2)
        self.number_edit = QLineEdit(self)
        grid.addWidget(self.number_edit, 1, 3)

        grid.addWidget(CompactLabel("Дата КП", self), 2, 0)
        self.date_edit = QLineEdit(self)
        self.date_edit.setText(date.today().isoformat())
        grid.addWidget(self.date_edit, 2, 1)
        grid.addWidget(CompactLabel("Регион", self), 2, 2)
        self.region_edit = QLineEdit(self)
        grid.addWidget(self.region_edit, 2, 3)
        for column in (1, 3):
            grid.setColumnStretch(column, 1)
        layout.addLayout(grid)

        self.available_check = QCheckBox("Считать позиции КП доступными", self)
        self.available_check.setChecked(True)
        layout.addWidget(self.available_check)
        layout.addStretch(1)

        actions = QHBoxLayout()
        actions.addStretch(1)
        cancel = QPushButton("Отмена", self)
        cancel.clicked.connect(self.reject)
        import_button = QPushButton("Импортировать КП", self)
        import_button.setProperty("role", "primary")
        import_button.clicked.connect(self._import_quote)
        actions.addWidget(cancel)
        actions.addWidget(import_button)
        layout.addLayout(actions)

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Коммерческое предложение",
            str(self.presenter.app_presenter.paths.repo_root),
            "Каталог (*.xlsx *.csv *.json *.yml *.xml)",
        )
        if path:
            self.path_edit.setText(path)

    def _import_quote(self) -> None:
        try:
            summary = self.presenter.stage_commercial_quote(
                self.path_edit.text(),
                supplier_name=self.supplier_edit.text(),
                quote_number=self.number_edit.text(),
                quote_date=self.date_edit.text(),
                region=self.region_edit.text(),
                assume_available=self.available_check.isChecked(),
            )
        except Exception as exc:
            QMessageBox.warning(self, "Импорт КП", str(exc))
            return
        self.completed = True
        QMessageBox.information(
            self,
            "Импорт КП",
            f"Источник: {summary.supplier_name}. Строк источника: {summary.records_total}.",
        )
        self.accept()


__all__ = ["CatalogCommercialQuoteDialog"]
