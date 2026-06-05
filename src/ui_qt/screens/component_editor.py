from __future__ import annotations

from typing import Any, Mapping

from ui_qt.models import SolutionComponentTableModel
from ui_qt.presenters import ComponentEditorPresenter, QtAppPresenter
from ui_qt.widgets import CollapsibleSection, CompactLabel, InfoHint, SmartTable
from ui_qt.widgets.text_rules import assert_short_text

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QCheckBox,
        QComboBox,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QLineEdit,
        QMenu,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QVBoxLayout,
        QWidget,
    )
except ModuleNotFoundError as exc:
    if exc.name != "PySide6":
        raise
    Qt = None  # type: ignore[assignment]
    QCheckBox = QComboBox = QFrame = QGridLayout = QHBoxLayout = QLineEdit = None  # type: ignore[assignment]
    QMenu = QMessageBox = QPlainTextEdit = QPushButton = QScrollArea = QSizePolicy = None  # type: ignore[assignment]
    QVBoxLayout = QWidget = object  # type: ignore[assignment,misc]

_COMMON_FIELDS = (
    "id",
    "name",
    "quantity",
    "purchase_cost",
    "implementation_cost",
    "migration_cost",
    "testing_cost",
    "monthly_cost",
    "annual_cost",
    "energy_cost",
)
_COST_FIELDS = (
    ("quantity", "Кол."),
    ("purchase_cost", "Покупка"),
    ("implementation_cost", "Внедрение"),
    ("migration_cost", "Миграция"),
    ("testing_cost", "Тесты"),
    ("monthly_cost", "Ежемес."),
    ("annual_cost", "Ежегодно"),
    ("energy_cost", "Энергия"),
)


class ComponentEditorScreen(QWidget):  # type: ignore[misc,valid-type]
    """Qt screen for the solution-component editor route."""

    def __init__(
        self,
        app_presenter: QtAppPresenter | None = None,
        parent: QWidget | None = None,  # type: ignore[valid-type]
    ) -> None:
        if QLineEdit is None:
            raise RuntimeError("PySide6 is required to create ComponentEditorScreen")
        super().__init__(parent)
        self.presenter = ComponentEditorPresenter(app_presenter)
        self.selected_index: int | None = None
        self.mode = "list"
        self._entries: dict[str, QLineEdit] = {}
        self._metric_entries: dict[str, QLineEdit] = {}
        self._suppress_scope_event = False
        self._build_ui()
        self._set_form_values(self.presenter.new_form_values())
        self.refresh_table()
        self.show_list()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        header = self._build_header()
        self.model = SolutionComponentTableModel(self.presenter.list_table_rows())
        self.table = SmartTable(
            self.model,
            self,
            empty_title="Нет компонентов",
            empty_status="Создайте первый",
            add_text="Создать",
            on_add=self.start_create,
            on_edit=self.load_row,
            on_select=self.load_row,
            on_delete=self.delete_row,
        )
        self.form = self._build_form()

        layout.addWidget(header, 0)
        layout.addWidget(self.table, 1)
        layout.addWidget(self.form, 1)

    def _build_header(self) -> QWidget:  # type: ignore[valid-type]
        header = QFrame(self)
        header.setObjectName("surface")
        row = QHBoxLayout(header)
        row.setContentsMargins(12, 8, 12, 8)
        row.setSpacing(8)
        self.route_label = CompactLabel("Выбрать → Изменить", header)
        row.addWidget(self.route_label, 0)
        row.addWidget(
            InfoHint(
                "Выберите строку, измените поля и сохраните. Необязательные параметры раскрываются отдельно.",
                header,
            ),
            0,
        )
        row.addStretch(1)
        self.status_label = CompactLabel("Готово", header)
        row.addWidget(self.status_label, 0)
        return header

    def _build_form(self) -> QWidget:  # type: ignore[valid-type]
        form = QFrame(self)
        self.form = form
        form.setObjectName("surface")
        form.setMinimumHeight(360)
        form.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout = QVBoxLayout(form)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(8)

        content = QWidget(form)
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(8)

        main_grid = QGridLayout()
        main_grid.setContentsMargins(0, 0, 0, 0)
        main_grid.setHorizontalSpacing(10)
        main_grid.setVerticalSpacing(8)

        self._add_line(main_grid, "id", "ID", 0, 0, placeholder="auto")
        self._add_line(main_grid, "name", "Название", 1, 0)

        main_grid.addWidget(CompactLabel("Профиль", content), 0, 2)
        self.scope_combo = QComboBox(content)
        for option in self.presenter.scope_options():
            self.scope_combo.addItem(option.label, option.value)
        self.scope_combo.setToolTip("Профиль определяет набор типов и метрик компонента.")
        self.scope_combo.currentIndexChanged.connect(self._on_scope_change)
        main_grid.addWidget(self.scope_combo, 0, 3)

        main_grid.addWidget(CompactLabel("Тип", content), 1, 2)
        self.type_combo = QComboBox(content)
        main_grid.addWidget(self.type_combo, 1, 3)

        for column in (1, 3):
            main_grid.setColumnStretch(column, 1)

        content_layout.addLayout(main_grid)
        content_layout.addWidget(self._build_cost_section(), 0)
        content_layout.addWidget(self._build_metrics_section(), 0)
        content_layout.addWidget(self._build_extra_section(), 0)
        content_layout.addWidget(self._build_preview_section(), 0)
        content_layout.addStretch(1)

        scroll = QScrollArea(form)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)
        layout.addLayout(self._build_buttons(), 0)
        return form

    def _build_cost_section(self) -> CollapsibleSection:
        content = QFrame(self)
        content.setObjectName("card")
        layout = QGridLayout(content)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(8)
        for index, (name, label) in enumerate(_COST_FIELDS):
            row = index // 2
            column = 0 if index % 2 == 0 else 2
            self._add_line(layout, name, label, row, column, parent=content)
        for column in (1, 3):
            layout.setColumnStretch(column, 1)
        return CollapsibleSection(
            "Стоимость",
            content,
            self,
            tooltip="Количество, покупка и регулярные расходы скрыты по умолчанию.",
            expanded=False,
        )

    def _build_metrics_section(self) -> CollapsibleSection:
        content = QFrame(self)
        content.setObjectName("card")
        self.metrics_layout = QGridLayout(content)
        self.metrics_layout.setContentsMargins(10, 8, 10, 8)
        self.metrics_layout.setHorizontalSpacing(10)
        self.metrics_layout.setVerticalSpacing(6)
        return CollapsibleSection(
            "Метрики",
            content,
            self,
            tooltip="Профильные поля нужны только для аналитики, энергии и отчёта.",
            expanded=False,
        )

    def _build_extra_section(self) -> CollapsibleSection:
        content = QFrame(self)
        content.setObjectName("card")
        layout = QGridLayout(content)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(6)

        layout.addWidget(CompactLabel("Описание", content), 0, 0)
        self.description_edit = QLineEdit(content)
        layout.addWidget(self.description_edit, 0, 1)

        self.strict_check = QCheckBox("Строгая аналитика", content)
        self.strict_check.setToolTip("Если выключено, компонент останется черновиком.")
        layout.addWidget(self.strict_check, 1, 0, 1, 2)

        layout.addWidget(CompactLabel("Допущения", content), 2, 0)
        self.assumptions_edit = QPlainTextEdit(content)
        self.assumptions_edit.setFixedHeight(74)
        self.assumptions_edit.setPlaceholderText("По одному на строку")
        layout.addWidget(self.assumptions_edit, 2, 1)
        layout.setColumnStretch(1, 1)
        return CollapsibleSection(
            "Дополнительно",
            content,
            self,
            tooltip="Описание, комментарии и режим строгой аналитики скрыты по умолчанию.",
            expanded=False,
        )

    def _build_preview_section(self) -> CollapsibleSection:
        self.preview_text = QPlainTextEdit(self)
        self.preview_text.setObjectName("card")
        self.preview_text.setReadOnly(True)
        self.preview_text.setFixedHeight(108)
        self.preview_text.setPlainText("Нажмите «Проверить».")
        return CollapsibleSection(
            "Проверка",
            self.preview_text,
            self,
            tooltip="Нормализация показывает готовность к аналитике до сохранения.",
            expanded=False,
        )

    def _build_buttons(self) -> QHBoxLayout:  # type: ignore[valid-type]
        buttons = QHBoxLayout()
        self.cancel_button = QPushButton("К списку", self)
        self.cancel_button.clicked.connect(self.show_list)
        buttons.addWidget(self.cancel_button)
        buttons.addStretch(1)

        self.check_button = QPushButton("Проверить", self)
        self.check_button.clicked.connect(self.preview_form)
        buttons.addWidget(self.check_button)

        self.save_button = QPushButton("Сохранить", self)
        self.save_button.setProperty("role", "primary")
        self.save_button.clicked.connect(self.save_form)
        buttons.addWidget(self.save_button)

        return buttons

    def _add_line(
        self,
        layout: QGridLayout,  # type: ignore[valid-type]
        name: str,
        label: str,
        row: int,
        column: int,
        *,
        parent: QWidget | None = None,  # type: ignore[valid-type]
        placeholder: str = "",
    ) -> QLineEdit:  # type: ignore[valid-type]
        assert_short_text(label, field=f"ComponentEditorScreen.{name}")
        owner = parent or self.form
        layout.addWidget(CompactLabel(label, owner), row, column)
        entry = QLineEdit(owner)
        entry.setMinimumWidth(170)
        entry.setMinimumHeight(32)
        if placeholder:
            entry.setPlaceholderText(placeholder)
        layout.addWidget(entry, row, column + 1)
        self._entries[name] = entry
        return entry

    def _on_scope_change(self) -> None:
        if self._suppress_scope_event:
            return
        self._rebuild_type_options()
        self._rebuild_metric_fields()

    def _rebuild_type_options(self) -> None:
        scope = self._current_scope()
        current = self.type_combo.currentData()
        self.type_combo.clear()
        for option in self.presenter.component_type_options(scope):
            self.type_combo.addItem(option.label, option.value)
        index = self.type_combo.findData(current)
        if index >= 0:
            self.type_combo.setCurrentIndex(index)

    def _rebuild_metric_fields(self, values: Mapping[str, Any] | None = None) -> None:
        while self.metrics_layout.count():
            item = self.metrics_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._metric_entries = {}
        metrics = dict(values or {})
        for index, field in enumerate(self.presenter.metric_fields(self._current_scope())):
            row = index // 2
            column = 0 if index % 2 == 0 else 2
            owner = self.metrics_layout.parentWidget() or self
            self.metrics_layout.addWidget(CompactLabel(field.label, owner), row, column)
            entry = QLineEdit(owner)
            entry.setText(str(metrics.get(field.name, "")))
            entry.setMinimumWidth(150)
            entry.setMinimumHeight(32)
            self._metric_entries[field.name] = entry
            self.metrics_layout.addWidget(entry, row, column + 1)
        for column in (1, 3):
            self.metrics_layout.setColumnStretch(column, 1)

    def refresh_table(self) -> None:
        self.model.replace_rows(self.presenter.list_table_rows())
        self.table.refresh_state()
        if self.mode == "list":
            self._set_status(self.presenter.status_summary())

    def show_list(self) -> None:
        self.mode = "list"
        self.selected_index = None
        self.table.setVisible(True)
        self.form.setVisible(False)
        self.route_label.setText("Выбрать → Изменить")
        self._set_status(self.presenter.status_summary())

    def start_create(self) -> None:
        self.mode = "create"
        self.selected_index = None
        self._set_form_values(self.presenter.new_form_values())
        self.preview_text.setPlainText("Нажмите «Проверить».")
        self.table.setVisible(False)
        self.form.setVisible(True)
        self.route_label.setText("Новая запись")
        self._set_status("Черновик")

    def load_row(self, row_index: int, _row: dict[str, Any] | None = None) -> None:
        self.mode = "edit"
        self.selected_index = row_index
        self._set_form_values(self.presenter.form_values_for_index(row_index))
        self.table.setVisible(False)
        self.form.setVisible(True)
        self.route_label.setText("Редактирование")
        self.preview_form(show_errors=False)
        self._set_status("Изменение")

    def delete_row(self, row_index: int, _row: dict[str, Any] | None = None) -> None:
        answer = QMessageBox.question(self, "Удалить?", "Удалить компонент?")
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.presenter.delete(row_index)
        self.refresh_table()
        self.show_list()
        self._set_status("Удалено")

    def preview_form(self, *, show_errors: bool = True) -> None:
        try:
            self.preview_text.setPlainText(self.presenter.preview(self._collect_values()))
            self._set_status("Проверено")
        except ValueError as error:
            if show_errors:
                QMessageBox.warning(self, "Ошибка ввода", str(error))
            self._set_status("Ошибка ввода")

    def save_form(self) -> None:
        try:
            result = self.presenter.save(self._collect_values(), selected_index=self.selected_index)
        except ValueError as error:
            QMessageBox.warning(self, "Ошибка ввода", str(error))
            self._set_status("Ошибка ввода")
            return
        self.selected_index = result.index
        self.refresh_table()
        self._set_form_values(self.presenter.form_values_for_index(result.index))
        self.preview_form(show_errors=False)
        self.mode = "edit"
        self.route_label.setText("Редактирование")
        self.table.setVisible(False)
        self.form.setVisible(True)
        self._set_status("Сохранено")

    def _set_form_values(self, values: Mapping[str, Any]) -> None:
        self._suppress_scope_event = True
        for name in _COMMON_FIELDS:
            self._entries[name].setText(str(values.get(name, "")))
        self._set_combo_value(self.scope_combo, str(values.get("scope") or "technical"))
        self._suppress_scope_event = False
        self._rebuild_type_options()
        self._set_combo_value(
            self.type_combo,
            str(values.get("component_type") or self.type_combo.currentData()),
        )
        self.description_edit.setText(str(values.get("description") or ""))
        self.strict_check.setChecked(bool(values.get("strict_analysis_participation", True)))
        self.assumptions_edit.setPlainText("\n".join(values.get("cost_assumptions") or []))
        self._rebuild_metric_fields(dict(values.get("metrics") or {}))

    def _collect_values(self) -> dict[str, Any]:
        values = {name: self._entries[name].text() for name in _COMMON_FIELDS}
        values.update(
            {
                "scope": self._current_scope(),
                "component_type": str(self.type_combo.currentData() or ""),
                "description": self.description_edit.text(),
                "strict_analysis_participation": self.strict_check.isChecked(),
                "cost_assumptions": self.assumptions_edit.toPlainText().splitlines(),
                "metrics": {
                    name: entry.text() for name, entry in self._metric_entries.items()
                },
            }
        )
        return values

    def _current_scope(self) -> str:
        return str(self.scope_combo.currentData() or "technical")

    def _set_combo_value(self, combo: QComboBox, value: str) -> None:  # type: ignore[valid-type]
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)
        self.status_label.setVisible(bool(text.strip()))
