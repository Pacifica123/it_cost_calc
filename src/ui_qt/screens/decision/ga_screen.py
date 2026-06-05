from __future__ import annotations

from typing import Any

from ui_qt.models import CandidateTableModel, RowTableModel
from ui_qt.presenters import GaInput, GaPresenter, QtAppPresenter, format_money
from ui_qt.widgets import CollapsibleSection, CompactLabel, SmartTable

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QAbstractSpinBox,
        QDoubleSpinBox,
        QFrame,
        QHBoxLayout,
        QPushButton,
        QSizePolicy,
        QSpinBox,
        QStackedLayout,
        QVBoxLayout,
        QWidget,
    )
except ModuleNotFoundError as exc:
    if exc.name != "PySide6":
        raise
    Qt = None  # type: ignore[assignment]
    QAbstractSpinBox = QDoubleSpinBox = QFrame = QHBoxLayout = QPushButton = QSizePolicy = QSpinBox = None  # type: ignore[assignment]
    QStackedLayout = QVBoxLayout = None  # type: ignore[assignment]
    QWidget = object  # type: ignore[assignment,misc]

_BEST_COLUMNS = ("category", "name", "quantity", "cost")
_BEST_HEADERS = {
    "category": "Категория",
    "name": "Название",
    "quantity": "Кол.",
    "cost": "Стоимость",
}


class GaScreen(QWidget):  # type: ignore[misc,valid-type]
    """Qt GA step for ПО/ТО workspaces."""

    def __init__(
        self,
        app_presenter: QtAppPresenter | None = None,
        parent: QWidget | None = None,  # type: ignore[valid-type]
        *,
        scope: str,
    ) -> None:
        if QVBoxLayout is None:
            raise RuntimeError("PySide6 is required to create GaScreen")
        super().__init__(parent)
        self.presenter = GaPresenter(app_presenter, scope=scope)
        self._summary_labels: dict[str, CompactLabel] = {}
        self._build_ui()
        self.refresh_data()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        layout.addWidget(self._build_summary_cards(), 0)
        layout.addWidget(self._build_controls(), 0)
        layout.addWidget(self._build_parameters_section(), 0)
        layout.addLayout(self._build_run_actions(), 0)

        self.result_hint = CompactLabel("Кандидатов пока нет", self, object_name="pageStatus")
        layout.addWidget(self.result_hint, 0)

        self.candidate_model = CandidateTableModel([])
        self.candidate_table = SmartTable(
            self.candidate_model,
            self,
            empty_title="Нет кандидатов",
            empty_status="Запустите GA",
            show_actions=False,
            compact=True,
            table_height=160,
        )
        self.candidate_section = CollapsibleSection(
            "Кандидаты",
            self.candidate_table,
            self,
            tooltip="Top-N решений текущей области.",
            expanded=False,
        )
        self.candidate_section.setVisible(False)
        layout.addWidget(self.candidate_section, 0)

        self.best_model = RowTableModel([], columns=_BEST_COLUMNS, headers=_BEST_HEADERS)
        self.best_table = SmartTable(
            self.best_model,
            self,
            empty_title="Нет состава",
            empty_status="Нет результата",
            show_actions=False,
            compact=True,
            table_height=140,
        )
        self.best_section = CollapsibleSection(
            "Состав",
            self.best_table,
            self,
            tooltip="Состав лучшего GA-кандидата.",
            expanded=False,
        )
        self.best_section.setVisible(False)
        layout.addWidget(self.best_section, 0)

        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        self.pool_label = CompactLabel("Пул пуст", self, object_name="pageStatus")
        self.transfer_button = QPushButton("Передать в AHP", self)
        self.transfer_button.setEnabled(False)
        self.transfer_button.clicked.connect(self._mark_pool_ready)
        actions.addWidget(self.pool_label, 1)
        actions.addWidget(self.transfer_button, 0)
        layout.addLayout(actions, 0)

    def _build_summary_cards(self) -> QWidget:  # type: ignore[valid-type]
        box = QWidget(self)
        row = QHBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)
        for key, title in (
            ("status", "Статус"),
            ("candidates", "Кандидатов"),
            ("best", "Лучший score"),
            ("cost", "Стоимость"),
        ):
            card, value_label = self._summary_card(title)
            self._summary_labels[key] = value_label
            row.addWidget(card, 1)
        return box

    def _summary_card(self, title: str) -> tuple[QFrame, CompactLabel]:  # type: ignore[valid-type]
        card = QFrame(self)
        card.setObjectName("surface")
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)
        layout.addWidget(CompactLabel(title, card, object_name="pageStatus"))
        value_label = CompactLabel("—", card, object_name="pageTitle")
        layout.addWidget(value_label)
        return card, value_label

    def _build_controls(self) -> QWidget:  # type: ignore[valid-type]
        frame = QFrame(self)
        frame.setObjectName("surface")
        frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        frame.setMinimumHeight(88)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(8)

        defaults = self.presenter.default_input()
        self.budget_input = self._money_input(defaults.budget)
        self.units_input = self._spin_input(int(defaults.min_units), 0, 10_000)
        self.population_input = self._spin_input(defaults.population, 4, 1000)
        self.generations_input = self._spin_input(defaults.generations, 1, 2000)
        self.mutation_input = self._ratio_input(defaults.mutation_rate)
        self.seed_input = self._spin_input(defaults.seed, 0, 1_000_000)

        fields = QWidget(frame)
        fields_layout = QHBoxLayout(fields)
        fields_layout.setContentsMargins(0, 0, 0, 0)
        fields_layout.setSpacing(10)
        fields_layout.addWidget(self._field_box("Бюджет", self.budget_input, fields), 1)
        fields_layout.addWidget(self._field_box("Мин. мест", self.units_input, fields), 1)
        fields_layout.addWidget(self._field_box("Популяция", self.population_input, fields), 1)
        fields_layout.addWidget(self._field_box("Поколений", self.generations_input, fields), 1)
        layout.addWidget(fields, 0)
        return frame

    def _build_parameters_section(self) -> CollapsibleSection:
        defaults_panel = QFrame(self)
        defaults_panel.setObjectName("card")
        row = QHBoxLayout(defaults_panel)
        row.setContentsMargins(10, 10, 10, 10)
        row.setSpacing(10)
        row.addWidget(self._field_box("Мутация", self.mutation_input, defaults_panel), 1)
        row.addWidget(self._field_box("Seed", self.seed_input, defaults_panel), 1)
        self.parameters_section = CollapsibleSection(
            "Параметры",
            defaults_panel,
            self,
            tooltip="Редкие параметры запуска GA.",
            expanded=False,
        )
        return self.parameters_section

    def _build_run_actions(self) -> QHBoxLayout:  # type: ignore[valid-type]
        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.addStretch(1)
        self.run_button = QPushButton("Запустить GA", self)
        self.run_button.setProperty("role", "primary")
        self.run_button.setMinimumHeight(38)
        self.run_button.clicked.connect(self._run_ga)
        actions.addWidget(self.run_button, 0)
        return actions

    def _field_box(self, title: str, field: QWidget, parent: QWidget) -> QWidget:  # type: ignore[valid-type]
        box = QWidget(parent)
        box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        layout.addWidget(CompactLabel(title, box, object_name="pageStatus"), 0)
        layout.addWidget(field, 0)
        return box

    def _spin_input(self, value: int, minimum: int, maximum: int) -> QSpinBox:  # type: ignore[valid-type]
        field = QSpinBox(self)
        self._make_compact_number_field(field)
        field.setRange(minimum, maximum)
        field.setValue(int(value))
        return field

    def _money_input(self, value: float) -> QDoubleSpinBox:  # type: ignore[valid-type]
        field = QDoubleSpinBox(self)
        self._make_compact_number_field(field)
        field.setRange(0.0, 1_000_000_000.0)
        field.setDecimals(2)
        field.setSingleStep(10_000.0)
        field.setValue(float(value))
        return field

    def _ratio_input(self, value: float) -> QDoubleSpinBox:  # type: ignore[valid-type]
        field = QDoubleSpinBox(self)
        self._make_compact_number_field(field)
        field.setRange(0.0, 1.0)
        field.setDecimals(4)
        field.setSingleStep(0.01)
        field.setValue(float(value))
        return field

    def _make_compact_number_field(self, field: QSpinBox | QDoubleSpinBox) -> None:  # type: ignore[valid-type]
        field.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        field.setMinimumHeight(36)
        field.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def refresh_data(self) -> None:
        summary = self.presenter.summary()
        self._summary_labels["status"].setText(summary.status)
        self._summary_labels["candidates"].setText(str(summary.candidate_count))
        self._summary_labels["best"].setText(f"{summary.best_score:.4f}")
        self._summary_labels["cost"].setText(format_money(summary.best_cost))
        self.candidate_model.replace_rows(self.presenter.candidate_rows())
        self.best_model.replace_rows(self.presenter.best_items_rows())
        self.candidate_table.refresh_state()
        self.best_table.refresh_state()
        has_result = summary.pool_count > 0
        self.result_hint.setVisible(not has_result)
        self.candidate_section.setVisible(has_result)
        self.best_section.setVisible(has_result)
        self.candidate_section.set_expanded(has_result)
        self.best_section.set_expanded(False)
        self.transfer_button.setEnabled(has_result)
        self.pool_label.setText(self.presenter.pool_summary_text())
        self.run_button.setEnabled(self.presenter.can_run())
        self.run_button.setToolTip("" if self.presenter.can_run() else "Сначала нужны данные.")

    def _values(self) -> GaInput:
        return GaInput(
            population=int(self.population_input.value()),
            generations=int(self.generations_input.value()),
            mutation_rate=float(self.mutation_input.value()),
            seed=int(self.seed_input.value()),
            budget=float(self.budget_input.value()),
            min_units=float(self.units_input.value()),
        )

    def _run_ga(self) -> None:
        try:
            self.presenter.run(self._values())
            self.pool_label.setText("GA готов")
        except Exception as exc:  # pragma: no cover - GUI status fallback
            self.pool_label.setText("Ошибка GA")
            self.pool_label.setToolTip(str(exc))
        self.refresh_data()

    def _mark_pool_ready(self) -> None:
        self.pool_label.setText("Пул передан")
