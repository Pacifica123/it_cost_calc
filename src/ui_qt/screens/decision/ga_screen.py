from __future__ import annotations

from typing import Any

from ui_qt.models import CandidateTableModel, RowTableModel
from ui_qt.presenters import GaInput, GaPresenter, QtAppPresenter, format_money
from ui_qt.widgets import CollapsibleSection, CompactLabel, EmptyState, InfoHint, SmartTable

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QDoubleSpinBox,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QLabel,
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
    QDoubleSpinBox = QFrame = QGridLayout = QHBoxLayout = QLabel = QPushButton = QSizePolicy = QSpinBox = None  # type: ignore[assignment]
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
        layout.addWidget(self.best_section, 0)

        layout.addStretch(1)

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
        layout = QGridLayout(frame)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(8)

        defaults = self.presenter.default_input()
        self.budget_input = self._money_input(defaults.budget)
        self.units_input = self._spin_input(int(defaults.min_units), 0, 10_000)
        self.population_input = self._spin_input(defaults.population, 4, 1000)
        self.generations_input = self._spin_input(defaults.generations, 1, 2000)
        self.mutation_input = self._ratio_input(defaults.mutation_rate)
        self.seed_input = self._spin_input(defaults.seed, 0, 1_000_000)

        layout.addWidget(QLabel("Бюджет", frame), 0, 0)
        layout.addWidget(self.budget_input, 0, 1)
        layout.addWidget(QLabel("Мин. мест", frame), 0, 2)
        layout.addWidget(self.units_input, 0, 3)
        layout.addWidget(QLabel("Популяция", frame), 1, 0)
        layout.addWidget(self.population_input, 1, 1)
        layout.addWidget(QLabel("Поколений", frame), 1, 2)
        layout.addWidget(self.generations_input, 1, 3)

        advanced = QWidget(frame)
        advanced_layout = QGridLayout(advanced)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        advanced_layout.setHorizontalSpacing(10)
        advanced_layout.addWidget(QLabel("Мутация", advanced), 0, 0)
        advanced_layout.addWidget(self.mutation_input, 0, 1)
        advanced_layout.addWidget(QLabel("Seed", advanced), 0, 2)
        advanced_layout.addWidget(self.seed_input, 0, 3)
        self.parameters_section = CollapsibleSection(
            "Параметры",
            advanced,
            frame,
            tooltip="Редкие параметры запуска GA.",
            expanded=False,
        )
        layout.addWidget(self.parameters_section, 2, 0, 1, 4)

        self.run_button = QPushButton("Запустить GA", frame)
        self.run_button.setProperty("role", "primary")
        self.run_button.clicked.connect(self._run_ga)
        layout.addWidget(self.run_button, 3, 3)
        return frame

    def _spin_input(self, value: int, minimum: int, maximum: int) -> QSpinBox:  # type: ignore[valid-type]
        field = QSpinBox(self)
        field.setRange(minimum, maximum)
        field.setValue(int(value))
        return field

    def _money_input(self, value: float) -> QDoubleSpinBox:  # type: ignore[valid-type]
        field = QDoubleSpinBox(self)
        field.setRange(0.0, 1_000_000_000.0)
        field.setDecimals(2)
        field.setSingleStep(10_000.0)
        field.setValue(float(value))
        return field

    def _ratio_input(self, value: float) -> QDoubleSpinBox:  # type: ignore[valid-type]
        field = QDoubleSpinBox(self)
        field.setRange(0.0, 1.0)
        field.setDecimals(4)
        field.setSingleStep(0.01)
        field.setValue(float(value))
        return field

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
