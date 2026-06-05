from __future__ import annotations

from typing import Any

from ui_qt.models import RowTableModel
from ui_qt.presenters import EnergyPresenter, EnergyProfileInput, QtAppPresenter
from ui_qt.widgets import CollapsibleSection, CompactLabel, EmptyState, InfoHint

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QDoubleSpinBox,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QHeaderView,
        QPushButton,
        QStackedLayout,
        QTableView,
        QVBoxLayout,
        QWidget,
    )
except ModuleNotFoundError as exc:
    if exc.name != "PySide6":
        raise
    Qt = None  # type: ignore[assignment]
    QDoubleSpinBox = QFrame = QGridLayout = QHBoxLayout = QHeaderView = QPushButton = None  # type: ignore[assignment]
    QStackedLayout = QTableView = QVBoxLayout = None  # type: ignore[assignment]
    QWidget = object  # type: ignore[assignment,misc]


class EnergyScreen(QWidget):  # type: ignore[misc,valid-type]
    """Compact Qt screen for electricity cost calculation."""

    def __init__(
        self,
        app_presenter: QtAppPresenter | None = None,
        parent: QWidget | None = None,  # type: ignore[valid-type]
    ) -> None:
        if QDoubleSpinBox is None:
            raise RuntimeError("PySide6 is required to create EnergyScreen")
        super().__init__(parent)
        self.presenter = EnergyPresenter(app_presenter)
        self.model = RowTableModel(columns=self._columns(), headers=self._headers())
        self._summary_labels: dict[str, CompactLabel] = {}
        self._build_ui()
        self.refresh_data()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        layout.addWidget(self._build_header(), 0)
        layout.addWidget(self._build_summary(), 0)
        layout.addWidget(self._build_profile(), 0)
        layout.addWidget(self._build_details(), 0)
        layout.addStretch(1)
        layout.addLayout(self._build_actions(), 0)

    def _build_header(self) -> QWidget:  # type: ignore[valid-type]
        header = QFrame(self)
        header.setObjectName("surface")
        row = QHBoxLayout(header)
        row.setContentsMargins(12, 8, 12, 8)
        row.setSpacing(8)
        row.addWidget(CompactLabel("Профиль энергии", header), 0)
        row.addWidget(
            InfoHint(
                "Экран считает месячную стоимость электроэнергии для серверов, клиентов и сети.",
                header,
            ),
            0,
        )
        row.addStretch(1)
        self.status_label = CompactLabel("Готово", header)
        row.addWidget(self.status_label, 0)
        return header

    def _build_summary(self) -> QWidget:  # type: ignore[valid-type]
        panel = QFrame(self)
        panel.setObjectName("surface")
        grid = QGridLayout(panel)
        grid.setContentsMargins(12, 10, 12, 10)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        for index, (key, label) in enumerate(
            (
                ("cost", "Итого"),
                ("kwh", "кВт·ч"),
                ("power", "Мощность"),
                ("items", "Позиций"),
            )
        ):
            card = QFrame(panel)
            card.setObjectName("card")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(12, 10, 12, 10)
            card_layout.setSpacing(4)
            card_layout.addWidget(CompactLabel(label, card, object_name="pageStatus"), 0)
            value = CompactLabel("—", card, object_name="pageTitle")
            card_layout.addWidget(value, 0)
            self._summary_labels[key] = value
            grid.addWidget(card, 0, index)
            grid.setColumnStretch(index, 1)
        return panel

    def _build_profile(self) -> QWidget:  # type: ignore[valid-type]
        panel = QFrame(self)
        panel.setObjectName("surface")
        grid = QGridLayout(panel)
        grid.setContentsMargins(12, 10, 12, 10)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)

        grid.addWidget(CompactLabel("ч/день", panel), 0, 0)
        self.hours_input = self._spin_box(8.0, suffix=" ч")
        grid.addWidget(self.hours_input, 0, 1)

        grid.addWidget(CompactLabel("дней", panel), 0, 2)
        self.days_input = self._spin_box(22.0, suffix=" д")
        grid.addWidget(self.days_input, 0, 3)

        grid.addWidget(CompactLabel("₽/кВт·ч", panel), 0, 4)
        self.price_input = self._spin_box(1.0, suffix=" ₽")
        grid.addWidget(self.price_input, 0, 5)

        for column in (1, 3, 5):
            grid.setColumnStretch(column, 1)
        return panel

    def _build_details(self) -> QWidget:  # type: ignore[valid-type]
        details = QWidget(self)
        details.setObjectName("contentArea")
        layout = QVBoxLayout(details)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self._build_parameters_section(), 0)
        layout.addWidget(self._build_equipment_section(), 0)
        return details

    def _build_parameters_section(self) -> CollapsibleSection:
        content = QFrame(self)
        content.setObjectName("card")
        grid = QGridLayout(content)
        grid.setContentsMargins(10, 8, 10, 8)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)
        grid.addWidget(CompactLabel("24/7", content), 0, 0)
        self.round_clock_label = CompactLabel("—", content)
        grid.addWidget(self.round_clock_label, 0, 1)
        grid.addWidget(CompactLabel("Источник", content), 1, 0)
        grid.addWidget(CompactLabel("ТО", content), 1, 1)
        grid.setColumnStretch(1, 1)
        return CollapsibleSection(
            "Параметры",
            content,
            self,
            tooltip="Серверы считаются круглосуточными, остальное использует профиль экрана.",
            expanded=False,
        )

    def _build_equipment_section(self) -> CollapsibleSection:
        content = QFrame(self)
        content.setObjectName("card")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.equipment_stack = QStackedLayout()
        self.equipment_table = QTableView(content)
        self.equipment_table.setModel(self.model)
        self.equipment_table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.equipment_table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self.equipment_table.setAlternatingRowColors(True)
        self.equipment_table.verticalHeader().setVisible(False)
        self.equipment_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.equipment_table.setMinimumHeight(220)
        self.empty_equipment = EmptyState(
            "Нет оборудования",
            content,
            status="Загрузите демо",
            action_text="Демо",
            action_callback=self.load_demo,
        )
        self.empty_equipment.setMinimumHeight(180)
        self.equipment_stack.addWidget(self.equipment_table)
        self.equipment_stack.addWidget(self.empty_equipment)
        layout.addLayout(self.equipment_stack)
        return CollapsibleSection(
            "Оборудование",
            content,
            self,
            tooltip="Таблица показывает только позиции, участвующие в энергозатратах.",
            expanded=False,
        )

    def _build_actions(self) -> QHBoxLayout:  # type: ignore[valid-type]
        row = QHBoxLayout()
        row.addStretch(1)
        self.refresh_button = QPushButton("Обновить", self)
        self.refresh_button.clicked.connect(self.refresh_data)
        row.addWidget(self.refresh_button, 0)
        self.calculate_button = QPushButton("Рассчитать", self)
        self.calculate_button.setProperty("role", "primary")
        self.calculate_button.clicked.connect(self.calculate)
        row.addWidget(self.calculate_button, 0)
        return row

    def _spin_box(self, value: float, *, suffix: str) -> QDoubleSpinBox:  # type: ignore[valid-type]
        box = QDoubleSpinBox(self)
        box.setRange(0.0, 100000.0)
        box.setDecimals(2)
        box.setValue(value)
        box.setSuffix(suffix)
        box.setMinimumHeight(34)
        box.setKeyboardTracking(False)
        return box

    def refresh_data(self) -> None:
        self.model.replace_rows(self.presenter.table_rows())
        self._sync_equipment_state()
        self._sync_round_clock_label()
        self._sync_summary()
        self.status_label.setText(f"{self.model.rowCount()} позиций")

    def load_demo(self) -> None:
        self.presenter.app_presenter.load_demo_dataset()
        self.refresh_data()
        self.status_label.setText("Демо загружено")

    def calculate(self) -> None:
        if self.model.is_empty():
            self.status_label.setText("Нет данных")
            return
        profile = EnergyProfileInput(
            hours_per_day=float(self.hours_input.value()),
            working_days=float(self.days_input.value()),
            cost_per_kwh=float(self.price_input.value()),
        )
        self.presenter.calculate(profile)
        self._sync_summary()
        self.status_label.setText("Рассчитано")

    def _sync_summary(self) -> None:
        summary = self.presenter.summary()
        self._summary_labels["cost"].setText(f"{summary.total_cost:.2f} ₽")
        self._summary_labels["kwh"].setText(f"{summary.total_kwh:.2f}")
        self._summary_labels["power"].setText(f"{summary.total_power:.0f} Вт")
        self._summary_labels["items"].setText(str(summary.item_count))

    def _sync_equipment_state(self) -> None:
        self.equipment_stack.setCurrentWidget(
            self.empty_equipment if self.model.is_empty() else self.equipment_table,
        )

    def _sync_round_clock_label(self) -> None:
        names = sorted(self.presenter.round_the_clock_names())
        self.round_clock_label.setText(str(len(names)) if names else "нет")
        if names:
            self.round_clock_label.setToolTip("\n".join(names))

    @staticmethod
    def _columns() -> tuple[str, ...]:
        return ("name", "category", "quantity", "max_power", "mode")

    @staticmethod
    def _headers() -> dict[str, str]:
        return {
            "name": "Название",
            "category": "Категория",
            "quantity": "Кол.",
            "max_power": "Вт",
            "mode": "Режим",
        }
