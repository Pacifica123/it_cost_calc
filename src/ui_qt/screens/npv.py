from __future__ import annotations

from ui_qt.models import RowTableModel
from ui_qt.presenters import NpvInput, NpvPresenter, QtAppPresenter
from ui_qt.presenters.npv_presenter import format_cash_flows, parse_cash_flows
from ui_qt.widgets import CollapsibleSection, CompactLabel, InfoHint

try:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QDoubleValidator, QIntValidator
    from PySide6.QtWidgets import (
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QHeaderView,
        QLineEdit,
        QPushButton,
        QTableView,
        QVBoxLayout,
        QWidget,
    )
except ModuleNotFoundError as exc:
    if exc.name != "PySide6":
        raise
    Qt = None  # type: ignore[assignment]
    QDoubleValidator = QIntValidator = None  # type: ignore[assignment]
    QFrame = QGridLayout = QHBoxLayout = QHeaderView = QLineEdit = QPushButton = None  # type: ignore[assignment]
    QTableView = QVBoxLayout = None  # type: ignore[assignment]
    QWidget = object  # type: ignore[assignment,misc]


class NpvScreen(QWidget):  # type: ignore[misc,valid-type]
    """Compact Qt screen for NPV analysis."""

    def __init__(
        self,
        app_presenter: QtAppPresenter | None = None,
        parent: QWidget | None = None,  # type: ignore[valid-type]
    ) -> None:
        if QLineEdit is None:
            raise RuntimeError("PySide6 is required to create NpvScreen")
        super().__init__(parent)
        self.presenter = NpvPresenter(app_presenter)
        self.cash_flow_model = RowTableModel(columns=self._columns(), headers=self._headers())
        self.basis_model = RowTableModel(columns=("name", "value"), headers={"name": "Источник", "value": "Сумма"})
        self._summary_labels: dict[str, CompactLabel] = {}
        self._build_ui()
        self._load_defaults()
        self._sync_summary()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        layout.addWidget(self._build_header(), 0)
        layout.addWidget(self._build_summary(), 0)
        layout.addWidget(self._build_inputs(), 0)
        layout.addWidget(self._build_details(), 1)
        layout.addLayout(self._build_actions(), 0)

    def _build_header(self) -> QWidget:  # type: ignore[valid-type]
        header = QFrame(self)
        header.setObjectName("surface")
        row = QHBoxLayout(header)
        row.setContentsMargins(12, 8, 12, 8)
        row.setSpacing(8)
        row.addWidget(CompactLabel("NPV-модель", header), 0)
        row.addWidget(
            InfoHint(
                "Экран оценивает инвестиции, будущие потоки и дисконтирование по годам.",
                header,
            ),
            0,
        )
        row.addStretch(1)
        self.status_label = CompactLabel("Нет расчёта", header)
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
                ("npv", "NPV"),
                ("investment", "Инвестиции"),
                ("years", "Лет"),
                ("status", "Статус"),
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

    def _build_inputs(self) -> QWidget:  # type: ignore[valid-type]
        panel = QFrame(self)
        panel.setObjectName("surface")
        grid = QGridLayout(panel)
        grid.setContentsMargins(12, 10, 12, 10)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)

        self.investment_input = self._line("0")
        self.rate_input = self._line("0.10")
        self.horizon_input = self._line("5")
        self.effect_input = self._line("0")

        if QDoubleValidator is not None:
            self.investment_input.setValidator(QDoubleValidator(0.0, 1_000_000_000_000.0, 2, self))
            self.rate_input.setValidator(QDoubleValidator(0.0, 10.0, 4, self))
            self.effect_input.setValidator(QDoubleValidator(-1_000_000_000_000.0, 1_000_000_000_000.0, 2, self))
        if QIntValidator is not None:
            self.horizon_input.setValidator(QIntValidator(1, 50, self))

        self._add_field(grid, "Инвестиции", self.investment_input, 0, 0)
        self._add_field(grid, "Ставка", self.rate_input, 0, 2)
        self._add_field(grid, "Лет", self.horizon_input, 1, 0)
        self._add_field(grid, "Эффект/год", self.effect_input, 1, 2)

        for column in (1, 3):
            grid.setColumnStretch(column, 1)
        return panel

    def _build_details(self) -> QWidget:  # type: ignore[valid-type]
        details = QWidget(self)
        details.setObjectName("contentArea")
        layout = QVBoxLayout(details)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self._build_cash_flow_section(), 0)
        layout.addWidget(self._build_basis_section(), 0)
        layout.addWidget(self._build_table_section(), 1)
        return details

    def _build_cash_flow_section(self) -> CollapsibleSection:
        content = QFrame(self)
        content.setObjectName("card")
        layout = QGridLayout(content)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(8)
        self.cash_flows_input = self._line("0, 0, 0, 0, 0")
        self._add_field(layout, "Потоки", self.cash_flows_input, 0, 0)
        layout.setColumnStretch(1, 1)
        return CollapsibleSection(
            "Потоки",
            content,
            self,
            tooltip="Введите денежные потоки по годам через запятую.",
            expanded=False,
        )

    def _build_basis_section(self) -> CollapsibleSection:
        content = QFrame(self)
        content.setObjectName("card")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        self.basis_table = QTableView(content)
        self.basis_table.setModel(self.basis_model)
        self.basis_table.verticalHeader().setVisible(False)
        self.basis_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.basis_table.setMinimumHeight(120)
        layout.addWidget(self.basis_table)
        return CollapsibleSection(
            "Основа TCO",
            content,
            self,
            tooltip="CAPEX, OPEX и энергия используются как финансовая база.",
            expanded=False,
        )

    def _build_table_section(self) -> CollapsibleSection:
        content = QFrame(self)
        content.setObjectName("card")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        self.cash_flow_table = QTableView(content)
        self.cash_flow_table.setModel(self.cash_flow_model)
        self.cash_flow_table.verticalHeader().setVisible(False)
        self.cash_flow_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.cash_flow_table.setMinimumHeight(180)
        layout.addWidget(self.cash_flow_table)
        return CollapsibleSection(
            "Таблица",
            content,
            self,
            tooltip="Показывает дисконтированные значения и накопленный NPV.",
            expanded=True,
        )

    def _build_actions(self) -> QHBoxLayout:  # type: ignore[valid-type]
        row = QHBoxLayout()
        self.from_costs_button = QPushButton("Из TCO", self)
        self.from_costs_button.clicked.connect(self.load_from_costs)
        row.addWidget(self.from_costs_button, 0)
        row.addStretch(1)
        self.calculate_button = QPushButton("Рассчитать", self)
        self.calculate_button.setProperty("role", "primary")
        self.calculate_button.clicked.connect(self.calculate)
        row.addWidget(self.calculate_button, 0)
        return row

    def _line(self, text: str = "") -> QLineEdit:  # type: ignore[valid-type]
        line = QLineEdit(self)
        line.setText(text)
        line.setMinimumHeight(34)
        return line

    def _add_field(self, grid: QGridLayout, label: str, widget: QWidget, row: int, column: int) -> None:  # type: ignore[valid-type]
        grid.addWidget(CompactLabel(label, self), row, column)
        grid.addWidget(widget, row, column + 1)

    def _load_defaults(self) -> None:
        values = self.presenter.default_input()
        self._apply_input(values)
        self.status_label.setText("Нет расчёта")

    def refresh_data(self) -> None:
        self.load_from_costs()

    def load_from_costs(self) -> None:
        try:
            values = self.presenter.prepare_from_costs(
                horizon_years=self._int_value(self.horizon_input.text(), default=5),
                annual_effect=self._float_value(self.effect_input.text(), default=0.0),
                discount_rate=self._float_value(self.rate_input.text(), default=0.1),
            )
        except Exception as exc:  # pragma: no cover - GUI status fallback
            self.status_label.setText("Ошибка TCO")
            self.setToolTip(str(exc))
            return
        self._apply_input(values)
        self.basis_model.replace_rows(self.presenter.basis_rows())
        self.status_label.setText("TCO загружен")

    def calculate(self) -> None:
        try:
            values = self._read_input()
            report = self.presenter.calculate(values)
        except Exception as exc:  # pragma: no cover - GUI status fallback
            self.status_label.setText("Ошибка")
            self.setToolTip(str(exc))
            return
        self.cash_flow_model.replace_rows(self.presenter.table_rows(report))
        self.basis_model.replace_rows(self.presenter.basis_rows(report))
        self._sync_summary(report)
        warnings = self.presenter.warnings(report)
        self.status_label.setText("Есть риски" if warnings else "Рассчитано")
        if warnings:
            self.status_label.setToolTip("\n".join(warnings))

    def _apply_input(self, values: NpvInput) -> None:
        self.investment_input.setText(f"{values.investment:.2f}")
        self.rate_input.setText(f"{values.discount_rate:.4f}")
        self.horizon_input.setText(str(values.horizon_years))
        self.effect_input.setText(f"{values.annual_effect:.2f}")
        self.cash_flows_input.setText(format_cash_flows(values.cash_flows))

    def _read_input(self) -> NpvInput:
        horizon = self._int_value(self.horizon_input.text(), default=5)
        annual_effect = self._float_value(self.effect_input.text(), default=0.0)
        return NpvInput(
            investment=self._float_value(self.investment_input.text(), default=0.0),
            discount_rate=self._float_value(self.rate_input.text(), default=0.1),
            cash_flows=parse_cash_flows(
                self.cash_flows_input.text(),
                fallback_years=horizon,
                fallback_value=annual_effect,
            ),
            annual_effect=annual_effect,
            horizon_years=horizon,
        )

    def _sync_summary(self, report: dict | None = None) -> None:
        summary = self.presenter.summary(report)
        self._summary_labels["npv"].setText(f"{summary.npv:.2f} ₽")
        self._summary_labels["investment"].setText(f"{summary.investment:.2f} ₽")
        self._summary_labels["years"].setText(str(summary.years))
        self._summary_labels["status"].setText(summary.status)

    @staticmethod
    def _float_value(text: str, *, default: float) -> float:
        try:
            return float(text.replace(" ", "").replace(",", "."))
        except ValueError:
            return float(default)

    @staticmethod
    def _int_value(text: str, *, default: int) -> int:
        try:
            return max(int(float(text.replace(" ", "").replace(",", "."))), 1)
        except ValueError:
            return int(default)

    @staticmethod
    def _columns() -> tuple[str, ...]:
        return ("year", "cash_flow", "discount_factor", "present_value", "accumulated_npv")

    @staticmethod
    def _headers() -> dict[str, str]:
        return {
            "year": "Год",
            "cash_flow": "Поток",
            "discount_factor": "Коэф.",
            "present_value": "PV",
            "accumulated_npv": "NPV",
        }
