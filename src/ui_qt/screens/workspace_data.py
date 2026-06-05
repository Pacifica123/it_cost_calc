from __future__ import annotations

from typing import Any

from shared.constants import ANALYSIS_SCOPE_SOFTWARE, ANALYSIS_SCOPE_TECHNICAL
from ui_qt.models import RowTableModel
from ui_qt.navigation import WorkflowStepper
from ui_qt.presenters import QtAppPresenter, WorkspacePresenter, format_money
from ui_qt.widgets import CollapsibleSection, CompactLabel, EmptyState, InfoHint, SmartTable

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QFrame,
        QHBoxLayout,
        QPushButton,
        QSizePolicy,
        QStackedLayout,
        QVBoxLayout,
        QWidget,
    )
except ModuleNotFoundError as exc:
    if exc.name != "PySide6":
        raise
    Qt = None  # type: ignore[assignment]
    QFrame = QHBoxLayout = QPushButton = QSizePolicy = QStackedLayout = QVBoxLayout = None  # type: ignore[assignment]
    QWidget = object  # type: ignore[assignment,misc]

_CAPEX_COLUMNS = ("category", "name", "quantity", "price", "total")
_CAPEX_HEADERS = {
    "category": "Категория",
    "name": "Название",
    "quantity": "Кол.",
    "price": "Цена",
    "total": "Итого",
}
_OPEX_COLUMNS = ("category", "name", "monthly_cost", "one_time_cost")
_OPEX_HEADERS = {
    "category": "Категория",
    "name": "Название",
    "monthly_cost": "Ежемес.",
    "one_time_cost": "Разово",
}
_CATEGORY_COLUMNS = ("group", "category", "items", "total")
_CATEGORY_HEADERS = {
    "group": "Группа",
    "category": "Категория",
    "items": "Позиций",
    "total": "Сумма",
}


class WorkspaceDataScreen(QWidget):  # type: ignore[misc,valid-type]
    """Qt data step for software and technical workspaces."""

    def __init__(
        self,
        app_presenter: QtAppPresenter | None = None,
        parent: QWidget | None = None,  # type: ignore[valid-type]
        *,
        scope: str,
    ) -> None:
        if QVBoxLayout is None:
            raise RuntimeError("PySide6 is required to create WorkspaceDataScreen")
        super().__init__(parent)
        self.presenter = WorkspacePresenter(app_presenter, scope=scope)
        self._summary_labels: dict[str, CompactLabel] = {}
        self._build_ui()
        self.refresh_data()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        layout.addWidget(self._build_header(), 0)
        layout.addWidget(WorkflowStepper(self), 0)
        layout.addWidget(self._build_summary_cards(), 0)

        self.content_stack = QStackedLayout()
        self.empty_state = EmptyState(
            "Нет данных",
            self,
            status="Загрузите демо",
            details=(
                "Нажмите «Демо» в верхней панели или создайте записи через редактор компонентов."
            ),
        )
        self.data_panel = self._build_data_panel()
        self.content_stack.addWidget(self.empty_state)
        self.content_stack.addWidget(self.data_panel)
        layout.addLayout(self.content_stack, 1)

        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        self.backlog_button = QPushButton("К GA", self)
        self.backlog_button.setEnabled(False)
        self.backlog_button.setToolTip("Экран GA будет перенесён следующим патчем маршрута.")
        actions.addStretch(1)
        actions.addWidget(self.backlog_button, 0)
        layout.addLayout(actions, 0)

    def _build_header(self) -> QWidget:  # type: ignore[valid-type]
        header = QFrame(self)
        header.setObjectName("surface")
        row = QHBoxLayout(header)
        row.setContentsMargins(12, 8, 12, 8)
        row.setSpacing(8)
        row.addWidget(CompactLabel("Данные", header), 0)
        row.addWidget(
            InfoHint(
                "Здесь собирается исходный набор области. GA, AHP, Pareto и гибрид будут отдельными шагами.",
                header,
            ),
            0,
        )
        row.addStretch(1)
        self.status_label = CompactLabel(self.presenter.label, header)
        row.addWidget(self.status_label, 0)
        return header

    def _build_summary_cards(self) -> QWidget:  # type: ignore[valid-type]
        box = QWidget(self)
        row = QHBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)
        for key, title in (
            ("capex", "CAPEX"),
            ("opex", "OPEX/мес"),
            ("once", "Разово"),
            ("rows", "Позиций"),
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
        value_label = CompactLabel("0", card, object_name="pageTitle")
        layout.addWidget(value_label)
        return card, value_label

    def _build_data_panel(self) -> QWidget:  # type: ignore[valid-type]
        panel = QWidget(self)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(10)

        self.category_model = RowTableModel(
            [],
            columns=_CATEGORY_COLUMNS,
            headers=_CATEGORY_HEADERS,
        )
        self.capex_model = RowTableModel([], columns=_CAPEX_COLUMNS, headers=_CAPEX_HEADERS)
        self.opex_model = RowTableModel([], columns=_OPEX_COLUMNS, headers=_OPEX_HEADERS)

        self.category_table = SmartTable(
            self.category_model,
            panel,
            empty_title="Нет категорий",
            empty_status="Загрузите демо",
            show_actions=False,
            compact=True,
            table_height=132,
        )
        self.capex_table = SmartTable(
            self.capex_model,
            panel,
            empty_title="Нет CAPEX",
            empty_status="Нет активов",
            show_actions=False,
            compact=True,
            table_height=148,
        )
        self.opex_table = SmartTable(
            self.opex_model,
            panel,
            empty_title="Нет OPEX",
            empty_status="Нет расходов",
            show_actions=False,
            compact=True,
            table_height=148,
        )

        self.category_section = CollapsibleSection(
            "Категории",
            self.category_table,
            panel,
            tooltip="Краткая сводка по группам данных текущей области.",
            expanded=True,
        )
        self.capex_section = CollapsibleSection(
            "CAPEX",
            self.capex_table,
            panel,
            tooltip="Разовые активы текущей области анализа.",
            expanded=False,
        )
        self.opex_section = CollapsibleSection(
            "OPEX",
            self.opex_table,
            panel,
            tooltip="Регулярные и проектные затраты текущей области.",
            expanded=False,
        )
        panel_layout.addWidget(self.category_section, 0)
        panel_layout.addWidget(self.capex_section, 0)
        panel_layout.addWidget(self.opex_section, 0)
        panel_layout.addStretch(1)
        return panel

    def refresh_data(self) -> None:
        summary = self.presenter.summary()
        self.status_label.setText(f"{summary.label}: {summary.row_count}")
        self._summary_labels["capex"].setText(format_money(summary.capex_total))
        self._summary_labels["opex"].setText(format_money(summary.opex_monthly))
        self._summary_labels["once"].setText(format_money(summary.opex_once))
        self._summary_labels["rows"].setText(str(summary.row_count))
        self.category_model.replace_rows(self.presenter.category_rows())
        self.capex_model.replace_rows(self.presenter.capex_table_rows())
        self.opex_model.replace_rows(self.presenter.opex_table_rows())
        self.category_table.refresh_state()
        self.capex_table.refresh_state()
        self.opex_table.refresh_state()
        self.category_section.set_expanded(summary.row_count > 0)
        self.capex_section.set_expanded(summary.capex_count > 0)
        self.opex_section.set_expanded(summary.opex_count > 0 and summary.capex_count == 0)
        self.content_stack.setCurrentWidget(self.data_panel if summary.row_count else self.empty_state)


class SoftwareWorkspaceScreen(WorkspaceDataScreen):  # type: ignore[misc]
    def __init__(
        self,
        app_presenter: QtAppPresenter | None = None,
        parent: QWidget | None = None,  # type: ignore[valid-type]
    ) -> None:
        super().__init__(app_presenter, parent, scope=ANALYSIS_SCOPE_SOFTWARE)


class TechnicalWorkspaceScreen(WorkspaceDataScreen):  # type: ignore[misc]
    def __init__(
        self,
        app_presenter: QtAppPresenter | None = None,
        parent: QWidget | None = None,  # type: ignore[valid-type]
    ) -> None:
        super().__init__(app_presenter, parent, scope=ANALYSIS_SCOPE_TECHNICAL)
