from __future__ import annotations

from shared.constants import ANALYSIS_SCOPE_SOFTWARE, ANALYSIS_SCOPE_TECHNICAL
from ui_qt.models import RowTableModel
from ui_qt.navigation import WorkflowStep, WorkflowStepper
from ui_qt.presenters import QtAppPresenter, WorkspacePresenter, format_money
from ui_qt.screens.decision import AhpScreen, GaScreen, HybridScreen, ParetoScreen
from ui_qt.widgets import CollapsibleSection, CompactLabel, EmptyState, InfoHint, SmartTable

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QFrame,
        QHBoxLayout,
        QPushButton,
        QSizePolicy,
        QScrollArea,
        QStackedLayout,
        QStackedWidget,
        QVBoxLayout,
        QWidget,
    )
except ModuleNotFoundError as exc:
    if exc.name != "PySide6":
        raise
    Qt = None  # type: ignore[assignment]
    QFrame = QHBoxLayout = QPushButton = QSizePolicy = QScrollArea = None  # type: ignore[assignment]
    QStackedLayout = QStackedWidget = QVBoxLayout = None  # type: ignore[assignment]
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
    """Qt data and GA steps for software and technical workspaces."""

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
        self.stepper = WorkflowStepper(
            self,
            steps=self._workflow_steps(False),
            active_step_id="data",
            on_step_changed=self._open_step,
        )
        layout.addWidget(self.stepper, 0)

        self.step_stack = QStackedWidget(self)
        self.data_step = self._build_data_step()
        self.data_step_page = self._wrap_step_scroll(self.data_step)
        self.ga_step = GaScreen(
            self.presenter.app_presenter,
            self,
            scope=self.presenter.scope,
            on_pool_changed=self._refresh_step_availability,
        )
        self.ga_step_page = self._wrap_step_scroll(self.ga_step)
        self.ahp_step = AhpScreen(
            self.presenter.app_presenter,
            self,
            scope=self.presenter.scope,
            on_result_changed=self._refresh_step_availability,
        )
        self.ahp_step_page = self._wrap_step_scroll(self.ahp_step)
        self.pareto_step = ParetoScreen(
            self.presenter.app_presenter,
            self,
            scope=self.presenter.scope,
            on_result_changed=self._refresh_step_availability,
        )
        self.pareto_step_page = self._wrap_step_scroll(self.pareto_step)
        self.hybrid_step = HybridScreen(
            self.presenter.app_presenter,
            self,
            scope=self.presenter.scope,
            on_result_changed=self._refresh_step_availability,
        )
        self.hybrid_step_page = self._wrap_step_scroll(self.hybrid_step)
        self.step_stack.addWidget(self.data_step_page)
        self.step_stack.addWidget(self.ga_step_page)
        self.step_stack.addWidget(self.ahp_step_page)
        self.step_stack.addWidget(self.pareto_step_page)
        self.step_stack.addWidget(self.hybrid_step_page)
        layout.addWidget(self.step_stack, 1)


    def _wrap_step_scroll(self, widget: QWidget) -> QScrollArea:  # type: ignore[valid-type]
        scroll = QScrollArea(self)
        scroll.setObjectName("contentArea")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        widget.setParent(scroll)
        scroll.setWidget(widget)
        return scroll

    def _workflow_steps(self, data_ready: bool) -> tuple[WorkflowStep, ...]:
        return (
            WorkflowStep("data", "Данные", enabled=True),
            WorkflowStep("ga", "GA", enabled=data_ready, tooltip="Сначала подготовьте данные."),
            WorkflowStep("ahp", "AHP", tooltip="Доступно после GA-пула."),
            WorkflowStep("pareto", "Pareto", tooltip="Доступно после AHP."),
            WorkflowStep("hybrid", "Гибрид", tooltip="Итог появится после Pareto."),
        )

    def _build_header(self) -> QWidget:  # type: ignore[valid-type]
        header = QFrame(self)
        header.setObjectName("surface")
        row = QHBoxLayout(header)
        row.setContentsMargins(12, 8, 12, 8)
        row.setSpacing(8)
        row.addWidget(CompactLabel("Данные", header), 0)
        row.addWidget(
            InfoHint(
                "Здесь собирается исходный набор области. GA запускается следующим шагом.",
                header,
            ),
            0,
        )
        row.addStretch(1)
        self.status_label = CompactLabel(self.presenter.label, header)
        row.addWidget(self.status_label, 0)
        return header

    def _build_data_step(self) -> QWidget:  # type: ignore[valid-type]
        step = QWidget(self)
        layout = QVBoxLayout(step)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.content_stack = QStackedLayout()
        self.empty_state = EmptyState(
            "Нет данных",
            step,
            status="Загрузите демо",
            details="Нажмите «Демо» или создайте записи через компоненты.",
        )
        self.data_panel = self._build_data_panel(step)
        self.content_stack.addWidget(self.empty_state)
        self.content_stack.addWidget(self.data_panel)
        layout.addLayout(self.content_stack, 1)
        layout.addWidget(self._build_summary_cards(), 0)

        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        self.ga_button = QPushButton("К GA", step)
        self.ga_button.clicked.connect(lambda: self._open_step("ga"))
        actions.addStretch(1)
        actions.addWidget(self.ga_button, 0)
        layout.addLayout(actions, 0)
        return step

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

    def _build_data_panel(self, parent: QWidget) -> QWidget:  # type: ignore[valid-type]
        panel = QWidget(parent)
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
            tooltip="Краткая сводка по группам данных.",
            expanded=True,
        )
        self.capex_section = CollapsibleSection(
            "CAPEX",
            self.capex_table,
            panel,
            tooltip="Разовые активы текущей области.",
            expanded=False,
        )
        self.opex_section = CollapsibleSection(
            "OPEX",
            self.opex_table,
            panel,
            tooltip="Регулярные и проектные затраты.",
            expanded=False,
        )
        self._detail_sections = [
            self.category_section,
            self.capex_section,
            self.opex_section,
        ]
        for section in self._detail_sections:
            section.add_toggle_callback(
                lambda expanded, current=section: self._sync_workspace_accordion(current, expanded)
            )

        panel_layout.addWidget(self.category_section, 0)
        panel_layout.addWidget(self.capex_section, 0)
        panel_layout.addWidget(self.opex_section, 0)
        panel_layout.addStretch(1)
        return panel

    def _sync_workspace_accordion(self, current: CollapsibleSection, expanded: bool) -> None:
        if not expanded:
            return
        for section in self._detail_sections:
            if section is not current:
                section.set_expanded(False)

    def _open_step(self, step_id: str) -> None:
        if step_id == "ga" and self.presenter.has_data():
            self.step_stack.setCurrentWidget(self.ga_step_page)
            self.stepper.set_active("ga")
            self.ga_step.refresh_data()
            self._refresh_step_availability()
            return
        if step_id == "ahp" and self._has_candidate_pool():
            self.step_stack.setCurrentWidget(self.ahp_step_page)
            self.stepper.set_active("ahp")
            self.ahp_step.refresh_data()
            self._refresh_step_availability()
            return
        if step_id == "pareto" and self._has_ahp_result():
            self.step_stack.setCurrentWidget(self.pareto_step_page)
            self.stepper.set_active("pareto")
            self.pareto_step.refresh_data()
            self._refresh_step_availability()
            return
        if step_id == "hybrid" and self._has_pareto_result():
            self.step_stack.setCurrentWidget(self.hybrid_step_page)
            self.stepper.set_active("hybrid")
            self.hybrid_step.refresh_data()
            self._refresh_step_availability()
            return
        self.step_stack.setCurrentWidget(self.data_step_page)
        self.stepper.set_active("data")
        self._refresh_step_availability()

    def _has_candidate_pool(self) -> bool:
        return bool(
            self.presenter.app_presenter.get_candidate_pool_snapshot(self.presenter.scope).candidates
        )

    def _has_ahp_result(self) -> bool:
        return self.presenter.app_presenter.get_ahp_result(self.presenter.scope).get("status") == "ok"

    def _has_pareto_result(self) -> bool:
        return self.presenter.app_presenter.get_pareto_result(self.presenter.scope).get("status") == "ok"

    def _refresh_step_availability(self) -> None:
        data_ready = self.presenter.has_data()
        pool_ready = self._has_candidate_pool()
        ahp_ready = self._has_ahp_result()
        pareto_ready = self._has_pareto_result()
        self.stepper.set_step_enabled("ga", data_ready)
        self.stepper.set_step_enabled("ahp", pool_ready)
        self.stepper.set_step_enabled("pareto", ahp_ready)
        self.stepper.set_step_enabled("hybrid", pareto_ready)

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
        data_ready = summary.row_count > 0
        self.stepper.set_step_enabled("ga", data_ready)
        self.ga_button.setEnabled(data_ready)
        self.ga_button.setToolTip("" if data_ready else "Сначала нужны данные.")
        self.category_section.set_expanded(data_ready)
        self.capex_section.set_expanded(False)
        self.opex_section.set_expanded(False)
        self.content_stack.setCurrentWidget(self.data_panel if data_ready else self.empty_state)
        self.ga_step.refresh_data()
        self.ahp_step.refresh_data()
        self.pareto_step.refresh_data()
        self.hybrid_step.refresh_data()
        self._refresh_step_availability()

    def current_step_id(self) -> str:
        return self.stepper.current_step_id

    def next_workflow_step(self) -> None:
        order = ("data", "ga", "ahp", "pareto", "hybrid")
        current = self.current_step_id()
        for step_id in order[order.index(current) + 1 :]:
            if self._can_open_step(step_id):
                self._open_step(step_id)
                return

    def previous_workflow_step(self) -> None:
        order = ("data", "ga", "ahp", "pareto", "hybrid")
        current = self.current_step_id()
        for step_id in reversed(order[: order.index(current)]):
            if self._can_open_step(step_id):
                self._open_step(step_id)
                return

    def run_primary_action(self) -> None:
        step_id = self.current_step_id()
        if step_id == "data":
            self._open_step("ga")
        elif step_id == "ga":
            self.ga_step.run_primary_action()
        elif step_id == "ahp":
            self.ahp_step.run_primary_action()
        elif step_id == "pareto":
            self.pareto_step.run_primary_action()
        elif step_id == "hybrid":
            self.hybrid_step.run_primary_action()
        self._refresh_step_availability()

    def _can_open_step(self, step_id: str) -> bool:
        if step_id == "data":
            return True
        if step_id == "ga":
            return self.presenter.has_data()
        if step_id == "ahp":
            return self._has_candidate_pool()
        if step_id == "pareto":
            return self._has_ahp_result()
        if step_id == "hybrid":
            return self._has_pareto_result()
        return False



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
