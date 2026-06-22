from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ui_qt.dialogs import ConfigurationDetailsDialog
from ui_qt.models import RowTableModel
from ui_qt.presenters import (
    ConfigurationDetailsPresenter,
    HybridPresenter,
    QtAppPresenter,
    format_money,
)
from ui_qt.widgets import CollapsibleSection, CompactLabel, SmartTable

try:
    from PySide6.QtWidgets import (
        QAbstractSpinBox,
        QDoubleSpinBox,
        QFrame,
        QHBoxLayout,
        QPushButton,
        QSizePolicy,
        QVBoxLayout,
        QWidget,
    )
except ModuleNotFoundError as exc:
    if exc.name != "PySide6":
        raise
    QAbstractSpinBox = QDoubleSpinBox = QFrame = QHBoxLayout = QPushButton = QSizePolicy = QVBoxLayout = None  # type: ignore[assignment]
    QWidget = object  # type: ignore[assignment,misc]

_RANKING_COLUMNS = ("rank", "name", "hybrid", "ga", "ahp", "pareto", "total_cost", "comment")
_RANKING_HEADERS = {
    "rank": "№",
    "name": "Альтернатива",
    "hybrid": "Hybrid",
    "ga": "GA",
    "ahp": "AHP",
    "pareto": "Pareto",
    "total_cost": "Стоимость",
    "comment": "Комментарий",
}
_WARNING_COLUMNS = ("index", "warning")
_WARNING_HEADERS = {"index": "№", "warning": "Предупреждение"}


class HybridScreen(QWidget):  # type: ignore[misc,valid-type]
    """Final GA/AHP/Pareto/Hybrid step for the current ПО/ТО pool."""

    def __init__(
        self,
        app_presenter: QtAppPresenter | None = None,
        parent: QWidget | None = None,  # type: ignore[valid-type]
        *,
        scope: str,
        on_result_changed: Callable[[], None] | None = None,
    ) -> None:
        if QVBoxLayout is None:
            raise RuntimeError("PySide6 is required to create HybridScreen")
        super().__init__(parent)
        self.presenter = HybridPresenter(app_presenter, scope=scope)
        self.details_presenter = ConfigurationDetailsPresenter(
            self.presenter.app_presenter,
            scope=scope,
        )
        self._on_result_changed = on_result_changed
        self._summary_labels: dict[str, CompactLabel] = {}
        self._finance_labels: dict[str, CompactLabel] = {}
        self._build_ui()
        self.refresh_data()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addWidget(self._build_summary_cards(), 0)
        layout.addWidget(self._build_finance_cards(), 0)
        layout.addWidget(self._build_controls(), 0)

        self.ranking_model = RowTableModel([], columns=_RANKING_COLUMNS, headers=_RANKING_HEADERS)
        self.ranking_table = SmartTable(
            self.ranking_model,
            self,
            empty_title="Нет Hybrid",
            empty_status="Запустите итог",
            show_actions=False,
            on_edit=self._show_configuration,
            compact=True,
            table_height=180,
        )
        self.ranking_section = CollapsibleSection(
            "Рейтинг",
            self.ranking_table,
            self,
            tooltip="Итоговая витрина объединяет GA, AHP и Pareto-статус текущего пула.",
            expanded=False,
        )
        layout.addWidget(self.ranking_section, 0)

        self.warning_model = RowTableModel([], columns=_WARNING_COLUMNS, headers=_WARNING_HEADERS)
        self.warning_table = SmartTable(
            self.warning_model,
            self,
            empty_title="Нет предупреждений",
            empty_status="Проверка чистая",
            show_actions=False,
            compact=True,
            table_height=120,
        )
        self.warning_section = CollapsibleSection(
            "Проверка",
            self.warning_table,
            self,
            tooltip="Короткие предупреждения по расхождениям GA, AHP и Pareto.",
            expanded=False,
        )
        layout.addWidget(self.warning_section, 0)
        self.status_label = CompactLabel("Hybrid ожидает запуск", self, object_name="pageStatus")
        layout.addWidget(self.status_label, 0)
        layout.addStretch(1)

    def _build_summary_cards(self) -> QWidget:  # type: ignore[valid-type]
        box = QWidget(self)
        row = QHBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)
        for key, title in (
            ("status", "Статус"),
            ("pool", "Пул"),
            ("winner", "Лидер"),
            ("score", "Score"),
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

    def _build_finance_cards(self) -> QWidget:  # type: ignore[valid-type]
        box = QWidget(self)
        row = QHBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)
        for key, title in (
            ("capex", "CAPEX"),
            ("opex", "OPEX/мес"),
            ("once", "Разово"),
            ("items", "Позиций"),
        ):
            card, value_label = self._summary_card(title)
            self._finance_labels[key] = value_label
            row.addWidget(card, 1)
        return box

    def _build_controls(self) -> QWidget:  # type: ignore[valid-type]
        frame = QFrame(self)
        frame.setObjectName("surface")
        frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        row = QHBoxLayout(frame)
        row.setContentsMargins(12, 10, 12, 10)
        row.setSpacing(10)
        row.addWidget(CompactLabel("λ GA", frame), 0)
        self.lambda_input = QDoubleSpinBox(frame)
        self.lambda_input.setRange(0.0, 1.0)
        self.lambda_input.setDecimals(2)
        self.lambda_input.setSingleStep(0.05)
        self.lambda_input.setValue(0.5)
        self.lambda_input.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.lambda_input.setMinimumHeight(34)
        row.addWidget(self.lambda_input, 0)
        row.addStretch(1)
        self.run_button = QPushButton("Собрать итог", frame)
        self.run_button.setProperty("role", "primary")
        self.run_button.clicked.connect(self._run_hybrid)
        row.addWidget(self.run_button, 0)
        return frame

    def refresh_data(self) -> None:
        summary = self.presenter.summary()
        self._summary_labels["status"].setText(summary.status)
        self._summary_labels["pool"].setText(str(summary.pool_count))
        self._summary_labels["winner"].setText(summary.winner_name)
        self._summary_labels["score"].setText("—" if summary.winner_score is None else f"{summary.winner_score:.4f}")
        finance = self.presenter.financial_summary()
        self._finance_labels["capex"].setText(format_money(finance.capex))
        self._finance_labels["opex"].setText(format_money(finance.opex_monthly))
        self._finance_labels["once"].setText(format_money(finance.one_time))
        self._finance_labels["items"].setText(str(finance.item_count) if finance.source_status == "ok" else "—")
        self.ranking_model.replace_rows(self.presenter.ranking_rows())
        self.warning_model.replace_rows(self.presenter.warning_rows())
        self.ranking_table.refresh_state()
        self.warning_table.refresh_state()
        self.run_button.setEnabled(self.presenter.can_run())
        self.run_button.setToolTip("" if self.presenter.can_run() else "Сначала нужен AHP.")
        self.status_label.setText(self.presenter.status_text())
        has_ranking = summary.ranked_count > 0
        self.ranking_section.set_expanded(has_ranking)
        self.warning_section.set_expanded(summary.warning_count > 0)

    def run_primary_action(self) -> None:
        self._run_hybrid()

    def _run_hybrid(self) -> None:
        try:
            self.presenter.run(lambda_value=float(self.lambda_input.value()))
        except Exception as exc:  # pragma: no cover - GUI status fallback
            self.status_label.setText("Ошибка Hybrid")
            self.status_label.setToolTip(str(exc))
        self.refresh_data()
        if self._on_result_changed is not None:
            self._on_result_changed()

    def _show_configuration(self, _row_index: int, row: dict[str, Any]) -> None:
        details = self.details_presenter.details(str(row.get("candidate_id") or ""))
        if details is not None:
            ConfigurationDetailsDialog(details, self).exec()
