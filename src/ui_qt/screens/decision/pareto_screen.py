from __future__ import annotations

from collections.abc import Callable

from ui_qt.models import RowTableModel
from ui_qt.presenters import ParetoPresenter, QtAppPresenter
from ui_qt.widgets import CollapsibleSection, CompactLabel, SmartTable

try:
    from PySide6.QtWidgets import QFrame, QHBoxLayout, QPushButton, QSizePolicy, QVBoxLayout, QWidget
except ModuleNotFoundError as exc:
    if exc.name != "PySide6":
        raise
    QFrame = QHBoxLayout = QPushButton = QSizePolicy = QVBoxLayout = None  # type: ignore[assignment]
    QWidget = object  # type: ignore[assignment,misc]

_RANKING_COLUMNS = ("rank", "name", "score", "status")
_RANKING_HEADERS = {
    "rank": "№",
    "name": "Альтернатива",
    "score": "Сумма",
    "status": "Pareto",
}
_NONDOMINATED_COLUMNS = ("id", "name")
_NONDOMINATED_HEADERS = {"id": "ID", "name": "Альтернатива"}


class ParetoScreen(QWidget):  # type: ignore[misc,valid-type]
    """Pareto step for the current ПО/ТО candidate pool."""

    def __init__(
        self,
        app_presenter: QtAppPresenter | None = None,
        parent: QWidget | None = None,  # type: ignore[valid-type]
        *,
        scope: str,
        on_result_changed: Callable[[], None] | None = None,
    ) -> None:
        if QVBoxLayout is None:
            raise RuntimeError("PySide6 is required to create ParetoScreen")
        super().__init__(parent)
        self.presenter = ParetoPresenter(app_presenter, scope=scope)
        self._on_result_changed = on_result_changed
        self._summary_labels: dict[str, CompactLabel] = {}
        self._build_ui()
        self.refresh_data()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addWidget(self._build_summary_cards(), 0)
        layout.addLayout(self._build_actions(), 0)

        self.ranking_model = RowTableModel([], columns=_RANKING_COLUMNS, headers=_RANKING_HEADERS)
        self.ranking_table = SmartTable(
            self.ranking_model,
            self,
            empty_title="Нет Pareto",
            empty_status="Запустите анализ",
            show_actions=False,
            compact=True,
            table_height=170,
        )
        self.ranking_section = CollapsibleSection(
            "Рейтинг",
            self.ranking_table,
            self,
            tooltip="Pareto показывает доминирование в текущем пуле.",
            expanded=True,
        )
        layout.addWidget(self.ranking_section, 0)

        self.nondominated_model = RowTableModel(
            [],
            columns=_NONDOMINATED_COLUMNS,
            headers=_NONDOMINATED_HEADERS,
        )
        self.nondominated_table = SmartTable(
            self.nondominated_model,
            self,
            empty_title="Нет лидеров",
            empty_status="Нет результата",
            show_actions=False,
            compact=True,
            table_height=120,
        )
        self.nondominated_section = CollapsibleSection(
            "Недом.",
            self.nondominated_table,
            self,
            tooltip="Недоминируемые альтернативы.",
            expanded=False,
        )
        layout.addWidget(self.nondominated_section, 0)
        self.status_label = CompactLabel("Pareto ожидает запуск", self, object_name="pageStatus")
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
            ("ranked", "Ранг."),
            ("winner", "Лидер"),
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

    def _build_actions(self) -> QHBoxLayout:  # type: ignore[valid-type]
        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.addStretch(1)
        self.run_button = QPushButton("Запустить Pareto", self)
        self.run_button.setProperty("role", "primary")
        self.run_button.clicked.connect(self._run_pareto)
        actions.addWidget(self.run_button, 0)
        return actions

    def refresh_data(self) -> None:
        summary = self.presenter.summary()
        self._summary_labels["status"].setText(summary.status)
        self._summary_labels["pool"].setText(str(summary.pool_count))
        self._summary_labels["ranked"].setText(str(summary.ranked_count))
        self._summary_labels["winner"].setText(summary.winner_name)
        self.ranking_model.replace_rows(self.presenter.ranking_rows())
        self.nondominated_model.replace_rows(self.presenter.nondominated_rows())
        self.ranking_table.refresh_state()
        self.nondominated_table.refresh_state()
        self.run_button.setEnabled(self.presenter.can_run())
        self.run_button.setToolTip("" if self.presenter.can_run() else "Сначала нужен GA-пул.")
        self.status_label.setText(self.presenter.status_text())
        self.ranking_section.set_expanded(summary.ranked_count > 0)
        self.nondominated_section.set_expanded(False)

    def _run_pareto(self) -> None:
        try:
            self.presenter.run()
        except Exception as exc:  # pragma: no cover - GUI status fallback
            self.status_label.setText("Ошибка Pareto")
            self.status_label.setToolTip(str(exc))
        self.refresh_data()
        if self._on_result_changed is not None:
            self._on_result_changed()
