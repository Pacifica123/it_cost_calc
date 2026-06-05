from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ui_qt.models import RowTableModel
from ui_qt.widgets.text_rules import assert_short_text

try:
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtWidgets import (
        QHBoxLayout,
        QHeaderView,
        QMenu,
        QPushButton,
        QStackedLayout,
        QTableView,
        QVBoxLayout,
        QWidget,
    )
except ModuleNotFoundError as exc:
    if exc.name != "PySide6":
        raise
    QPoint = Qt = None  # type: ignore[assignment]
    QHBoxLayout = QHeaderView = QMenu = QPushButton = QStackedLayout = QTableView = None  # type: ignore[assignment]
    QVBoxLayout = QWidget = object  # type: ignore[assignment,misc]

if QTableView is not None:
    from ui_qt.widgets.empty_state import EmptyState
else:
    EmptyState = object  # type: ignore[assignment,misc]

RowCallback = Callable[[int, dict[str, Any]], None]
ActionCallback = Callable[[], None]


class SmartTable(QWidget):  # type: ignore[misc,valid-type]
    """Compact table block with one primary CRUD action.

    The widget keeps the visible action set small: add is shown directly, while
    edit/delete live in a context menu and in the secondary ``Ещё`` menu.
    """

    def __init__(
        self,
        model: RowTableModel | None = None,
        parent: QWidget | None = None,  # type: ignore[valid-type]
        *,
        empty_title: str = "Нет данных",
        empty_status: str = "Добавьте запись",
        add_text: str = "Добавить",
        more_text: str = "Ещё",
        on_add: ActionCallback | None = None,
        on_edit: RowCallback | None = None,
        on_delete: RowCallback | None = None,
        on_select: RowCallback | None = None,
        show_actions_when_empty: bool = False,
        show_actions: bool = True,
        compact: bool = False,
        table_height: int | None = None,
    ) -> None:
        if QTableView is None:
            raise RuntimeError("PySide6 is required to create SmartTable")
        assert_short_text(empty_title, field="SmartTable.empty_title")
        assert_short_text(empty_status, field="SmartTable.empty_status")
        assert_short_text(add_text, field="SmartTable.add_text")
        assert_short_text(more_text, field="SmartTable.more_text")
        super().__init__(parent)
        self._model = model or RowTableModel()
        self._on_add = on_add
        self._on_edit = on_edit
        self._on_delete = on_delete
        self._on_select = on_select
        self._show_actions_when_empty = show_actions_when_empty
        self._show_actions = show_actions
        self._compact = compact

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6 if compact else 8)

        self.actions_bar = QWidget(self)
        actions = QHBoxLayout(self.actions_bar)
        actions.setContentsMargins(0, 0, 0, 0)
        actions.addStretch(1)

        self.add_button = QPushButton(add_text, self.actions_bar)
        self.add_button.setProperty("role", "primary")
        self.add_button.clicked.connect(self._emit_add)

        self.more_button = QPushButton(more_text, self.actions_bar)
        self.more_menu = QMenu(self.more_button)
        self.more_button.setMenu(self.more_menu)
        self.edit_action = self.more_menu.addAction("Изменить")
        self.delete_action = self.more_menu.addAction("Удалить")
        self.edit_action.triggered.connect(self._emit_edit)
        self.delete_action.triggered.connect(self._emit_delete)

        actions.addWidget(self.add_button, 0)
        actions.addWidget(self.more_button, 0)

        self.stack = QStackedLayout()
        self.table = QTableView(self)
        self.table.setModel(self._model)
        self.table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(False)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._show_context_menu)
        self.table.doubleClicked.connect(lambda _index: self._emit_edit())
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        if table_height is not None:
            self.table.setMinimumHeight(table_height)
            self.table.setMaximumHeight(table_height)
        elif compact:
            self.table.setMinimumHeight(96)
            self.table.setMaximumHeight(160)

        empty_action = add_text if show_actions and on_add is not None else None
        self.empty_state = EmptyState(
            empty_title,
            self,
            status=empty_status,
            action_text=empty_action,
            action_callback=self._emit_add if empty_action else None,
            compact=compact,
        )
        if compact:
            self.empty_state.setMaximumHeight(96)
        self.stack.addWidget(self.table)
        self.stack.addWidget(self.empty_state)

        layout.addWidget(self.actions_bar, 0)
        layout.addLayout(self.stack, 0 if compact else 1)
        self._connect_selection_model()
        self.refresh_state()

    @property
    def model(self) -> RowTableModel:
        return self._model

    def set_model(self, model: RowTableModel) -> None:
        self._model = model
        self.table.setModel(model)
        self._connect_selection_model()
        self.refresh_state()

    def replace_rows(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self._model.replace_rows(rows)
        self.refresh_state()

    def selected_row_index(self) -> int | None:
        selection = self.table.selectionModel()
        if selection is None:
            return None
        indexes = selection.selectedRows()
        if not indexes:
            return None
        row = indexes[0].row()
        return row if 0 <= row < self._model.rowCount() else None

    def selected_row(self) -> dict[str, Any] | None:
        row_index = self.selected_row_index()
        if row_index is None:
            return None
        return self._model.row_dict(row_index)

    def refresh_state(self) -> None:
        is_empty = self._model.is_empty()
        self.stack.setCurrentWidget(self.empty_state if is_empty else self.table)
        self.actions_bar.setVisible(self._show_actions and (self._show_actions_when_empty or not is_empty))
        self.add_button.setVisible(self._on_add is not None)
        self.more_button.setVisible(self._on_edit is not None or self._on_delete is not None)
        self.more_button.setEnabled(not is_empty)

    def _connect_selection_model(self) -> None:
        selection = self.table.selectionModel()
        if selection is not None:
            selection.selectionChanged.connect(lambda _selected, _deselected: self._emit_select())

    def _show_context_menu(self, point: QPoint) -> None:  # type: ignore[valid-type]
        if self._model.is_empty():
            return
        self.more_menu.exec(self.table.viewport().mapToGlobal(point))

    def _emit_add(self) -> None:
        if self._on_add is not None:
            self._on_add()

    def _emit_select(self) -> None:
        row_index = self.selected_row_index()
        row = self.selected_row()
        if row_index is not None and row is not None and self._on_select is not None:
            self._on_select(row_index, row)

    def _emit_edit(self) -> None:
        row_index = self.selected_row_index()
        row = self.selected_row()
        if row_index is not None and row is not None and self._on_edit is not None:
            self._on_edit(row_index, row)

    def _emit_delete(self) -> None:
        row_index = self.selected_row_index()
        row = self.selected_row()
        if row_index is not None and row is not None and self._on_delete is not None:
            self._on_delete(row_index, row)
