from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Mapping, Sequence

from ui_qt.models.table_roles import BaseQtTableModel, QT_TABLE


class RowTableModel(BaseQtTableModel):
    """Import-safe row table model for future Qt screens.

    The model exposes the Qt-style ``rowCount``/``columnCount``/``data`` API but
    also pure-Python helpers.  It deliberately stores plain dictionaries so the
    Qt UI stays separated from domain and storage details.
    """

    def __init__(
        self,
        rows: Iterable[Mapping[str, Any]] | None = None,
        columns: Sequence[str] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        try:
            super().__init__()
        except TypeError:
            # ``object`` fallback has no Qt constructor contract.
            pass
        self._columns: tuple[str, ...] = tuple(columns or ())
        self._headers: dict[str, str] = dict(headers or {})
        self._rows: list[dict[str, Any]] = []
        self.replace_rows(rows or [])

    @property
    def columns(self) -> tuple[str, ...]:
        return self._columns

    @property
    def headers(self) -> dict[str, str]:
        return dict(self._headers)

    def replace_rows(self, rows: Iterable[Mapping[str, Any]]) -> None:
        if hasattr(self, "beginResetModel"):
            self.beginResetModel()
        self._rows = [dict(row) for row in rows]
        if not self._columns:
            self._columns = self._infer_columns(self._rows)
        if hasattr(self, "endResetModel"):
            self.endResetModel()

    def append_row(self, row: Mapping[str, Any]) -> None:
        self.insert_row(self.rowCount(), row)

    def insert_row(self, row_index: int, row: Mapping[str, Any]) -> None:
        rows = [*self._rows]
        safe_index = max(0, min(row_index, len(rows)))
        rows.insert(safe_index, dict(row))
        self.replace_rows(rows)

    def update_row(self, row_index: int, row: Mapping[str, Any]) -> dict[str, Any]:
        old_row = self.row_dict(row_index)
        rows = [*self._rows]
        rows[row_index] = dict(row)
        self.replace_rows(rows)
        return old_row

    def remove_row(self, row_index: int) -> dict[str, Any]:
        row = self.row_dict(row_index)
        rows = [*self._rows]
        rows.pop(row_index)
        self.replace_rows(rows)
        return row

    def rowCount(self, _parent: Any = None) -> int:  # noqa: N802 - Qt API
        return len(self._rows)

    def columnCount(self, _parent: Any = None) -> int:  # noqa: N802 - Qt API
        return len(self._columns)

    def data(self, index: Any, role: int = QT_TABLE.display_role) -> Any:
        if role not in {QT_TABLE.display_role, QT_TABLE.edit_role}:
            return None
        if not self._is_valid_index(index):
            return None
        value = self._rows[index.row()].get(self._columns[index.column()], "")
        return self._format_value(value)

    def headerData(
        self, section: int, orientation: int, role: int = QT_TABLE.display_role
    ) -> Any:  # noqa: N802 - Qt API
        if role != QT_TABLE.display_role:
            return None
        if orientation == QT_TABLE.horizontal and 0 <= section < len(self._columns):
            column = self._columns[section]
            return self._headers.get(column, column)
        if orientation == QT_TABLE.vertical:
            return str(section + 1)
        return None

    def row_dict(self, row_index: int) -> dict[str, Any]:
        return deepcopy(self._rows[row_index])

    def as_rows(self) -> list[dict[str, Any]]:
        return deepcopy(self._rows)

    def is_empty(self) -> bool:
        return not self._rows

    def column_index(self, column: str) -> int:
        try:
            return self._columns.index(column)
        except ValueError as exc:
            raise KeyError(column) from exc

    def value_at(self, row_index: int, column: str) -> Any:
        return self._rows[row_index].get(column)

    @staticmethod
    def _infer_columns(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
        columns: list[str] = []
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(str(key))
        return tuple(columns)

    @staticmethod
    def _format_value(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:g}"
        return str(value)

    @staticmethod
    def _is_valid_index(index: Any) -> bool:
        if index is None:
            return False
        if hasattr(index, "isValid") and not index.isValid():
            return False
        return hasattr(index, "row") and hasattr(index, "column")
