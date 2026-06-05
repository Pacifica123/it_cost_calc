from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class QtTableCompat:
    """Small import-safe subset of Qt constants used by table models.

    PySide6 is an optional runtime during the migration.  The models keep the
    same public methods with or without Qt so presenters and tests can validate
    the data boundary before the visual screens are moved.
    """

    display_role: int
    edit_role: int
    horizontal: int
    vertical: int


def _qt_int(value: Any) -> int:
    raw_value = getattr(value, "value", value)
    return int(raw_value)


def load_qt_table_compat() -> tuple[type[Any], QtTableCompat]:
    """Return QAbstractTableModel and role constants.

    When PySide6 is missing, return a plain object base class and stable numeric
    constants.  That keeps pure adapter tests independent from GUI packages.
    """

    try:
        from PySide6.QtCore import QAbstractTableModel, Qt
    except ModuleNotFoundError as exc:
        if exc.name != "PySide6":
            raise
        return object, QtTableCompat(
            display_role=0,
            edit_role=2,
            horizontal=1,
            vertical=2,
        )

    return QAbstractTableModel, QtTableCompat(
        display_role=_qt_int(Qt.ItemDataRole.DisplayRole),
        edit_role=_qt_int(Qt.ItemDataRole.EditRole),
        horizontal=_qt_int(Qt.Orientation.Horizontal),
        vertical=_qt_int(Qt.Orientation.Vertical),
    )


BaseQtTableModel, QT_TABLE = load_qt_table_compat()
