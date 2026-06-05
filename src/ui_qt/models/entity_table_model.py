from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from ui_qt.models.row_table_model import RowTableModel

DEFAULT_ENTITY_COLUMNS = ("name", "quantity", "price", "monthly_cost", "one_time_cost")
DEFAULT_ENTITY_HEADERS = {
    "name": "Название",
    "quantity": "Кол.",
    "price": "Цена",
    "monthly_cost": "Ежемес.",
    "one_time_cost": "Разово",
    "scope": "Область",
    "component_type": "Тип",
}


class EntityTableModel(RowTableModel):
    """Table model for runtime entities edited by future Qt screens."""

    def __init__(
        self,
        rows: Iterable[Mapping[str, Any]] | None = None,
        columns: Sequence[str] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(
            rows=rows,
            columns=columns or DEFAULT_ENTITY_COLUMNS,
            headers={**DEFAULT_ENTITY_HEADERS, **dict(headers or {})},
        )
