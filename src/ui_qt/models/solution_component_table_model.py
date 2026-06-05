from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from ui_qt.models.row_table_model import RowTableModel

SOLUTION_COMPONENT_COLUMNS = (
    "name",
    "scope",
    "component_type",
    "editor_status",
    "readiness",
)
SOLUTION_COMPONENT_HEADERS = {
    "name": "Компонент",
    "scope": "Профиль",
    "component_type": "Тип",
    "editor_status": "Режим",
    "readiness": "Готовность",
}


class SolutionComponentTableModel(RowTableModel):
    """Compact table model for the Qt component editor screen."""

    def __init__(
        self,
        rows: Iterable[Mapping[str, Any]] | None = None,
        columns: Sequence[str] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(
            rows=rows,
            columns=columns or SOLUTION_COMPONENT_COLUMNS,
            headers={**SOLUTION_COMPONENT_HEADERS, **dict(headers or {})},
        )
