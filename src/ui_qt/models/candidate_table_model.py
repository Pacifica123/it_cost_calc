from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from ui_qt.models.row_table_model import RowTableModel

DEFAULT_CANDIDATE_COLUMNS = ("rank", "name", "score", "total_cost", "source")
DEFAULT_CANDIDATE_HEADERS = {
    "rank": "№",
    "name": "Кандидат",
    "score": "Оценка",
    "total_cost": "Стоимость",
    "source": "Источник",
}


class CandidateTableModel(RowTableModel):
    """Table model for GA/AHP candidate rows."""

    def __init__(
        self,
        rows: Iterable[Mapping[str, Any]] | None = None,
        columns: Sequence[str] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(
            rows=rows,
            columns=columns or DEFAULT_CANDIDATE_COLUMNS,
            headers={**DEFAULT_CANDIDATE_HEADERS, **dict(headers or {})},
        )
