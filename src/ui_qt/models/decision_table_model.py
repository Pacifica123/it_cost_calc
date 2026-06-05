from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from ui_qt.models.row_table_model import RowTableModel

DEFAULT_DECISION_COLUMNS = ("rank", "name", "ga_score", "ahp_score", "hybrid_score")
DEFAULT_DECISION_HEADERS = {
    "rank": "№",
    "name": "Вариант",
    "ga_score": "GA",
    "ahp_score": "AHP",
    "hybrid_score": "Гибрид",
    "pareto_status": "Pareto",
}


class DecisionTableModel(RowTableModel):
    """Table model for decision results across GA/AHP/Pareto/Hybrid."""

    def __init__(
        self,
        rows: Iterable[Mapping[str, Any]] | None = None,
        columns: Sequence[str] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(
            rows=rows,
            columns=columns or DEFAULT_DECISION_COLUMNS,
            headers={**DEFAULT_DECISION_HEADERS, **dict(headers or {})},
        )
