from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class EntityTableSection:
    """Compact CRUD table block used inside dense ПО/ТО workspaces."""

    SHORT_COLUMN_NAMES = {
        "name": "Название",
        "quantity": "Кол.",
        "price": "Цена",
        "monthly_cost": "Ежемес.",
        "one_time_cost": "Разово",
    }

    def __init__(
        self,
        parent,
        title: str,
        columns: tuple[str, ...],
        column_names: dict[str, str],
        on_add,
        on_edit,
        on_delete,
        *,
        compact: bool = False,
        height: int | None = None,
    ):
        self.compact = compact
        self.frame = ttk.LabelFrame(parent, text=title, padding=(8, 6), style="Panel.TLabelframe")
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(0, weight=1)

        table_height = height if height is not None else (5 if compact else 7)
        self.table = ttk.Treeview(self.frame, columns=columns, show="headings", height=table_height)
        for column in columns:
            self.table.column(
                column,
                stretch=tk.YES,
                width=self._column_width(column),
                minwidth=self._column_min_width(column),
            )
            self.table.heading(column, text=self._heading(column, column_names))
        self.table.grid(row=0, column=0, sticky="nsew")

        y_scroll = ttk.Scrollbar(self.frame, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=y_scroll.set)
        y_scroll.grid(row=0, column=1, sticky="ns")

        buttons = ttk.Frame(self.frame, style="PanelBody.TFrame")
        buttons.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        for column in range(3):
            buttons.columnconfigure(column, weight=1, uniform="entity-actions")
        ttk.Button(buttons, text="Добавить", command=on_add).grid(row=0, column=0, sticky="ew")
        ttk.Button(buttons, text="Ред." if compact else "Редактировать", command=on_edit).grid(
            row=0, column=1, sticky="ew", padx=(6, 0)
        )
        ttk.Button(buttons, text="Удал." if compact else "Удалить", command=on_delete).grid(
            row=0, column=2, sticky="ew", padx=(6, 0)
        )

    def _heading(self, column: str, column_names: dict[str, str]) -> str:
        if self.compact:
            return self.SHORT_COLUMN_NAMES.get(column, column_names.get(column, column))
        return column_names.get(column, column)

    def _column_width(self, column: str) -> int:
        if self.compact:
            return {
                "name": 150,
                "quantity": 54,
                "price": 82,
                "monthly_cost": 88,
                "one_time_cost": 92,
            }.get(column, 90)
        return {
            "name": 210,
            "quantity": 90,
            "price": 120,
            "monthly_cost": 140,
            "one_time_cost": 150,
        }.get(column, 120)

    def _column_min_width(self, column: str) -> int:
        return {
            "name": 90 if self.compact else 120,
            "quantity": 38,
            "price": 60,
            "monthly_cost": 68,
            "one_time_cost": 72,
        }.get(column, 60)

    def grid(self, **kwargs):
        self.frame.grid(**kwargs)

    def pack(self, **kwargs):
        self.frame.pack(**kwargs)

    def replace_rows(self, rows: list[dict]) -> None:
        self.table.delete(*self.table.get_children())
        columns = self.table["columns"]
        for row in rows:
            self.table.insert("", "end", values=[row.get(column, "") for column in columns])

    def get_selected_index(self) -> int | None:
        selected_item = self.table.focus()
        if not selected_item:
            return None
        return list(self.table.get_children()).index(selected_item)
