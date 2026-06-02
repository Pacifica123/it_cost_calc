from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class EntityTableSection:
    def __init__(
        self,
        parent,
        title: str,
        columns: tuple[str, ...],
        column_names: dict[str, str],
        on_add,
        on_edit,
        on_delete,
    ):
        self.frame = ttk.LabelFrame(parent, text=title, padding=(8, 6))
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(0, weight=1)

        self.table = ttk.Treeview(self.frame, columns=columns, show="headings", height=7)
        for column in columns:
            self.table.column(column, stretch=tk.YES, width=self._column_width(column))
            self.table.heading(column, text=column_names.get(column, column))
        self.table.grid(row=0, column=0, sticky="nsew")

        y_scroll = ttk.Scrollbar(self.frame, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=y_scroll.set)
        y_scroll.grid(row=0, column=1, sticky="ns")

        buttons = ttk.Frame(self.frame)
        buttons.grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Button(buttons, text="Добавить", command=on_add).pack(side="left")
        ttk.Button(buttons, text="Редактировать", command=on_edit).pack(side="left", padx=(6, 0))
        ttk.Button(buttons, text="Удалить", command=on_delete).pack(side="left", padx=(6, 0))

    def _column_width(self, column: str) -> int:
        return {
            "name": 210,
            "quantity": 90,
            "price": 120,
            "monthly_cost": 140,
            "one_time_cost": 150,
        }.get(column, 120)

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
