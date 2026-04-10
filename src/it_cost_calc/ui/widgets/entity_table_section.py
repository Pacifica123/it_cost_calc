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
        self.frame = tk.Frame(parent)
        tk.Label(self.frame, text=title).pack()

        self.table = ttk.Treeview(self.frame, columns=columns, show="headings")
        for column in columns:
            self.table.column(column, stretch=tk.YES)
            self.table.heading(column, text=column_names.get(column, column))
        self.table.pack(fill="x")

        buttons = tk.Frame(self.frame)
        buttons.pack(fill="x")
        tk.Button(buttons, text="Добавить", command=on_add).pack(side="left")
        tk.Button(buttons, text="Редактировать", command=on_edit).pack(side="left")
        tk.Button(buttons, text="Удалить", command=on_delete).pack(side="left")

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
