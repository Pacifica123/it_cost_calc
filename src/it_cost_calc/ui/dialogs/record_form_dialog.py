from __future__ import annotations

import tkinter as tk
from tkinter import messagebox


class RecordFormDialog(tk.Toplevel):
    def __init__(self, parent, title: str, fields: list[dict], initial_values: dict | None = None):
        super().__init__(parent)
        self.title(title)
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        self.result: dict | None = None
        self._entries: dict[str, tk.Entry] = {}
        self._fields = fields
        self._initial_values = initial_values or {}

        container = tk.Frame(self, padx=12, pady=12)
        container.pack(fill="both", expand=True)

        for row, field in enumerate(fields):
            tk.Label(container, text=field["label"]).grid(row=row, column=0, sticky="w", pady=4)
            entry = tk.Entry(container, width=32)
            entry.grid(row=row, column=1, sticky="ew", pady=4)
            entry.insert(0, str(self._initial_values.get(field["name"], field.get("default", ""))))
            self._entries[field["name"]] = entry

        buttons = tk.Frame(container)
        buttons.grid(row=len(fields), column=0, columnspan=2, sticky="e", pady=(10, 0))
        tk.Button(buttons, text="Сохранить", command=self._on_save).pack(side="left", padx=(0, 8))
        tk.Button(buttons, text="Отмена", command=self.destroy).pack(side="left")

        container.columnconfigure(1, weight=1)
        self.bind("<Return>", lambda _event: self._on_save())
        self.bind("<Escape>", lambda _event: self.destroy())

    def _on_save(self):
        try:
            payload = {}
            for field in self._fields:
                raw_value = self._entries[field["name"]].get()
                parser = field.get("parser", lambda value, _label: value)
                payload[field["name"]] = parser(raw_value, field["label"])
            self.result = payload
            self.destroy()
        except ValueError as error:
            messagebox.showerror("Ошибка ввода", str(error), parent=self)

    def show(self) -> dict | None:
        self.wait_window()
        return self.result
