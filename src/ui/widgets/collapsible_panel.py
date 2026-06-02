from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class CollapsiblePanel(ttk.Frame):
    """Small reusable disclosure panel for dense Tk interfaces.

    The panel keeps a visible header and can hide or show its content area. It is
    deliberately lightweight so it can be used inside ordinary frames as well as
    inside ``ttk.Panedwindow`` containers where users resize neighbouring areas.
    """

    def __init__(
        self,
        parent: tk.Misc,
        *,
        title: str,
        initially_open: bool = True,
        header_hint: str | None = None,
    ) -> None:
        super().__init__(parent, padding=4)
        self.title = title
        self._opened = bool(initially_open)
        self._indicator_var = tk.StringVar()
        self._hint_var = tk.StringVar(value=header_hint or "")

        header = ttk.Frame(self)
        header.pack(fill="x")

        self.toggle_button = ttk.Button(header, command=self.toggle, width=3)
        self.toggle_button.pack(side="left")

        ttk.Label(header, text=title, font=("TkDefaultFont", 10, "bold")).pack(
            side="left", padx=(6, 8)
        )
        if header_hint:
            ttk.Label(header, textvariable=self._hint_var).pack(side="left", fill="x", expand=True)

        self.content = ttk.Frame(self)
        self._sync_visibility()

    @property
    def opened(self) -> bool:
        return self._opened

    def toggle(self) -> None:
        self._opened = not self._opened
        self._sync_visibility()

    def open(self) -> None:
        self._opened = True
        self._sync_visibility()

    def close(self) -> None:
        self._opened = False
        self._sync_visibility()

    def set_hint(self, text: str) -> None:
        self._hint_var.set(text)

    def _sync_visibility(self) -> None:
        self._indicator_var.set("▼" if self._opened else "▶")
        self.toggle_button.configure(text=self._indicator_var.get())
        if self._opened:
            self.content.pack(fill="both", expand=True, pady=(6, 0))
        else:
            self.content.pack_forget()
