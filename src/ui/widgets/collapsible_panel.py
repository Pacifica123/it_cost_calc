from __future__ import annotations

import tkinter as tk
from collections.abc import Callable

from ui.theme import MUTED_TEXT, PANEL, PANEL_BODY, PANEL_BORDER, PANEL_HEADER, TEXT


class CollapsiblePanel(tk.Frame):
    """Reusable disclosure panel for dense Tk interfaces.

    The header and body use classic Tk widgets with explicit backgrounds.  This
    avoids a Windows-specific ttk repaint issue where the panel briefly appears
    with the intended palette and then returns to white after the whole
    application finishes drawing.
    """

    def __init__(
        self,
        parent: tk.Misc,
        *,
        title: str,
        initially_open: bool = True,
        header_hint: str | None = None,
        on_toggle: Callable[["CollapsiblePanel"], None] | None = None,
    ) -> None:
        super().__init__(
            parent,
            background=PANEL,
            highlightbackground=PANEL_BORDER,
            highlightcolor=PANEL_BORDER,
            highlightthickness=1,
            bd=0,
        )
        self.title = title
        self._opened = bool(initially_open)
        self._on_toggle = on_toggle
        self._indicator_var = tk.StringVar()
        self._hint_var = tk.StringVar(value=header_hint or "")

        self.header = tk.Frame(self, background=PANEL_HEADER, padx=10, pady=6)
        self.header.pack(fill="x")
        self.header.columnconfigure(2, weight=1)

        self.toggle_indicator = tk.Label(
            self.header,
            textvariable=self._indicator_var,
            width=2,
            anchor="center",
            cursor="hand2",
            background=PANEL_HEADER,
            foreground=MUTED_TEXT,
            font=("TkDefaultFont", 8, "bold"),
        )
        self.toggle_indicator.grid(row=0, column=0, sticky="w")

        self.title_label = tk.Label(
            self.header,
            text=title,
            cursor="hand2",
            background=PANEL_HEADER,
            foreground=TEXT,
            font=("TkDefaultFont", 9, "bold"),
        )
        self.title_label.grid(row=0, column=1, sticky="w", padx=(4, 10))

        self.hint_label: tk.Label | None = None
        if header_hint:
            self.hint_label = tk.Label(
                self.header,
                textvariable=self._hint_var,
                wraplength=520,
                justify="left",
                cursor="hand2",
                background=PANEL_HEADER,
                foreground=MUTED_TEXT,
                font=("TkDefaultFont", 8),
            )
            self.hint_label.grid(row=0, column=2, sticky="ew")

        self.content = tk.Frame(self, background=PANEL_BODY, padx=10, pady=10)
        self._bind_header_toggle()
        self._sync_visibility()

    @property
    def opened(self) -> bool:
        return self._opened

    def toggle(self, _event=None) -> None:
        self._opened = not self._opened
        self._sync_visibility()
        self._notify_toggle()

    def open(self) -> None:
        if self._opened:
            return
        self._opened = True
        self._sync_visibility()
        self._notify_toggle()

    def close(self) -> None:
        if not self._opened:
            return
        self._opened = False
        self._sync_visibility()
        self._notify_toggle()

    def set_hint(self, text: str) -> None:
        self._hint_var.set(text)

    def set_toggle_callback(self, callback: Callable[["CollapsiblePanel"], None] | None) -> None:
        self._on_toggle = callback

    def _notify_toggle(self) -> None:
        if self._on_toggle is not None:
            self._on_toggle(self)

    def _bind_header_toggle(self) -> None:
        for widget in (self.header, self.toggle_indicator, self.title_label):
            widget.bind("<Button-1>", self.toggle)
        if self.hint_label is not None:
            self.hint_label.bind("<Button-1>", self.toggle)

    def _sync_visibility(self) -> None:
        self._indicator_var.set("▾" if self._opened else "▸")
        if self._opened:
            self.content.pack(fill="both", expand=True, padx=1, pady=(0, 1))
        else:
            self.content.pack_forget()
