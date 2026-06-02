from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class CollapsiblePanel(ttk.Frame):
    """Reusable disclosure panel for dense Tk interfaces.

    The indicator is a compact text arrow instead of a full button.  This keeps
    the affordance visible but prevents the bright platform button from
    dominating dense ПО/ТО workspaces.
    """

    def __init__(
        self,
        parent: tk.Misc,
        *,
        title: str,
        initially_open: bool = True,
        header_hint: str | None = None,
    ) -> None:
        super().__init__(parent, padding=0, style="Panel.TFrame")
        self.title = title
        self._opened = bool(initially_open)
        self._indicator_var = tk.StringVar()
        self._hint_var = tk.StringVar(value=header_hint or "")

        self.header = ttk.Frame(self, padding=(8, 5), style="PanelHeader.TFrame")
        self.header.pack(fill="x")
        self.header.columnconfigure(2, weight=1)

        self.toggle_indicator = ttk.Label(
            self.header,
            textvariable=self._indicator_var,
            style="Disclosure.TLabel",
            width=2,
            anchor="center",
            cursor="hand2",
        )
        self.toggle_indicator.grid(row=0, column=0, sticky="w")

        self.title_label = ttk.Label(
            self.header,
            text=title,
            style="PanelTitle.TLabel",
            cursor="hand2",
        )
        self.title_label.grid(row=0, column=1, sticky="w", padx=(4, 10))

        self.hint_label: ttk.Label | None = None
        if header_hint:
            self.hint_label = ttk.Label(
                self.header,
                textvariable=self._hint_var,
                style="PanelHint.TLabel",
                wraplength=520,
                justify="left",
            )
            self.hint_label.grid(row=0, column=2, sticky="ew")

        self.content = ttk.Frame(self, padding=(8, 8), style="Surface.TFrame")
        self._bind_header_toggle()
        self._sync_visibility()

    @property
    def opened(self) -> bool:
        return self._opened

    def toggle(self, _event=None) -> None:
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
