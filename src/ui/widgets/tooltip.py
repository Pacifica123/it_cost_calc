"""Small tooltip helper for classic Tk/ttk widgets.

Tkinter does not ship a dedicated tooltip widget, but a borderless ``Toplevel``
works reliably for lightweight contextual help without adding dependencies.
"""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from typing import TypeAlias

from ui.theme import PANEL_BORDER, TEXT

TooltipTextSource: TypeAlias = str | tk.Variable | Callable[[], str]


class ToolTip:
    """Attach delayed contextual help to a Tk widget."""

    def __init__(
        self,
        widget: tk.Misc,
        text: TooltipTextSource,
        *,
        delay_ms: int = 450,
        wraplength: int = 360,
    ) -> None:
        self.widget = widget
        self.text_source = text
        self.delay_ms = delay_ms
        self.wraplength = wraplength
        self._after_id: str | None = None
        self._window: tk.Toplevel | None = None

        if not self._resolve_text():
            return

        self.widget.bind("<Enter>", self._schedule_show, add="+")
        self.widget.bind("<Leave>", self._hide, add="+")
        self.widget.bind("<ButtonPress>", self._hide, add="+")
        self.widget.bind("<Destroy>", self._destroy, add="+")

    def _resolve_text(self) -> str:
        source = self.text_source
        try:
            if isinstance(source, tk.Variable):
                return str(source.get()).strip()
            if callable(source):
                return str(source()).strip()
            return str(source).strip()
        except tk.TclError:
            return ""

    def _schedule_show(self, _event: tk.Event | None = None) -> None:
        self._cancel_scheduled_show()
        self._after_id = self.widget.after(self.delay_ms, self._show)

    def _show(self) -> None:
        self._after_id = None
        try:
            if self._window is not None or not self.widget.winfo_exists():
                return
            x = self.widget.winfo_rootx() + 18
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8
            self._window = tk.Toplevel(self.widget)
            self._window.withdraw()
            self._window.wm_overrideredirect(True)
            self._window.wm_geometry(f"+{x}+{y}")
            try:
                self._window.attributes("-topmost", True)
            except tk.TclError:
                pass
        except tk.TclError:
            self._window = None
            return

        text = self._resolve_text()
        if not text:
            self._hide()
            return

        label = tk.Label(
            self._window,
            text=text,
            justify="left",
            wraplength=self.wraplength,
            background="#fffbe6",
            foreground=TEXT,
            relief="solid",
            borderwidth=1,
            highlightbackground=PANEL_BORDER,
            padx=8,
            pady=6,
        )
        label.pack()
        self._window.deiconify()

    def _hide(self, _event: tk.Event | None = None) -> None:
        self._cancel_scheduled_show()
        if self._window is None:
            return
        try:
            self._window.destroy()
        except tk.TclError:
            pass
        finally:
            self._window = None

    def _destroy(self, _event: tk.Event | None = None) -> None:
        self._hide()

    def _cancel_scheduled_show(self) -> None:
        if self._after_id is None:
            return
        try:
            self.widget.after_cancel(self._after_id)
        except tk.TclError:
            pass
        finally:
            self._after_id = None


def attach_tooltip(
    widget: tk.Misc,
    text: TooltipTextSource,
    *,
    delay_ms: int = 450,
    wraplength: int = 360,
) -> tk.Misc:
    """Attach tooltip behavior and return the original widget for fluent UI code."""

    tooltip = ToolTip(widget, text, delay_ms=delay_ms, wraplength=wraplength)
    setattr(widget, "_tooltip", tooltip)
    return widget
