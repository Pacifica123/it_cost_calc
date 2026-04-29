from __future__ import annotations

import tkinter as tk


class BaseScrollableTab(tk.Frame):
    """Base frame that gives a tab vertical scrolling by default.

    Child widgets should be placed inside ``self.inner_frame``.  The canvas is
    intentionally lightweight and works with ordinary Tk/ttk widgets, so the
    same base can be used by data-entry tabs, analytical tabs and result tabs.
    """

    def __init__(self, parent, *, width: int = 585, height: int = 400):
        super().__init__(parent)

        self.canvas = tk.Canvas(self, width=width, height=height, highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)

        self.scrollbar = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.inner_frame = tk.Frame(self.canvas)
        self._inner_window_id = self.canvas.create_window(
            (0, 0), window=self.inner_frame, anchor="nw"
        )

        self.inner_frame.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.bind("<Enter>", self._bind_mousewheel)
        self.bind("<Leave>", self._unbind_mousewheel)
        self.inner_frame.bind("<Enter>", self._bind_mousewheel)
        self.inner_frame.bind("<Leave>", self._unbind_mousewheel)

    def _on_inner_configure(self, _event=None) -> None:
        self.update_scrollregion()

    def _on_canvas_configure(self, event) -> None:
        self.canvas.itemconfigure(self._inner_window_id, width=event.width)
        self.update_scrollregion()

    def _bind_mousewheel(self, _event=None) -> None:
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel_linux)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel_linux)

    def _unbind_mousewheel(self, _event=None) -> None:
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_mousewheel(self, event) -> None:
        if self._can_scroll():
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_mousewheel_linux(self, event) -> None:
        if not self._can_scroll():
            return
        direction = -1 if event.num == 4 else 1
        self.canvas.yview_scroll(direction, "units")

    def _can_scroll(self) -> bool:
        bbox = self.canvas.bbox("all")
        return bool(bbox and (bbox[3] - bbox[1]) > self.canvas.winfo_height())

    def update_scrollregion(self) -> None:
        self.inner_frame.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
