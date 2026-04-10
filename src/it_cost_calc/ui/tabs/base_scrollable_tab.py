from __future__ import annotations

import tkinter as tk


class BaseScrollableTab(tk.Frame):
    def __init__(self, parent, *, width: int = 585, height: int = 400):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, width=width, height=height)
        self.canvas.pack(side="left", fill="both", expand=True)

        self.scrollbar = tk.Scrollbar(self)
        self.scrollbar.pack(side="right", fill="y")

        self.canvas.config(yscrollcommand=self.scrollbar.set)
        self.scrollbar.config(command=self.canvas.yview)

        self.inner_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.inner_frame, anchor="nw")
        self.inner_frame.bind("<Configure>", lambda _event: self.update_scrollregion())

    def update_scrollregion(self) -> None:
        self.inner_frame.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
