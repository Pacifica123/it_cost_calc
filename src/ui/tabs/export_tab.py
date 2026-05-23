"""Export tab consumes application use cases instead of formatting data inline."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from shared.formatting import format_cost_summary
from ui.tabs.base_scrollable_tab import BaseScrollableTab


class ExportTab(BaseScrollableTab):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        root = self.inner_frame

        self.export_button = tk.Button(root, text="Экспортировать в CSV", command=self._export)
        self.export_button.pack(fill="x")

        self.decision_report_button = tk.Button(
            root,
            text="Сформировать DecisionReport JSON/Markdown/CSV",
            command=self._export_decision_report,
        )
        self.decision_report_button.pack(fill="x", pady=(6, 0))

        self.summary_text = tk.Text(root, height=10, width=50)
        self.summary_text.pack(fill="both", expand=True)

        self.update_summary()

    def _export(self) -> None:
        self.app.export_to_csv()
        self.update_summary()

    def _export_decision_report(self) -> None:
        try:
            paths = self.app.export_decision_report()
            self.update_summary()
            self.summary_text.insert(
                tk.END,
                "\n\nDecisionReport сформирован:"
                f"\n- JSON: {paths.get('json')}"
                f"\n- Markdown: {paths.get('markdown')}"
                f"\n- CSV: {paths.get('csv')}",
            )
            messagebox.showinfo(
                "DecisionReport",
                "Итоговый отчёт выбора сформирован в data/generated.",
                parent=self,
            )
        except Exception as error:
            messagebox.showerror("Ошибка DecisionReport", str(error), parent=self)

    def update_summary(self) -> None:
        totals = self.app.update_total_costs()
        summary = format_cost_summary(totals)
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert(tk.END, summary)
