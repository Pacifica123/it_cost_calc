"""Export tab with live context and full/partial export modes."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from ui.tabs.base_scrollable_tab import BaseScrollableTab


EXPORT_MODE_LABELS = {
    "full_decision_report": "Полный DecisionReport: JSON/Markdown/CSV",
    "cost_summary": "Только CAPEX/OPEX/TCO с оговорками",
    "technical_analysis": "Только аналитика ТО: GA/AHP/Pareto",
    "software_analysis": "Только аналитика ПО: GA/AHP/Pareto",
    "npv": "Только NPV с оговорками",
    "raw_costs_csv": "Сырой CSV затрат",
}


class ExportTab(BaseScrollableTab):
    def __init__(self, parent, app):
        super().__init__(parent, width=1180, height=620)
        self.app = app
        self.export_mode_var = tk.StringVar(value="full_decision_report")
        self._build_ui()
        self.update_summary()

    def _build_ui(self) -> None:
        root = self.inner_frame
        root.columnconfigure(0, weight=1)

        control_box = ttk.LabelFrame(root, text="Экспорт")
        control_box.pack(fill="x", padx=10, pady=10)
        control_box.columnconfigure(1, weight=1)

        ttk.Label(control_box, text="Что экспортировать:").grid(
            row=0, column=0, padx=6, pady=6, sticky="w"
        )
        self.mode_combo = ttk.Combobox(
            control_box,
            textvariable=self.export_mode_var,
            values=list(EXPORT_MODE_LABELS),
            state="readonly",
            width=28,
        )
        self.mode_combo.grid(row=0, column=1, padx=6, pady=6, sticky="ew")
        self.mode_combo.bind("<<ComboboxSelected>>", lambda _event: self.update_summary())

        self.mode_label = ttk.Label(control_box, wraplength=880, justify="left")
        self.mode_label.grid(row=1, column=0, columnspan=3, padx=6, pady=(0, 6), sticky="ew")

        ttk.Button(
            control_box,
            text="Экспортировать выбранный фрагмент",
            command=self._export_selected,
        ).grid(row=0, column=2, padx=6, pady=6, sticky="e")

        ttk.Button(
            control_box,
            text="Обновить сводку",
            command=self.update_summary,
        ).grid(row=2, column=0, padx=6, pady=(0, 6), sticky="w")
        ttk.Button(
            control_box,
            text="Открыть папку с отчётами",
            command=self._open_reports_folder,
        ).grid(row=2, column=2, padx=6, pady=(0, 6), sticky="e")

        self.summary_text = tk.Text(root, height=28, width=90, wrap="word", background="#ffffff")
        self.summary_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def _export_selected(self) -> None:
        mode = self.export_mode_var.get()
        try:
            paths = self.app.export_selected_fragment(mode)
            self.update_summary()
            self.summary_text.insert(
                tk.END,
                "\n\nЭкспорт выполнен:\n" + self._format_paths(paths),
            )
            messagebox.showinfo(
                "Экспорт",
                "Файл(ы) сформированы в data/generated.",
                parent=self,
            )
        except Exception as error:
            messagebox.showerror("Ошибка экспорта", str(error), parent=self)

    def _format_paths(self, paths: dict) -> str:
        return "\n".join(f"- {key}: {value}" for key, value in paths.items())

    def _open_reports_folder(self) -> None:
        try:
            path = self.app.open_generated_reports_folder()
            self.summary_text.insert(tk.END, f"\n\nОткрыта папка отчётов: {path}")
        except Exception as error:
            messagebox.showerror("Папка отчётов", str(error), parent=self)

    def update_summary(self) -> None:
        mode = self.export_mode_var.get()
        self.mode_label.configure(text=EXPORT_MODE_LABELS.get(mode, mode))
        summary = self.app.build_export_dashboard_summary()
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert(tk.END, summary)
