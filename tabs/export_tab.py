# tabs/export_tab.py
import tkinter as tk
from tkinter import ttk

class ExportTab(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app

        self.export_button = tk.Button(self, text="Экспортировать в CSV", command=self.app.export_to_csv)
        self.export_button.pack(fill="x")

        self.summary_text = tk.Text(self, height=10, width=50)
        self.summary_text.pack(fill="both", expand=True)

        self.update_summary()

    def update_summary(self):
        totals = self.app.total_costs.calculate_totals()
        summary = f"Капитальные затраты: {totals['total_capital']}\n"
        summary += f"Операционные затраты (разовые): {totals['total_operational_one_time']}\n"
        summary += f"Операционные затраты (периодические): {totals['total_operational_monthly']}\n"
        summary += f"Стоимость электроэнергии: {totals['electricity_costs']}\n"
        summary += f"Общая стоимость: {totals['total_capital'] + totals['total_operational_one_time'] + totals['total_operational_monthly'] + totals['electricity_costs']}"

        self.summary_text.delete('1.0', tk.END)
        self.summary_text.insert(tk.END, summary)
