"""Electricity tab using equipment/application services instead of direct CRUD access."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from application.services.equipment_service import EquipmentService
from application.use_cases.calculate_electricity_costs import (
    CalculateElectricityCostsUseCase,
)
from shared.validation import parse_float
from ui.tabs.base_scrollable_tab import BaseScrollableTab


class EnergyTab(BaseScrollableTab):
    def __init__(
        self,
        parent,
        equipment_service: EquipmentService,
        calculate_electricity_costs_use_case: CalculateElectricityCostsUseCase,
    ):
        super().__init__(parent)
        self.equipment_service = equipment_service
        self.calculate_electricity_costs_use_case = calculate_electricity_costs_use_case
        self.total_cost = 0.0
        self.equipment_data: dict[str, dict] = {}

        self.equipment_frame = tk.Frame(self.inner_frame)
        self.equipment_frame.pack(fill="x")

        tk.Label(self.equipment_frame, text="Оборудование").pack()
        self.equipment_table = ttk.Treeview(
            self.equipment_frame,
            columns=("name", "quantity", "max_power"),
            show="headings",
        )
        self.equipment_table.heading("name", text="Наименование")
        self.equipment_table.heading("quantity", text="Количество")
        self.equipment_table.heading("max_power", text="Максимальная мощность (Вт)")
        for column in ("name", "quantity", "max_power"):
            self.equipment_table.column(column, stretch=tk.YES)
        self.equipment_table.pack(fill="x")

        self.buttons_frame = tk.Frame(self.equipment_frame)
        self.buttons_frame.pack(fill="x")
        tk.Button(self.buttons_frame, text="Редактировать", command=self.edit_equipment).pack(
            side="left"
        )
        tk.Button(self.buttons_frame, text="Удалить", command=self.delete_equipment).pack(
            side="left"
        )

        self.calculation_frame = tk.Frame(self.inner_frame)
        self.calculation_frame.pack(fill="x")

        tk.Label(self.calculation_frame, text="Часов в рабочем дне").pack()
        self.hours_per_day_entry = tk.Entry(self.calculation_frame)
        self.hours_per_day_entry.pack()
        self.hours_per_day_entry.insert(0, "8")

        tk.Label(self.calculation_frame, text="Рабочих дней в месяце").pack()
        self.working_days_entry = tk.Entry(self.calculation_frame)
        self.working_days_entry.pack()
        self.working_days_entry.insert(0, "22")

        tk.Label(self.calculation_frame, text="Стоимость 1 кВтч (руб.)").pack()
        self.cost_per_kwh_entry = tk.Entry(self.calculation_frame)
        self.cost_per_kwh_entry.pack()
        self.cost_per_kwh_entry.insert(0, "1")

        self.calculate_button = tk.Button(
            self.calculation_frame, text="Рассчитать затраты", command=self.calculate_costs
        )
        self.calculate_button.pack()

        self.result_label = tk.Label(self.inner_frame, text="Результаты расчета:")
        self.result_label.pack()

        self.update_equipment_table()

    def update_equipment_table(self) -> None:
        self.equipment_table.delete(*self.equipment_table.get_children())

        merged_rows: list[dict] = []
        for row in self.equipment_service.list_energy_relevant_items():
            name = row.name
            payload = {
                "name": name,
                "quantity": float(row.quantity),
                "max_power": float(self.equipment_data.get(name, {}).get("max_power", 0.0)),
            }
            self.equipment_data[name] = payload
            merged_rows.append(payload)

        for row in merged_rows:
            self.equipment_table.insert(
                "", "end", values=(row["name"], row["quantity"], row["max_power"])
            )

        self.update_scrollregion()

    def edit_equipment(self) -> None:
        selected_item = self.equipment_table.focus()
        if not selected_item:
            messagebox.showwarning("Нет выбора", "Выберите строку для редактирования", parent=self)
            return

        values = self.equipment_table.item(selected_item, "values")
        name = str(values[0])
        quantity = str(values[1])
        max_power = str(values[2])

        popup = tk.Toplevel(self)
        popup.title("Параметры мощности")
        tk.Label(popup, text="Наименование:").pack()
        tk.Label(popup, text=name).pack()
        tk.Label(popup, text="Количество:").pack()
        tk.Label(popup, text=quantity).pack()
        tk.Label(popup, text="Максимальная мощность (Вт):").pack()
        max_power_entry = tk.Entry(popup)
        max_power_entry.insert(0, max_power)
        max_power_entry.pack()

        def save_data() -> None:
            try:
                parsed_power = parse_float(max_power_entry.get(), "Максимальная мощность")
            except ValueError as error:
                messagebox.showerror("Ошибка ввода", str(error), parent=popup)
                return

            self.equipment_data[name] = {
                "name": name,
                "quantity": float(quantity),
                "max_power": parsed_power,
            }
            popup.destroy()
            self.update_equipment_table()

        tk.Button(popup, text="Сохранить", command=save_data).pack()

    def delete_equipment(self) -> None:
        selected_item = self.equipment_table.focus()
        if not selected_item:
            messagebox.showwarning("Нет выбора", "Выберите строку для удаления", parent=self)
            return
        name = str(self.equipment_table.item(selected_item, "values")[0])
        self.equipment_data.pop(name, None)
        self.equipment_table.delete(selected_item)

    def calculate_costs(self) -> None:
        try:
            hours_per_day = parse_float(self.hours_per_day_entry.get(), "Часов в рабочем дне")
            working_days = parse_float(self.working_days_entry.get(), "Рабочих дней в месяце")
            cost_per_kwh = parse_float(self.cost_per_kwh_entry.get(), "Стоимость 1 кВтч")
        except ValueError as error:
            messagebox.showerror("Ошибка ввода", str(error), parent=self)
            return

        equipment_rows = []
        for item in self.equipment_table.get_children():
            values = self.equipment_table.item(item, "values")
            equipment_rows.append(
                {
                    "name": str(values[0]),
                    "quantity": float(values[1]),
                    "max_power": float(values[2]),
                }
            )

        report = self.calculate_electricity_costs_use_case.execute(
            equipment_rows=equipment_rows,
            hours_per_day=hours_per_day,
            working_days=working_days,
            cost_per_kwh=cost_per_kwh,
            round_the_clock_names=self.equipment_service.list_round_the_clock_equipment_names(),
        )
        self.total_cost = report["total_cost"]
        self.result_label["text"] = f"Общие затраты на электроэнергию: {self.total_cost:.2f} руб."

    def get_electricity_cost(self) -> float:
        return self.total_cost


ElectricityCostsTab = EnergyTab

__all__ = ["EnergyTab", "ElectricityCostsTab"]
