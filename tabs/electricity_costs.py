# tabs/electricity_costs.py
import tkinter as tk
from tkinter import ttk
from crud import CRUD
import pprint

class ElectricityCostsTab(tk.Frame):
    def __init__(self, parent, crud):
        super().__init__(parent)

        self.crud = crud

        self.total_cost = 0

        # Скроллбар
        self.canvas = tk.Canvas(self, width=585, height=400)
        self.canvas.pack(side="left", fill="both", expand=True)

        self.scrollbar = tk.Scrollbar(self)
        self.scrollbar.pack(side="right", fill="y")

        self.canvas.config(yscrollcommand=self.scrollbar.set)
        self.scrollbar.config(command=self.canvas.yview)

        self.inner_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.inner_frame, anchor='nw')

        # Таблица для оборудования
        self.equipment_frame = tk.Frame(self.inner_frame)
        self.equipment_frame.pack(fill="x")

        tk.Label(self.equipment_frame, text="Оборудование").pack()
        self.equipment_table = ttk.Treeview(self.equipment_frame, columns=("name", "quantity", "max_power"), show="headings")
        self.equipment_table.column("name", stretch=tk.YES)
        self.equipment_table.column("quantity", stretch=tk.YES)
        self.equipment_table.column("max_power", stretch=tk.YES)
        self.equipment_table.heading("name", text="Наименование")
        self.equipment_table.heading("quantity", text="Количество")
        self.equipment_table.heading("max_power", text="Максимальная мощность (Вт)")
        self.equipment_table.pack(fill="x")

        # Кнопки для редактирования и удаления записей
        self.buttons_frame = tk.Frame(self.equipment_frame)
        self.buttons_frame.pack(fill="x")
        tk.Button(self.buttons_frame, text="Редактировать", command=self.edit_equipment).pack(side="left")
        tk.Button(self.buttons_frame, text="Удалить", command=self.delete_equipment).pack(side="left")

        # Поля для расчета затрат
        self.calculation_frame = tk.Frame(self.inner_frame)
        self.calculation_frame.pack(fill="x")

        tk.Label(self.calculation_frame, text="Часов в рабочем дне").pack()
        self.hours_per_day_entry = tk.Entry(self.calculation_frame)
        self.hours_per_day_entry.pack()

        tk.Label(self.calculation_frame, text="Рабочих дней в месяце").pack()
        self.working_days_entry = tk.Entry(self.calculation_frame)
        self.working_days_entry.pack()

        tk.Label(self.calculation_frame, text="Стоимость 1 кВтч (руб.)").pack()
        self.cost_per_kwh_entry = tk.Entry(self.calculation_frame)
        self.cost_per_kwh_entry.pack()

        # Кнопка для расчета затрат
        self.calculate_button = tk.Button(self.calculation_frame, text="Рассчитать затраты", command=self.calculate_costs)
        self.calculate_button.pack()

        # Поле для вывода результатов
        self.result_label = tk.Label(self.inner_frame, text="Результаты расчета:")
        self.result_label.pack()

        self.inner_frame.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

        # Подтягиваем данные из CapitalCostsTab
        self.update_equipment_table()

        self.equipment_data = {}

    def update_equipment_table(self):
        print(" => СЮДА ЗАШЛО")
        for item in self.equipment_table.get_children():
            self.equipment_table.delete(item)

        for entity_name in ["server", "client", "network"]:
            if entity_name in self.crud.entities:
                for row in self.crud.entities[entity_name]:
                    if "quantity" in row:
                        name = row["name"]
                        quantity = row["quantity"]
                        max_power = 0  # По умолчанию
                        self.equipment_data[name] = {"quantity": quantity, "max_power": max_power}
                        self.equipment_table.insert("", "end", values=(name, quantity, max_power))

    def edit_equipment(self):
        def save_data():
            index = selected_index
            max_power = int(max_power_entry.get())
            self.equipment_data[name]["max_power"] = max_power
            self.equipment_table.delete(selected_item)
            self.equipment_table.insert("", index, values=(name, quantity, max_power))
            popup.destroy()

        selected_item = self.equipment_table.focus()
        selected_index = list(self.equipment_table.get_children()).index(selected_item)

        if selected_item:
            popup = tk.Toplevel(self)
            values = self.equipment_table.item(selected_item, "values")
            name = values[0]
            quantity = values[1]
            max_power = values[2]

            tk.Label(popup, text="Наименование:").pack()
            tk.Label(popup, text=name).pack()
            tk.Label(popup, text="Количество:").pack()
            tk.Label(popup, text=quantity).pack()
            tk.Label(popup, text="Максимальная мощность (Вт):").pack()
            max_power_entry = tk.Entry(popup)
            max_power_entry.insert(0, max_power)
            max_power_entry.pack()
            tk.Button(popup, text="Сохранить", command=save_data).pack()
        else:
            print("Выберите строку для редактирования")

    def delete_equipment(self):
        selected_item = self.equipment_table.focus()
        selected_index = list(self.equipment_table.get_children()).index(selected_item)
        name = self.equipment_table.item(selected_item, "values")[0]
        del self.equipment_data[name]
        self.equipment_table.delete(selected_item)

    def calculate_costs(self):
        total_cost = 0


        cost_per_kwh = float(self.cost_per_kwh_entry.get())

        for item in self.equipment_table.get_children():
            hours_per_day = float(self.hours_per_day_entry.get())
            working_days = float(self.working_days_entry.get())
            values = self.equipment_table.item(item, "values")
            name = values[0]
            quantity = float(values[1])
            max_power = float(values[2])
            print(f"кол-во {quantity}; итого мощность = {quantity*max_power} Вт")
            print(f"hours_per_day = {hours_per_day}")
            print(" --- варианты --- ")
            pprint.pprint(self.crud.entities["server"])
            print(" --- -------- --- ")
            if any(obj.get('name') == name for obj in self.crud.entities["server"]):
            # if name.startswith("Сервер"):
                # Серверное оборудование работает круглосуточно
                hours_per_day = 24
                working_days = 30

            energy_consumption = (max_power / 1000) * quantity * hours_per_day * working_days
            print(f"Потребляемая мощность у-ва {name}: {energy_consumption}")

            cost = energy_consumption * cost_per_kwh
            print(f"цена за рабочий месяц - {cost}")
            total_cost += cost

            print(f"Затраты на {name}: {cost} руб.")

        self.total_cost = total_cost
        self.result_label['text'] = f"Общие затраты на электроэнергию: {total_cost} руб."

    def get_electricity_cost(self):
        return self.total_cost
