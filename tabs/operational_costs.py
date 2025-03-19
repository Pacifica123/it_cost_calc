# tabs/operational_costs.py
import tkinter as tk
from tkinter import ttk
from crud import CRUD
import tkinter as tk
from tkinter import ttk

# import table
from crud import CRUD


class OperationalCostsTab(tk.Frame):
    def __init__(self, parent, crud):
        super().__init__(parent)
        self.crud = crud

        # Скроллбар
        self.canvas = tk.Canvas(self, width=585, height=400)
        self.canvas.pack(side="left", fill="both", expand=True)

        self.scrollbar = tk.Scrollbar(self)
        self.scrollbar.pack(side="right", fill="y")

        self.canvas.config(yscrollcommand=self.scrollbar.set)
        self.scrollbar.config(command=self.canvas.yview)

        self.inner_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.inner_frame, anchor='nw')

        self.inner_frame.bind("<Configure>", lambda e: self.update_scrollregion())

        # Настройка столбцов для равномерного распределения
        self.inner_frame.columnconfigure(0, weight=1)
        self.inner_frame.columnconfigure(1, weight=1)

        # Таблицы
        self.tables = {
            "subscription_licenses": self.create_table("Лицензии по подписке", ("name", "monthly_cost")),
            "server_rental": self.create_table("Аренда серверов", ("name", "monthly_cost")),
            "migration": self.create_table("Миграция", ("name", "one_time_cost")),
            "testing": self.create_table("Тестирование", ("name", "one_time_cost")),
            "backup": self.create_table("Резервирование", ("name", "monthly_cost")),
            "labor_costs": self.create_table("Оплата труда", ("name", "monthly_cost")),
            "server_administration": self.create_table("Администрирование серверов", ("name", "monthly_cost")),
        }

        # Располагаем таблицы в два столбца
        for i, (entity_name, (frame, table, buttons_frame)) in enumerate(self.tables.items()):
            row = i // 3
            column = i % 3
            frame.grid(row=row, column=column, padx=10, pady=10, sticky="nsew")

            tk.Button(buttons_frame, text="Добавить", command=lambda e=entity_name, t=table: self.add_row(e, t)).pack(
                side="left")
            tk.Button(buttons_frame, text="Редактировать",
                      command=lambda e=entity_name, t=table: self.edit_row(e, t)).pack(side="left")
            tk.Button(buttons_frame, text="Удалить", command=lambda e=entity_name, t=table: self.delete_row(e, t)).pack(
                side="left")

        self.update_scrollregion()

    def create_table(self, label_text, columns):
        frame = tk.Frame(self.inner_frame)
        tk.Label(frame, text=label_text).pack()

        column_names = {
            "name": "Наименование",
            "monthly_cost": "Ежемесячная стоимость",
            "one_time_cost": "Единовременная стоимость"
        }

        table = ttk.Treeview(frame, columns=columns, show="headings")
        for column in columns:
            table.column(column, stretch=tk.YES)
            table.heading(column, text=column_names.get(column, column))
        table.pack(fill="x")

        buttons_frame = tk.Frame(frame)
        buttons_frame.pack(fill="x")

        return frame, table, buttons_frame

    def add_row(self, entity_name, table):
        def save_data():
            data = {"name": name_entry.get(),
                    "one_time_cost" if entity_name in ["migration", "testing"] else "monthly_cost": float(
                        cost_entry.get())}
            self.crud.add(entity_name, data, table)
            popup.destroy()
            self.update_scrollregion()

        popup = tk.Toplevel(self)
        tk.Label(popup, text="Наименование").pack()
        name_entry = tk.Entry(popup)
        name_entry.pack()
        tk.Label(popup, text="Стоимость").pack()
        cost_entry = tk.Entry(popup)
        cost_entry.pack()
        tk.Button(popup, text="Сохранить", command=save_data).pack()

    def edit_row(self, entity_name, table):
        def save_data():
            index = selected_index
            if entity_name in ["migration", "testing"]:
                data = {
                    "name": name_entry.get(),
                    "one_time_cost": float(cost_entry.get())
                }
            else:
                data = {
                    "name": name_entry.get(),
                    "monthly_cost": float(cost_entry.get())
                }
            self.crud.update(entity_name, index, data, table)
            popup.destroy()

        selected_item = table.focus()
        selected_index = list(table.get_children()).index(selected_item)

        if selected_item:
            popup = tk.Toplevel(self)
            tk.Label(popup, text="Наименование").pack()
            name_entry = tk.Entry(popup)
            name_entry.insert(0, self.crud.entities[entity_name][selected_index]["name"])
            name_entry.pack()
            tk.Label(popup, text="Стоимость").pack()
            cost_entry = tk.Entry(popup)
            if entity_name in ["migration", "testing"]:
                cost_entry.insert(0, self.crud.entities[entity_name][selected_index]["one_time_cost"])
            else:
                cost_entry.insert(0, self.crud.entities[entity_name][selected_index]["monthly_cost"])
            cost_entry.pack()
            tk.Button(popup, text="Сохранить", command=save_data).pack()
        else:
            print("Выберите строку для редактирования")

    def delete_row(self, entity_name, table):
        selected_item = table.focus()
        selected_index = list(table.get_children()).index(selected_item)
        self.crud.delete(entity_name, selected_index, table)
        self.update_scrollregion()

    def update_scrollregion(self):
        self.inner_frame.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

