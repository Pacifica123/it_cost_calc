import tkinter as tk
from tkinter import ttk
from crud import CRUD
import pprint


class CapitalCostsTab(tk.Frame):
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

        self.inner_frame.columnconfigure(0, weight=1)
        self.inner_frame.columnconfigure(1, weight=1)

        # Таблицы
        self.tables = {
            "server": self.create_table("Серверное оборудование", ("name", "quantity", "price")),
            "client": self.create_table("Клиентское оборудование", ("name", "quantity", "price")),
            "network": self.create_table("Сетевое оборудование", ("name", "quantity", "price")),
            "licenses": self.create_table("Лицензии ПО", ("name", "quantity", "price")),
        }

        # Располагаем таблицы в два столбца
        for i, (entity_name, (frame, table, buttons_frame)) in enumerate(self.tables.items()):
            row = i // 2
            column = i % 2
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

        # Словарь для перевода названий столбцов
        column_names = {
            "name": "Наименование",
            "quantity": "Количество",
            "price": "Цена"
        }

        table = ttk.Treeview(frame, columns=columns, show="headings")
        for column in columns:
            table.column(column, stretch=tk.YES)
            table.heading(column, text=column_names[column])  # Используем переведенные названия
        table.pack(fill="x")

        buttons_frame = tk.Frame(frame)
        buttons_frame.pack(fill="x")

        return frame, table, buttons_frame

    def add_row(self, entity_name, table):
        def save_data():
            data = {
                "name": name_entry.get(),
                "quantity": int(quantity_entry.get()),
                "price": float(price_entry.get())
            }
            self.crud.add(entity_name, data, table)
            popup.destroy()
            self.update_scrollregion()

        popup = tk.Toplevel(self)
        tk.Label(popup, text="Наименование").pack()
        name_entry = tk.Entry(popup)
        name_entry.pack()
        tk.Label(popup, text="Количество").pack()
        quantity_entry = tk.Entry(popup)
        quantity_entry.pack()
        tk.Label(popup, text="Цена").pack()
        price_entry = tk.Entry(popup)
        price_entry.pack()
        tk.Button(popup, text="Сохранить", command=save_data).pack()

    def edit_row(self, entity_name, table):
        def save_data():
            index = selected_index
            data = {
                "name": name_entry.get(),
                "quantity": int(quantity_entry.get()),
                "price": float(price_entry.get())
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
            tk.Label(popup, text="Количество").pack()
            quantity_entry = tk.Entry(popup)
            quantity_entry.insert(0, self.crud.entities[entity_name][selected_index]["quantity"])
            quantity_entry.pack()
            tk.Label(popup, text="Цена").pack()
            price_entry = tk.Entry(popup)
            price_entry.insert(0, self.crud.entities[entity_name][selected_index]["price"])
            price_entry.pack()
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

