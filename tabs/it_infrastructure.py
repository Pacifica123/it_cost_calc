# tabs/it_infrastructure.py
import tkinter as tk
from tkinter import ttk
from crud import CRUD

class ITInfrastructureTab(tk.Frame):
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

        # Кнопка для создания новой статьи
        self.create_article_button = tk.Button(self.inner_frame, text="Создать статью", command=self.create_article)
        self.create_article_button.pack(fill="x")

        self.inner_frame.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

        self.tables = {}

    def create_article(self):
        def save_article():
            article_name = name_entry.get()
            consider_quantity = quantity_var.get()
            expense_type = expense_type_var.get()

            # Определение столбцов для таблицы
            columns = ["name"]
            if consider_quantity:
                columns.append("quantity")
            if expense_type == "periodic":
                columns.append("monthly_cost")
            elif expense_type == "one_time":
                columns.append("one_time_cost")

            # Создание новой статьи в интерфейсе
            new_table = self.create_table(self.inner_frame, article_name, columns)
            self.tables[article_name] = new_table

            # Добавление кнопок для новой статьи
            tk.Button(new_table[2], text="Добавить", command=lambda table=new_table[1]: self.add_row(article_name, table)).pack(side="left")
            tk.Button(new_table[2], text="Редактировать", command=lambda table=new_table[1]: self.edit_row(article_name, table)).pack(side="left")
            tk.Button(new_table[2], text="Удалить", command=lambda table=new_table[1]: self.delete_row(article_name, table)).pack(side="left")

            self.inner_frame.update_idletasks()
            self.canvas.config(scrollregion=self.canvas.bbox("all"))

            popup.destroy()

        popup = tk.Toplevel(self)
        tk.Label(popup, text="Наименование статьи").pack()
        name_entry = tk.Entry(popup)
        name_entry.pack()

        tk.Label(popup, text="Учитывать количество").pack()
        quantity_var = tk.BooleanVar()
        tk.Checkbutton(popup, variable=quantity_var).pack()

        tk.Label(popup, text="Тип расходов").pack()
        expense_type_var = tk.StringVar()
        expense_type_var.set("periodic")  # Default value
        tk.OptionMenu(popup, expense_type_var, "periodic", "one_time").pack()

        tk.Button(popup, text="Сохранить", command=save_article).pack()

    def create_table(self, parent, label_text, columns):
        frame = tk.Frame(parent)
        frame.pack(fill="x")
        tk.Label(frame, text=label_text).pack()

        column_names = {
            "name": "Наименование",
            "quantity": "Количество",
            "monthly_cost": "Ежемесячная стоимость",
            "one_time_cost": "Разовая стоимость"
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
            data = {}
            for column in table["columns"]:
                if column == "name":
                    data[column] = name_entry.get()
                elif column == "quantity":
                    data[column] = int(quantity_entry.get())
                elif column == "monthly_cost":
                    data[column] = float(monthly_cost_entry.get())
                elif column == "one_time_cost":
                    data[column] = float(one_time_cost_entry.get())

            self.crud.add(entity_name, data, table)
            popup.destroy()

        popup = tk.Toplevel(self)
        for column in table["columns"]:
            if column == "name":
                tk.Label(popup, text="Наименование").pack()
                name_entry = tk.Entry(popup)
                name_entry.pack()
            elif column == "quantity":
                tk.Label(popup, text="Количество").pack()
                quantity_entry = tk.Entry(popup)
                quantity_entry.pack()
            elif column == "monthly_cost":
                tk.Label(popup, text="Ежемесячная стоимость").pack()
                monthly_cost_entry = tk.Entry(popup)
                monthly_cost_entry.pack()
            elif column == "one_time_cost":
                tk.Label(popup, text="Разовая стоимость").pack()
                one_time_cost_entry = tk.Entry(popup)
                one_time_cost_entry.pack()

        tk.Button(popup, text="Сохранить", command=save_data).pack()

    def edit_row(self, entity_name, table):
        def save_data():
            index = selected_index
            data = {}
            for column in table["columns"]:
                if column == "name":
                    data[column] = name_entry.get()
                elif column == "quantity":
                    data[column] = int(quantity_entry.get())
                elif column == "monthly_cost":
                    data[column] = float(monthly_cost_entry.get())
                elif column == "one_time_cost":
                    data[column] = float(one_time_cost_entry.get())

            self.crud.update(entity_name, index, data, table)
            popup.destroy()

        selected_item = table.focus()
        selected_index = list(table.get_children()).index(selected_item)

        if selected_item:
            popup = tk.Toplevel(self)
            for column in table["columns"]:
                if column == "name":
                    tk.Label(popup, text="Наименование").pack()
                    name_entry = tk.Entry(popup)
                    name_entry.insert(0, self.crud.entities[entity_name][selected_index]["name"])
                    name_entry.pack()
                elif column == "quantity":
                    tk.Label(popup, text="Количество").pack()
                    quantity_entry = tk.Entry(popup)
                    quantity_entry.insert(0, self.crud.entities[entity_name][selected_index]["quantity"])
                    quantity_entry.pack()
                elif column == "monthly_cost":
                    tk.Label(popup, text="Ежемесячная стоимость").pack()
                    monthly_cost_entry = tk.Entry(popup)
                    monthly_cost_entry.insert(0, self.crud.entities[entity_name][selected_index]["monthly_cost"])
                    monthly_cost_entry.pack()
                elif column == "one_time_cost":
                    tk.Label(popup, text="Разовая стоимость").pack()
                    one_time_cost_entry = tk.Entry(popup)
                    one_time_cost_entry.insert(0, self.crud.entities[entity_name][selected_index]["one_time_cost"])
                    one_time_cost_entry.pack()

            tk.Button(popup, text="Сохранить", command=save_data).pack()
        else:
            print("Выберите строку для редактирования")

    def delete_row(self, entity_name, table):
        selected_item = table.focus()
        selected_index = list(table.get_children()).index(selected_item)
        self.crud.delete(entity_name, selected_index, table)
