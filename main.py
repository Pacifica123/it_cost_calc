# main.py
import tkinter as tk
from tkinter import ttk
from tabs import capital_costs, operational_costs, it_infrastructure, electricity_costs, export_tab
from crud import CRUD
from utils import TotalCosts

class CalculatorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Калькулятор стоимости ИТ-инфраструктуры")
        self.geometry("1280x720")

        self.crud = CRUD()
        self.total_costs = TotalCosts()

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)

        self.capital_costs_tab = capital_costs.CapitalCostsTab(self.notebook, self.crud)
        self.notebook.add(self.capital_costs_tab, text="Капитальные затраты")

        self.operational_costs_tab = operational_costs.OperationalCostsTab(self.notebook, self.crud)
        self.notebook.add(self.operational_costs_tab, text="Операционные затраты")

        self.it_infrastructure_tab = it_infrastructure.ITInfrastructureTab(self.notebook, self.crud)
        self.notebook.add(self.it_infrastructure_tab, text="ИТ-инфраструктура")

        self.electricity_tab = electricity_costs.ElectricityCostsTab(self.notebook, self.crud)
        self.notebook.add(self.electricity_tab, text="Электроэнергия")

        self.export_tab = export_tab.ExportTab(self.notebook, self)
        self.notebook.add(self.export_tab, text="Экспорт")

        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_change)

        self.preinitialize_data()

    def on_tab_change(self, event):
        if self.notebook.index("current") == self.notebook.index(self.electricity_tab):
            self.electricity_tab.update_equipment_table()
            self.update_total_costs()
        self.export_tab.update_summary()

    def update_total_costs(self):
        self.total_costs.update_capital_costs(self.crud)
        self.total_costs.update_operational_costs(self.crud)

    def export_to_csv(self):
        self.total_costs.update_capital_costs(self.crud)
        self.total_costs.update_operational_costs(self.crud)
        self.total_costs.update_electricity_costs(self.electricity_tab.get_electricity_cost())
        self.total_costs.export_to_csv("total_costs.csv")
        self.export_tab.update_summary()  # Обновите сводку
        print("Данные успешно экспортированы в total_costs.csv")

    def preinitialize_data(self):
        # Примерные данные для автозаполнения
        capital_data = {
            "server": [
                {"name": "Rock w9102p", "quantity": 2, "price": 117192.0},
                {"name": "Rock w9102p2", "quantity": 1, "price": 203070.0},
            ],
            "client": [
                {"name": "Atlas H434", "quantity": 3, "price": 37799.0},
                {"name": "AuBox", "quantity": 1, "price": 54999.0},
                {"name": "RAGE H322", "quantity": 2, "price": 137999.0},
                {"name": "i-SENSYS MF752Cdw", "quantity": 2, "price": 47999.0},
            ],
            "network": [
                {"name": "B3011UiAS-RM", "quantity": 1, "price": 19499.0},
                {"name": "GIGA KN-1012", "quantity": 1, "price": 15199.0},                {"name": "CRS326-24G-2S+RM", "quantity": 2, "price": 25999.0},
            ],
            "licenses": [
                {"name": "MS Windows 11 Pro", "quantity": 1, "price": 19850.0},
            ],
        }

        operational_data = {
            "subscription_licenses": [
                {"name": "MS Visual Studio Enterprise", "monthly_cost": 24815.0},
                {"name": "Microsoft 365", "monthly_cost": 1289.0},
            ],
            "server_rental": [
                # {"name": "Аренда сервера в DataCenter", "monthly_cost": 500.0},
            ],
            "migration": [
                # {"name": "Миграция на новую платформу", "one_time_cost": 10000.0},
            ],
            "testing": [
                # {"name": "Тестирование безопасности", "one_time_cost": 2000.0},
            ],
            "backup": [
                # {"name": "Резервное копирование данных", "monthly_cost": 200.0},
            ],
            "labor_costs": [
                # {"name": "Зарплата администратора", "monthly_cost": 30000.0},
            ],
            "server_administration": [
                # {"name": "Администрирование серверов", "monthly_cost": 5000.0},
            ],
        }

        # Автозаполнение данных
        for entity_name, data_list in capital_data.items():
            for data in data_list:
                self.capital_costs_tab.crud.add(entity_name, data, self.capital_costs_tab.tables[entity_name][1])

        for entity_name, data_list in operational_data.items():
            for data in data_list:
                if entity_name in ["migration", "testing"]:
                    data["one_time_cost"] = float(data["one_time_cost"])
                else:
                    data["monthly_cost"] = float(data["monthly_cost"])
                self.operational_costs_tab.crud.add(entity_name, data, self.operational_costs_tab.tables[entity_name][1])


if __name__ == "__main__":
    app = CalculatorApp()
    app.mainloop()
