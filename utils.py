# utils.py
import csv

class TotalCosts:
    def __init__(self):
        self.capital_costs = {}
        self.operational_costs = {}
        self.electricity_costs = 0

    def update_capital_costs(self, crud):
        self.capital_costs = {
            "server": crud.entities.get("server", []),
            "client": crud.entities.get("client", []),
            "network": crud.entities.get("network", []),
            "licenses": crud.entities.get("licenses", []),
        }

    def update_operational_costs(self, crud):
        self.operational_costs = {
            "subscription_licenses": crud.entities.get("subscription_licenses", []),
            "server_rental": crud.entities.get("server_rental", []),
            "migration": crud.entities.get("migration", []),
            "testing": crud.entities.get("testing", []),
            "backup": crud.entities.get("backup", []),
            "labor_costs": crud.entities.get("labor_costs", []),
            "server_administration": crud.entities.get("server_administration", []),
        }

    def update_electricity_costs(self, electricity_cost):
        self.electricity_costs = electricity_cost

    def calculate_totals(self):
        total_capital = sum(item["quantity"] * item["price"] for items in self.capital_costs.values() for item in items)
        total_operational_one_time = sum(item["one_time_cost"] for items in self.operational_costs.values() for item in items if "one_time_cost" in item)
        total_operational_monthly = sum(item["monthly_cost"] for items in self.operational_costs.values() for item in items if "monthly_cost" in item)

        return {
            "total_capital": total_capital,
            "total_operational_one_time": total_operational_one_time,
            "total_operational_monthly": total_operational_monthly,
            "electricity_costs": self.electricity_costs,
        }

    # def export_to_csv(self, filename):
    #     with open(filename, 'w', newline='') as csvfile:
    #         fieldnames = ["Category", "Name", "Quantity", "Price/Cost", "Type"]
    #         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #
    #         writer.writeheader()
    #
    #         # Capital Costs
    #         for category, items in self.capital_costs.items():
    #             for item in items:
    #                 writer.writerow({
    #                     "Category": category,
    #                     "Name": item["name"],
    #                     "Quantity": item["quantity"],
    #                     "Price/Cost": item["price"],
    #                     "Type": "Capital"
    #                 })
    #
    #         # Operational Costs
    #         for category, items in self.operational_costs.items():
    #             for item in items:
    #                 if "one_time_cost" in item:
    #                     writer.writerow({
    #                         "Category": category,
    #                         "Name": item["name"],
    #                         "Quantity": 1,  # For one-time costs
    #                         "Price/Cost": item["one_time_cost"],
    #                         "Type": "Operational One-Time"
    #                     })
    #                 elif "monthly_cost" in item:
    #                     writer.writerow({
    #                         "Category": category,
    #                         "Name": item["name"],
    #                         "Quantity": 1,  # For monthly costs
    #                         "Price/Cost": item["monthly_cost"],
    #                         "Type": "Operational Monthly"
    #                     })
    #
    #         # Electricity Costs
    #         writer.writerow({
    #             "Category": "Electricity",
    #             "Name": "Total Electricity Costs",
    #             "Quantity": 1,
    #             "Price/Cost": self.electricity_costs,
    #             "Type": "Electricity"
    #         })
    def export_to_csv(self, filename):
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ["Категория", "Наименование", "Количество", "Цена/Затраты", "Тип"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()

            # Капитальные затраты
            writer.writerow(
                {"Категория": "", "Наименование": "Капитальные затраты", "Количество": "", "Цена/Затраты": "",
                 "Тип": ""})
            for category, items in self.capital_costs.items():
                for item in items:
                    writer.writerow({
                        "Категория": "  " + category,
                        "Наименование": item["name"],
                        "Количество": item["quantity"],
                        "Цена/Затраты": item["price"],
                        "Тип": "Капитальные"
                    })
                category_total = sum(item["quantity"] * item["price"] for item in items)
                writer.writerow({"Категория": "  " + category, "Наименование": "Итого", "Количество": "",
                                 "Цена/Затраты": category_total, "Тип": ""})
            total_capital = sum(
                sum(item["quantity"] * item["price"] for item in items) for items in self.capital_costs.values())
            writer.writerow({"Категория": "", "Наименование": "Итого капитальные затраты", "Количество": "",
                             "Цена/Затраты": total_capital, "Тип": ""})
            writer.writerow({})

            # Операционные затраты
            writer.writerow(
                {"Категория": "", "Наименование": "Операционные затраты", "Количество": "", "Цена/Затраты": "",
                 "Тип": ""})
            for category, items in self.operational_costs.items():
                for item in items:
                    if "one_time_cost" in item:
                        writer.writerow({
                            "Категория": "  " + category,
                            "Наименование": item["name"],
                            "Количество": 1,
                            "Цена/Затраты": item["one_time_cost"],
                            "Тип": "Операционные разовые"
                        })
                    elif "monthly_cost" in item:
                        writer.writerow({
                            "Категория": "  " + category,
                            "Наименование": item["name"],
                            "Количество": 1,
                            "Цена/Затраты": item["monthly_cost"],
                            "Тип": "Операционные периодические"
                        })
                category_total_one_time = sum(item["one_time_cost"] for item in items if "one_time_cost" in item)
                category_total_monthly = sum(item["monthly_cost"] for item in items if "monthly_cost" in item)
                writer.writerow({"Категория": "  " + category, "Наименование": "Итого разовые", "Количество": "",
                                 "Цена/Затраты": category_total_one_time, "Тип": ""})
                writer.writerow({"Категория": "  " + category, "Наименование": "Итого периодические", "Количество": "",
                                 "Цена/Затраты": category_total_monthly, "Тип": ""})
            total_operational_one_time = sum(
                sum(item["one_time_cost"] for item in items if "one_time_cost" in item) for items in
                self.operational_costs.values())
            total_operational_monthly = sum(
                sum(item["monthly_cost"] for item in items if "monthly_cost" in item) for items in
                self.operational_costs.values())
            writer.writerow({"Категория": "", "Наименование": "Итого операционные разовые", "Количество": "",
                             "Цена/Затраты": total_operational_one_time, "Тип": ""})
            writer.writerow({"Категория": "", "Наименование": "Итого операционные периодические", "Количество": "",
                             "Цена/Затраты": total_operational_monthly, "Тип": ""})
            writer.writerow({})

            # Затраты на электроэнергию
            writer.writerow({"Категория": "", "Наименование": "Затраты на электроэнергию", "Количество": "",
                             "Цена/Затраты": self.electricity_costs, "Тип": "Электроэнергия"})

            # Общие итоги
            writer.writerow({"Категория": "", "Наименование": "Итого все разовые затраты", "Количество": "",
                             "Цена/Затраты": total_capital + total_operational_one_time, "Тип": ""})
            writer.writerow({"Категория": "", "Наименование": "Итого все периодические затраты", "Количество": "",
                             "Цена/Затраты": total_operational_monthly, "Тип": ""})
            writer.writerow({"Категория": "", "Наименование": "Итого все затраты на электроэнергию", "Количество": "",
                             "Цена/Затраты": self.electricity_costs, "Тип": ""})
            writer.writerow({"Категория": "", "Наименование": "ОБЩИЙ ИТОГ", "Количество": "",
                             "Цена/Затраты": total_capital + total_operational_one_time + total_operational_monthly + self.electricity_costs,
                             "Тип": ""})
    def get_summary(self):
        # Сформируйте сводку в виде строки
        summary = f"Капитальные затраты: {self.capital_costs}\n"
        summary += f"Операционные затраты: {self.operational_costs}\n"
        summary += f"Стоимость электроэнергии: {self.electricity_costs}\n"
        summary += f"Общая стоимость: {self.capital_costs + self.operational_costs + self.electricity_costs}"
        return summary
