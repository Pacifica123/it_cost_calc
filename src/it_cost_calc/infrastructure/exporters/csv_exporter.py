"""CSV export adapters for the desktop application."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Iterable, Mapping

from it_cost_calc.domain import to_plain_data

logger = logging.getLogger(__name__)


class TotalCostsCsvExporter:
    def export(
        self,
        filename: str | Path,
        capital_costs: Mapping[str, Iterable[Any]],
        operational_costs: Mapping[str, Iterable[Any]],
        electricity_costs: float,
    ) -> None:
        path = Path(filename)
        normalized_capital = {
            category: [to_plain_data(item) for item in items]
            for category, items in capital_costs.items()
        }
        normalized_operational = {
            category: [to_plain_data(item) for item in items]
            for category, items in operational_costs.items()
        }

        logger.info("Начат экспорт CSV-отчёта: %s", path)
        with path.open("w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["Категория", "Наименование", "Количество", "Цена/Затраты", "Тип"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            writer.writerow(
                {
                    "Категория": "",
                    "Наименование": "Капитальные затраты",
                    "Количество": "",
                    "Цена/Затраты": "",
                    "Тип": "",
                }
            )
            for category, items in normalized_capital.items():
                items = list(items)
                for item in items:
                    writer.writerow(
                        {
                            "Категория": "  " + category,
                            "Наименование": item["name"],
                            "Количество": item["quantity"],
                            "Цена/Затраты": item["price"],
                            "Тип": "Капитальные",
                        }
                    )
                category_total = sum(item["quantity"] * item["price"] for item in items)
                writer.writerow(
                    {
                        "Категория": "  " + category,
                        "Наименование": "Итого",
                        "Количество": "",
                        "Цена/Затраты": category_total,
                        "Тип": "",
                    }
                )

            total_capital = sum(
                sum(item["quantity"] * item["price"] for item in items)
                for items in normalized_capital.values()
            )
            writer.writerow(
                {
                    "Категория": "",
                    "Наименование": "Итого капитальные затраты",
                    "Количество": "",
                    "Цена/Затраты": total_capital,
                    "Тип": "",
                }
            )
            writer.writerow({})

            writer.writerow(
                {
                    "Категория": "",
                    "Наименование": "Операционные затраты",
                    "Количество": "",
                    "Цена/Затраты": "",
                    "Тип": "",
                }
            )
            for category, items in normalized_operational.items():
                items = list(items)
                for item in items:
                    if "one_time_cost" in item and float(item.get("one_time_cost", 0.0)):
                        writer.writerow(
                            {
                                "Категория": "  " + category,
                                "Наименование": item["name"],
                                "Количество": 1,
                                "Цена/Затраты": item["one_time_cost"],
                                "Тип": "Операционные разовые",
                            }
                        )
                    if "monthly_cost" in item and float(item.get("monthly_cost", 0.0)):
                        writer.writerow(
                            {
                                "Категория": "  " + category,
                                "Наименование": item["name"],
                                "Количество": 1,
                                "Цена/Затраты": item["monthly_cost"],
                                "Тип": "Операционные периодические",
                            }
                        )
                category_total_one_time = sum(item.get("one_time_cost", 0.0) for item in items)
                category_total_monthly = sum(item.get("monthly_cost", 0.0) for item in items)
                writer.writerow(
                    {
                        "Категория": "  " + category,
                        "Наименование": "Итого разовые",
                        "Количество": "",
                        "Цена/Затраты": category_total_one_time,
                        "Тип": "",
                    }
                )
                writer.writerow(
                    {
                        "Категория": "  " + category,
                        "Наименование": "Итого периодические",
                        "Количество": "",
                        "Цена/Затраты": category_total_monthly,
                        "Тип": "",
                    }
                )

            total_operational_one_time = sum(
                sum(item.get("one_time_cost", 0.0) for item in items)
                for items in normalized_operational.values()
            )
            total_operational_monthly = sum(
                sum(item.get("monthly_cost", 0.0) for item in items)
                for items in normalized_operational.values()
            )
            writer.writerow(
                {
                    "Категория": "",
                    "Наименование": "Итого операционные разовые",
                    "Количество": "",
                    "Цена/Затраты": total_operational_one_time,
                    "Тип": "",
                }
            )
            writer.writerow(
                {
                    "Категория": "",
                    "Наименование": "Итого операционные периодические",
                    "Количество": "",
                    "Цена/Затраты": total_operational_monthly,
                    "Тип": "",
                }
            )
            writer.writerow({})

            writer.writerow(
                {
                    "Категория": "",
                    "Наименование": "Затраты на электроэнергию",
                    "Количество": "",
                    "Цена/Затраты": electricity_costs,
                    "Тип": "Электроэнергия",
                }
            )
            writer.writerow(
                {
                    "Категория": "",
                    "Наименование": "Итого все разовые затраты",
                    "Количество": "",
                    "Цена/Затраты": total_capital + total_operational_one_time,
                    "Тип": "",
                }
            )
            writer.writerow(
                {
                    "Категория": "",
                    "Наименование": "Итого все периодические затраты",
                    "Количество": "",
                    "Цена/Затраты": total_operational_monthly,
                    "Тип": "",
                }
            )
            writer.writerow(
                {
                    "Категория": "",
                    "Наименование": "Итого все затраты на электроэнергию",
                    "Количество": "",
                    "Цена/Затраты": electricity_costs,
                    "Тип": "",
                }
            )
            writer.writerow(
                {
                    "Категория": "",
                    "Наименование": "ОБЩИЙ ИТОГ",
                    "Количество": "",
                    "Цена/Затраты": total_capital
                    + total_operational_one_time
                    + total_operational_monthly
                    + electricity_costs,
                    "Тип": "",
                }
            )
        logger.info(
            "CSV-отчёт сохранён: %s (капитальных категорий=%s, операционных категорий=%s)",
            path,
            len(normalized_capital),
            len(normalized_operational),
        )
