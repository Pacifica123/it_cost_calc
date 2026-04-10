"""Aggregation service for capital, operational, and electricity costs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from it_cost_calc.domain import (
    CapitalItem,
    OperationalExpense,
    ensure_capital_item,
    ensure_operational_expense,
)
from it_cost_calc.infrastructure.exporters.csv_exporter import TotalCostsCsvExporter
from it_cost_calc.shared.constants import CAPITAL_COST_CATEGORIES, OPERATIONAL_COST_CATEGORIES


class CostAggregationService:
    def __init__(self, exporter: TotalCostsCsvExporter | None = None):
        self.exporter = exporter or TotalCostsCsvExporter()
        self.capital_costs: dict[str, list[CapitalItem]] = {}
        self.operational_costs: dict[str, list[OperationalExpense]] = {}
        self.electricity_costs = 0.0

    def _extract_entities(self, source) -> dict[str, list[Any]]:
        if hasattr(source, "entities"):
            return source.entities
        if isinstance(source, dict):
            return source
        raise TypeError(
            "Cost aggregation source must expose an 'entities' mapping or be a mapping itself"
        )

    def update_capital_costs(self, source) -> None:
        if hasattr(source, "list_capital_items"):
            self.capital_costs = {
                category: list(source.list_capital_items(category))
                for category in CAPITAL_COST_CATEGORIES
            }
            return
        entities = self._extract_entities(source)
        self.capital_costs = {
            category: [
                ensure_capital_item(item, category=category) for item in entities.get(category, [])
            ]
            for category in CAPITAL_COST_CATEGORIES
        }

    def update_operational_costs(self, source) -> None:
        if hasattr(source, "list_operational_items"):
            self.operational_costs = {
                category: list(source.list_operational_items(category))
                for category in OPERATIONAL_COST_CATEGORIES
            }
            return
        entities = self._extract_entities(source)
        self.operational_costs = {
            category: [ensure_operational_expense(item) for item in entities.get(category, [])]
            for category in OPERATIONAL_COST_CATEGORIES
        }

    def update_electricity_costs(self, electricity_cost: float) -> None:
        self.electricity_costs = float(electricity_cost)

    def calculate_totals(self) -> dict[str, float]:
        total_capital = sum(
            item.total_cost for items in self.capital_costs.values() for item in items
        )
        total_operational_one_time = sum(
            item.one_time_cost for items in self.operational_costs.values() for item in items
        )
        total_operational_monthly = sum(
            item.monthly_cost for items in self.operational_costs.values() for item in items
        )
        return {
            "total_capital": total_capital,
            "total_operational_one_time": total_operational_one_time,
            "total_operational_monthly": total_operational_monthly,
            "electricity_costs": self.electricity_costs,
        }

    def export_to_csv(self, filename: str | Path) -> None:
        self.exporter.export(
            filename=filename,
            capital_costs=self.capital_costs,
            operational_costs=self.operational_costs,
            electricity_costs=self.electricity_costs,
        )
