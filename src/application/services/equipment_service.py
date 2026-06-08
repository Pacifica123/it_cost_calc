"""Application service for structured access to equipment and operational entities."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from application.ports import EntityRepository
from application.services.runtime_entity_normalization_service import normalize_runtime_row
from domain import (
    CapitalItem,
    OperationalExpense,
    ensure_capital_item,
    ensure_operational_expense,
    to_plain_data,
)
from shared.constants import CAPITAL_COST_CATEGORIES, OPERATIONAL_COST_CATEGORIES


class EquipmentService:
    def __init__(self, repository: EntityRepository):
        self.repository = repository

    @property
    def entities(self) -> dict[str, list[dict[str, Any]]]:
        return self.repository.entities

    def _validate_category(self, category: str, allowed_categories: tuple[str, ...]) -> None:
        if category not in allowed_categories:
            raise ValueError(f"Unsupported category: {category}")

    def list_capital_items(self, category: str) -> list[CapitalItem]:
        self._validate_category(category, CAPITAL_COST_CATEGORIES)
        return [
            ensure_capital_item(
                normalize_runtime_row(item, category=category),
                category=category,
            )
            for item in self.repository.list(category)
        ]

    def list_capital_item_rows(self, category: str) -> list[dict[str, Any]]:
        return [to_plain_data(item) for item in self.list_capital_items(category)]

    def get_capital_item(self, category: str, index: int) -> CapitalItem:
        self._validate_category(category, CAPITAL_COST_CATEGORIES)
        return ensure_capital_item(
            normalize_runtime_row(self.repository.get(category, index), category=category),
            category=category,
        )

    def get_capital_item_row(self, category: str, index: int) -> dict[str, Any]:
        return to_plain_data(self.get_capital_item(category, index))

    def add_capital_item(self, category: str, item: CapitalItem | dict[str, Any]) -> CapitalItem:
        self._validate_category(category, CAPITAL_COST_CATEGORIES)
        model = ensure_capital_item(
            normalize_runtime_row(to_plain_data(item), category=category),
            category=category,
        )
        self.repository.add(category, model)
        return model

    def update_capital_item(
        self, category: str, index: int, item: CapitalItem | dict[str, Any]
    ) -> CapitalItem:
        self._validate_category(category, CAPITAL_COST_CATEGORIES)
        model = ensure_capital_item(
            normalize_runtime_row(to_plain_data(item), category=category),
            category=category,
        )
        self.repository.update(category, index, model)
        return model

    def delete_capital_item(self, category: str, index: int) -> CapitalItem:
        self._validate_category(category, CAPITAL_COST_CATEGORIES)
        return ensure_capital_item(
            normalize_runtime_row(self.repository.delete(category, index), category=category),
            category=category,
        )

    def list_operational_items(self, category: str) -> list[OperationalExpense]:
        self._validate_category(category, OPERATIONAL_COST_CATEGORIES)
        return [
            ensure_operational_expense(normalize_runtime_row(item, category=category))
            for item in self.repository.list(category)
        ]

    def list_operational_item_rows(self, category: str) -> list[dict[str, Any]]:
        return [to_plain_data(item) for item in self.list_operational_items(category)]

    def get_operational_item(self, category: str, index: int) -> OperationalExpense:
        self._validate_category(category, OPERATIONAL_COST_CATEGORIES)
        return ensure_operational_expense(
            normalize_runtime_row(self.repository.get(category, index), category=category)
        )

    def get_operational_item_row(self, category: str, index: int) -> dict[str, Any]:
        return to_plain_data(self.get_operational_item(category, index))

    def add_operational_item(
        self, category: str, item: OperationalExpense | dict[str, Any]
    ) -> OperationalExpense:
        self._validate_category(category, OPERATIONAL_COST_CATEGORIES)
        model = ensure_operational_expense(
            normalize_runtime_row(to_plain_data(item), category=category)
        )
        self.repository.add(category, model)
        return model

    def update_operational_item(
        self, category: str, index: int, item: OperationalExpense | dict[str, Any]
    ) -> OperationalExpense:
        self._validate_category(category, OPERATIONAL_COST_CATEGORIES)
        model = ensure_operational_expense(
            normalize_runtime_row(to_plain_data(item), category=category)
        )
        self.repository.update(category, index, model)
        return model

    def delete_operational_item(self, category: str, index: int) -> OperationalExpense:
        self._validate_category(category, OPERATIONAL_COST_CATEGORIES)
        return ensure_operational_expense(
            normalize_runtime_row(self.repository.delete(category, index), category=category)
        )

    def list_energy_relevant_items(self) -> list[CapitalItem]:
        rows: list[CapitalItem] = []
        for category in ("server", "client", "network"):
            rows.extend(self.list_capital_items(category))
        return rows

    def list_energy_relevant_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for category in ("server", "client", "network"):
            for row in self.repository.list(category):
                payload = normalize_runtime_row(to_plain_data(row), category=category)
                rows.append(payload)
        return rows

    def list_round_the_clock_equipment_names(self) -> set[str]:
        return {item.name for item in self.list_capital_items("server") if item.name}

    def replace_all(self, payload: dict[str, list[dict[str, Any]]]) -> None:
        normalized_payload: dict[str, list[dict[str, Any]]] = {}
        for entity_name, rows in payload.items():
            normalized_rows: list[dict[str, Any]] = []
            for row in rows:
                if entity_name in CAPITAL_COST_CATEGORIES or entity_name in OPERATIONAL_COST_CATEGORIES:
                    normalized_rows.append(normalize_runtime_row(row, category=entity_name))
                else:
                    normalized_rows.append(deepcopy(row))
            normalized_payload[entity_name] = normalized_rows

        bulk_replace = getattr(self.repository, "replace_all", None)
        if callable(bulk_replace):
            bulk_replace(normalized_payload)
            return

        self.repository.clear()
        for entity_name, rows in normalized_payload.items():
            for row in rows:
                self.repository.add(entity_name, row)
