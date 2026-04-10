"""Application service for mutable catalog-like entities."""

from __future__ import annotations

from copy import deepcopy

from it_cost_calc.application.ports import EntityRepository


class EntityCatalogService:
    def __init__(self, repository: EntityRepository):
        self.repository = repository

    @property
    def entities(self) -> dict[str, list[dict]]:
        return self.repository.entities

    def list_entities(self, entity_name: str) -> list[dict]:
        return self.repository.list(entity_name)

    def get_entity(self, entity_name: str, index: int) -> dict:
        return self.repository.get(entity_name, index)

    def add_entity(self, entity_name: str, data: dict) -> dict:
        return self.repository.add(entity_name, data)

    def update_entity(self, entity_name: str, index: int, data: dict) -> dict:
        return self.repository.update(entity_name, index, data)

    def delete_entity(self, entity_name: str, index: int) -> dict:
        return self.repository.delete(entity_name, index)

    def replace_all(self, payload: dict[str, list[dict]]) -> None:
        self.repository.clear()
        for entity_name, rows in payload.items():
            for row in rows:
                self.repository.add(entity_name, deepcopy(row))
