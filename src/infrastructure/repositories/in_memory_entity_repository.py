"""In-memory entity repository decoupled from Tkinter widgets."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from domain import to_plain_data


class InMemoryEntityRepository:
    def __init__(self, initial_entities: dict[str, list[dict[str, Any]]] | None = None):
        self._entities: dict[str, list[dict[str, Any]]] = deepcopy(initial_entities or {})

    @property
    def entities(self) -> dict[str, list[dict[str, Any]]]:
        return self._entities

    def list(self, entity_name: str) -> list[dict[str, Any]]:
        return self._entities.setdefault(entity_name, [])

    def get(self, entity_name: str, index: int) -> dict[str, Any]:
        return self.list(entity_name)[index]

    def add(self, entity_name: str, data: Any) -> dict[str, Any]:
        row = to_plain_data(data)
        self.list(entity_name).append(row)
        return row

    def update(self, entity_name: str, index: int, data: Any) -> dict[str, Any]:
        row = to_plain_data(data)
        self.list(entity_name)[index] = row
        return row

    def delete(self, entity_name: str, index: int) -> dict[str, Any]:
        return self.list(entity_name).pop(index)

    def clear(self) -> None:
        self._entities.clear()
