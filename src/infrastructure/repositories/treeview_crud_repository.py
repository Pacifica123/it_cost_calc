"""Backward-compatible CRUD facade with storage decoupled from Tkinter widgets."""

from __future__ import annotations

from tkinter import ttk

from infrastructure.repositories.in_memory_entity_repository import (
    InMemoryEntityRepository,
)


class TreeviewCrudRepository:
    def __init__(self, repository: InMemoryEntityRepository | None = None):
        self.repository = repository or InMemoryEntityRepository()

    @property
    def entities(self) -> dict[str, list[dict]]:
        return self.repository.entities

    def list(self, entity_name: str) -> list[dict]:
        return self.repository.list(entity_name)

    def get(self, entity_name: str, index: int) -> dict:
        return self.repository.get(entity_name, index)

    def add(self, entity_name: str, data: dict, table: ttk.Treeview | None = None) -> dict:
        row = self.repository.add(entity_name, data)
        if table is not None:
            self.sync_table(entity_name, table)
        return row

    def delete(self, entity_name: str, index: int, table: ttk.Treeview | None = None) -> dict:
        row = self.repository.delete(entity_name, index)
        if table is not None:
            self.sync_table(entity_name, table)
        return row

    def update(
        self, entity_name: str, index: int, data: dict, table: ttk.Treeview | None = None
    ) -> dict:
        row = self.repository.update(entity_name, index, data)
        if table is not None:
            self.sync_table(entity_name, table)
        return row

    def clear(self) -> None:
        self.repository.clear()

    def sync_table(self, entity_name: str, table: ttk.Treeview) -> None:
        table.delete(*table.get_children())
        rows = self.repository.list(entity_name)
        columns = table["columns"]
        for row in rows:
            table.insert("", "end", values=[row.get(column, "") for column in columns])
