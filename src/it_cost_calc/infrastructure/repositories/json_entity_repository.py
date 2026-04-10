"""JSON-backed repository for mutable runtime entities."""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

from it_cost_calc.domain import to_plain_data
from it_cost_calc.infrastructure.storage import JsonFileStorage

logger = logging.getLogger(__name__)


class JsonEntityRepository:
    def __init__(
        self,
        path: str | Path,
        storage: JsonFileStorage | None = None,
        *,
        autosave: bool = True,
    ):
        self.path = Path(path)
        self.storage = storage or JsonFileStorage()
        self.autosave = autosave
        self._entities: dict[str, list[dict[str, Any]]] = {}
        self._load_if_exists()

    @property
    def entities(self) -> dict[str, list[dict[str, Any]]]:
        return self._entities

    def _load_if_exists(self) -> None:
        if not self.path.exists():
            self._entities = {}
            logger.debug("Runtime-хранилище ещё не создано: %s", self.path)
            return
        payload = self.storage.read(self.path)
        self._entities = deepcopy(payload.get("entities", payload))
        logger.info(
            "Runtime-хранилище загружено из %s (%s сущностей)",
            self.path,
            sum(len(rows) for rows in self._entities.values()),
        )

    def _persist(self) -> None:
        if self.autosave:
            self.save()

    def save(self) -> Path:
        saved_path = self.storage.write(self.path, {"entities": self._entities})
        logger.debug("Runtime-хранилище сохранено: %s", saved_path)
        return saved_path

    def list(self, entity_name: str) -> list[dict[str, Any]]:
        return self._entities.setdefault(entity_name, [])

    def get(self, entity_name: str, index: int) -> dict[str, Any]:
        return self.list(entity_name)[index]

    def add(self, entity_name: str, data: Any) -> dict[str, Any]:
        row = to_plain_data(data)
        self.list(entity_name).append(row)
        logger.debug("Добавлена запись в %s: %s", entity_name, row)
        self._persist()
        return row

    def update(self, entity_name: str, index: int, data: Any) -> dict[str, Any]:
        row = to_plain_data(data)
        self.list(entity_name)[index] = row
        logger.debug("Обновлена запись %s[%s]: %s", entity_name, index, row)
        self._persist()
        return row

    def delete(self, entity_name: str, index: int) -> dict[str, Any]:
        row = self.list(entity_name).pop(index)
        logger.debug("Удалена запись %s[%s]: %s", entity_name, index, row)
        self._persist()
        return row

    def clear(self) -> None:
        logger.warning("Runtime-хранилище очищено: %s", self.path)
        self._entities.clear()
        self._persist()
