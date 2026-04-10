from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

from infrastructure.storage import JsonFileStorage

logger = logging.getLogger(__name__)


class DemoDatasetTarget(Protocol):
    def replace_all(self, payload: dict[str, list[dict]]) -> None: ...


class LoadDemoDatasetUseCase:
    def __init__(self, storage: JsonFileStorage, target: DemoDatasetTarget):
        self.storage = storage
        self.target = target

    def execute(self, path: str | Path) -> dict[str, list[dict]]:
        logger.info("Загрузка демо-данных из %s", path)
        payload = self.storage.read(path)
        entities = payload.get("entities", payload)
        self.target.replace_all(entities)
        logger.info(
            "Демо-данные загружены: категорий=%s, строк=%s",
            len(entities),
            sum(len(rows) for rows in entities.values()),
        )
        return entities
