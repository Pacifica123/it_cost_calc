from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class JsonFileStorage:
    def read(self, path: str | Path) -> Any:
        file_path = Path(path)
        logger.debug("Чтение JSON из %s", file_path)
        with file_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        logger.debug("JSON успешно прочитан из %s", file_path)
        return payload

    def write(self, path: str | Path, payload: Any) -> Path:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug("Запись JSON в %s", file_path)
        with file_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        logger.info("JSON сохранён: %s", file_path)
        return file_path
