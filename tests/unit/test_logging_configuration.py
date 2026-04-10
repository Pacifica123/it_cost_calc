from __future__ import annotations

import logging
from pathlib import Path

from infrastructure.logging import configure_logging


def test_configure_logging_writes_file(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_path = configure_logging(log_dir=log_dir, level="DEBUG", force=True)

    logging.getLogger("tests.logging").info("Проверка файлового логирования")

    content = log_path.read_text(encoding="utf-8")
    assert log_path.parent == log_dir
    assert "Проверка файлового логирования" in content
