from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Final

DEFAULT_LOG_FILE_NAME: Final[str] = "it_cost_calc.log"
DEFAULT_LOG_FORMAT: Final[str] = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DEFAULT_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"
_ENV_LOG_LEVEL: Final[str] = "IT_COST_LOG_LEVEL"
_MANAGED_HANDLER_MARKER: Final[str] = "_it_cost_calc_managed"


def normalize_log_level(level: int | str | None) -> int:
    if isinstance(level, int):
        return level
    if isinstance(level, str) and level.strip():
        resolved = logging.getLevelName(level.strip().upper())
        if isinstance(resolved, int):
            return resolved
    env_level = os.getenv(_ENV_LOG_LEVEL, "INFO")
    resolved = logging.getLevelName(env_level.strip().upper())
    if isinstance(resolved, int):
        return resolved
    return logging.INFO


def resolve_repo_root(repo_root: str | Path | None = None) -> Path:
    if repo_root is not None:
        return Path(repo_root).resolve()
    return Path(__file__).resolve().parents[3]


def resolve_log_directory(
    repo_root: str | Path | None = None,
    *,
    log_dir: str | Path | None = None,
) -> Path:
    if log_dir is not None:
        return Path(log_dir).resolve()
    return resolve_repo_root(repo_root) / "data" / "generated" / "logs"


def _mark_handler(handler: logging.Handler) -> logging.Handler:
    setattr(handler, _MANAGED_HANDLER_MARKER, True)
    return handler


def _clear_managed_handlers(root_logger: logging.Logger) -> None:
    for handler in list(root_logger.handlers):
        if getattr(handler, _MANAGED_HANDLER_MARKER, False):
            root_logger.removeHandler(handler)
            handler.close()


def configure_logging(
    level: int | str | None = None,
    *,
    repo_root: str | Path | None = None,
    log_dir: str | Path | None = None,
    console: bool = True,
    force: bool = False,
) -> Path:
    """Настраивает общее логирование приложения.

    Логи пишутся в ``data/generated/logs/it_cost_calc.log``. Консольный обработчик
    показывает сообщения начиная с указанного уровня, файловый — сохраняет всё от DEBUG.
    Повторный вызов не дублирует обработчики; при ``force=True`` конфигурация
    пересоздаётся.
    """

    effective_level = normalize_log_level(level)
    target_dir = resolve_log_directory(repo_root, log_dir=log_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    log_path = target_dir / DEFAULT_LOG_FILE_NAME

    root_logger = logging.getLogger()
    has_managed_handlers = any(
        getattr(handler, _MANAGED_HANDLER_MARKER, False) for handler in root_logger.handlers
    )
    if has_managed_handlers and not force:
        return log_path

    if force:
        _clear_managed_handlers(root_logger)

    root_logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT)

    file_handler = _mark_handler(logging.FileHandler(log_path, encoding="utf-8"))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    if console:
        console_handler = _mark_handler(logging.StreamHandler())
        console_handler.setLevel(effective_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    logging.captureWarnings(True)
    logging.getLogger(__name__).debug(
        "Логирование настроено: level=%s, log_path=%s",
        logging.getLevelName(effective_level),
        log_path,
    )
    return log_path


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
