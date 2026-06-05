from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from application.services.equipment_service import EquipmentService
from application.use_cases.load_demo_dataset import LoadDemoDatasetUseCase
from infrastructure.logging import configure_logging
from infrastructure.repositories.json_entity_repository import JsonEntityRepository
from infrastructure.storage import JsonFileStorage

if TYPE_CHECKING:
    from app import CalculatorApp

logger = logging.getLogger(__name__)


def create_app() -> "CalculatorApp":
    from app import CalculatorApp

    logger.info("Создание экземпляра CalculatorApp")
    return CalculatorApp()


def create_qt_app(argv: Sequence[str] | None = None):
    """Create the parallel Qt application instance.

    The Tkinter application remains the default runtime during the migration;
    this helper is used by scripts/run_qt_app.py and future Qt smoke checks.
    """

    from ui_qt.app import create_qt_application

    logger.info("Создание экземпляра QApplication для Qt UI")
    return create_qt_application(argv)


def main_qt(argv: Sequence[str] | None = None) -> int:
    """Run the parallel PySide6/Qt UI shell."""

    configure_logging()
    logger.info("Запуск параллельного Qt-интерфейса")
    from ui_qt.app import run_qt_app

    return run_qt_app(argv)


def smoke_check_qt(argv: Sequence[str] | None = None) -> int:
    """Create the Qt shell without showing a window."""

    configure_logging()
    logger.info("Smoke-проверка Qt-интерфейса")
    from ui_qt.app import smoke_check

    return smoke_check(argv)


def load_demo_data(
    repo_root: Path | None = None,
    *,
    fixture_path: Path | None = None,
    runtime_entities_path: Path | None = None,
) -> dict[str, list[dict]]:
    root = repo_root or Path(__file__).resolve().parents[1]
    log_path = configure_logging(repo_root=root)
    logger.debug("CLI-загрузка демо-данных. Лог-файл: %s", log_path)
    data_root = root / "data"
    storage = JsonFileStorage()
    repository = JsonEntityRepository(
        runtime_entities_path or data_root / "generated" / "runtime_entities.json", storage
    )
    service = EquipmentService(repository)
    use_case = LoadDemoDatasetUseCase(storage, service)
    dataset_path = fixture_path or data_root / "fixtures" / "demo_dataset.json"
    logger.info("Запуск загрузки демо-данных из %s", dataset_path)
    return use_case.execute(dataset_path)


def main() -> None:
    configure_logging()
    logger.info("Запуск настольного приложения")
    app = create_app()
    app.mainloop()
