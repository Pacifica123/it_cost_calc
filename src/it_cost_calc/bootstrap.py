from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from it_cost_calc.application.services.equipment_service import EquipmentService
from it_cost_calc.application.use_cases.load_demo_dataset import LoadDemoDatasetUseCase
from it_cost_calc.infrastructure.logging import configure_logging
from it_cost_calc.infrastructure.repositories.json_entity_repository import JsonEntityRepository
from it_cost_calc.infrastructure.storage import JsonFileStorage

if TYPE_CHECKING:
    from it_cost_calc.app import CalculatorApp

logger = logging.getLogger(__name__)


def create_app() -> "CalculatorApp":
    from it_cost_calc.app import CalculatorApp

    logger.info("Создание экземпляра CalculatorApp")
    return CalculatorApp()


def load_demo_data(
    repo_root: Path | None = None,
    *,
    fixture_path: Path | None = None,
    runtime_entities_path: Path | None = None,
) -> dict[str, list[dict]]:
    root = repo_root or Path(__file__).resolve().parents[2]
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
