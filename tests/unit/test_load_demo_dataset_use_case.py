from pathlib import Path

from it_cost_calc.application.services.entity_catalog_service import EntityCatalogService
from it_cost_calc.application.use_cases.load_demo_dataset import LoadDemoDatasetUseCase
from it_cost_calc.infrastructure.repositories.in_memory_entity_repository import (
    InMemoryEntityRepository,
)
from it_cost_calc.infrastructure.storage import JsonFileStorage


def test_load_demo_dataset_use_case_reads_fixture_and_populates_catalog():
    fixture_path = Path(__file__).resolve().parents[2] / "data" / "fixtures" / "demo_dataset.json"
    use_case = LoadDemoDatasetUseCase(
        storage=JsonFileStorage(),
        target=EntityCatalogService(InMemoryEntityRepository()),
    )

    dataset = use_case.execute(fixture_path)

    assert "server" in dataset
    assert dataset["server"][0]["name"] == "Rock w9102p"
