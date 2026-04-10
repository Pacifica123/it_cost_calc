from pathlib import Path

from application.services.entity_catalog_service import EntityCatalogService
from application.use_cases.load_demo_dataset import LoadDemoDatasetUseCase
from infrastructure.repositories.in_memory_entity_repository import (
    InMemoryEntityRepository,
)
from infrastructure.storage import JsonFileStorage


def test_load_demo_dataset_use_case_reads_fixture_and_populates_catalog():
    fixture_path = Path(__file__).resolve().parents[2] / "data" / "fixtures" / "demo_dataset.json"
    use_case = LoadDemoDatasetUseCase(
        storage=JsonFileStorage(),
        target=EntityCatalogService(InMemoryEntityRepository()),
    )

    dataset = use_case.execute(fixture_path)

    assert "server" in dataset
    assert dataset["server"][0]["name"] == "Rock w9102p"
