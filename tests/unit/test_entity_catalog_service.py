from application.services.entity_catalog_service import EntityCatalogService
from infrastructure.repositories.in_memory_entity_repository import (
    InMemoryEntityRepository,
)


def test_entity_catalog_service_replaces_dataset_and_exposes_rows():
    repository = InMemoryEntityRepository()
    service = EntityCatalogService(repository)

    service.replace_all(
        {
            "server": [{"name": "srv", "quantity": 1, "price": 100.0}],
            "backup": [{"name": "cloud", "monthly_cost": 10.0}],
        }
    )

    assert service.list_entities("server")[0]["name"] == "srv"
    assert service.list_entities("backup")[0]["monthly_cost"] == 10.0
