from pathlib import Path

from domain import CapitalItem, OperationalExpense
from infrastructure.repositories.json_entity_repository import JsonEntityRepository
from infrastructure.storage import JsonFileStorage


def test_json_entity_repository_persists_entities(tmp_path: Path):
    path = tmp_path / "runtime_entities.json"
    repository = JsonEntityRepository(path, JsonFileStorage())

    repository.add("server", CapitalItem(name="srv", quantity=1, price=100.0))
    repository.add("backup", OperationalExpense(name="cloud", monthly_cost=12.0))

    reloaded = JsonEntityRepository(path, JsonFileStorage())
    assert reloaded.list("server")[0]["price"] == 100.0
    assert reloaded.list("backup")[0]["monthly_cost"] == 12.0
