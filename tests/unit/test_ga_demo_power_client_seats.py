from pathlib import Path
import json

from application.services.equipment_service import EquipmentService
from infrastructure.repositories.in_memory_entity_repository import InMemoryEntityRepository
from ui.tabs.genetic_optimization_tab import GeneticOptimizationTab
from application.services.genetic_optimization_service import RuntimeOptimizationItem
from domain.decision.ahp.aggregation import aggregate_configuration


def test_demo_dataset_contains_power_and_client_seats_for_ga():
    fixture_path = Path(__file__).resolve().parents[2] / "data" / "fixtures" / "demo_dataset.json"
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    entities = payload["entities"]

    for category in ("server", "client", "network"):
        assert all("max_power" in row for row in entities[category])

    printer = next(row for row in entities["client"] if row["name"] == "i-SENSYS MF752Cdw")
    assert printer["quantity"] == 2
    assert printer["client_seats"] == 0
    assert printer["max_power"] > 0


def test_genetic_tab_client_capacity_uses_client_seats_when_present():
    tab = object.__new__(GeneticOptimizationTab)
    subset = [
        RuntimeOptimizationItem(name="Atlas H434", category="client", quantity=3, client_seats=3),
        RuntimeOptimizationItem(name="AuBox", category="client", quantity=1, client_seats=1),
        RuntimeOptimizationItem(name="i-SENSYS MF752Cdw", category="client", quantity=2, client_seats=0),
        RuntimeOptimizationItem(name="Router", category="network", quantity=1),
    ]

    assert tab._client_capacity(subset) == 4

    rows = [item.properties for item in subset]
    assert tab._client_capacity_from_dicts(rows) == 4


def test_energy_relevant_rows_preserve_demo_power_fields():
    repository = InMemoryEntityRepository(
        {
            "server": [{"name": "Server", "quantity": 1, "price": 100.0, "max_power": 500.0}],
            "client": [
                {
                    "name": "Printer",
                    "quantity": 2,
                    "price": 50.0,
                    "max_power": 850.0,
                    "client_seats": 0,
                }
            ],
            "network": [{"name": "Switch", "quantity": 1, "price": 10.0, "max_power": 18.0}],
        }
    )
    service = EquipmentService(repository)

    rows = service.list_energy_relevant_rows()

    assert {row["name"] for row in rows} == {"Server", "Printer", "Switch"}
    printer = next(row for row in rows if row["name"] == "Printer")
    assert printer["max_power"] == 850.0
    assert printer["client_seats"] == 0


def test_ahp_aggregate_counts_client_seats_not_client_category_quantity():
    config = {
        "id": "cfg",
        "devices": [
            {"role": "client", "client_seats": 3, "cost": 100.0},
            {"role": "client", "client_seats": 0, "cost": 50.0},
            {"role": "network", "cost": 10.0},
        ],
        "meta": {"people": 3},
    }

    aggregate = aggregate_configuration(config)

    assert aggregate["counts"]["client"] == 3


def test_genetic_tab_category_policy_supports_required_excluded_optional_states():
    from application.services.analysis_scope_profile_service import (
        CATEGORY_POLICY_EXCLUDED,
        CATEGORY_POLICY_OPTIONAL,
        CATEGORY_POLICY_REQUIRED,
        AnalysisScopeProfileService,
    )

    service = AnalysisScopeProfileService()
    policy = service.normalize_category_policy(
        "technical",
        {
            "server": CATEGORY_POLICY_REQUIRED,
            "client": CATEGORY_POLICY_EXCLUDED,
            "network": CATEGORY_POLICY_OPTIONAL,
        },
    )

    assert service.required_categories("technical", category_policy=policy) == ["server"]
    assert service.excluded_categories("technical", category_policy=policy) == ["client"]
