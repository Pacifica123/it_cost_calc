from pathlib import Path
import json

from application.services.equipment_service import EquipmentService
from infrastructure.repositories.in_memory_entity_repository import InMemoryEntityRepository
from application.services.analysis_scope_profile_service import AnalysisScopeProfileService
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


def test_profile_client_capacity_uses_client_seats_when_present():
    service = AnalysisScopeProfileService()
    subset = [
        RuntimeOptimizationItem(name="Atlas H434", category="client", quantity=3, client_seats=3),
        RuntimeOptimizationItem(name="AuBox", category="client", quantity=1, client_seats=1),
        RuntimeOptimizationItem(name="i-SENSYS MF752Cdw", category="client", quantity=2, client_seats=0),
        RuntimeOptimizationItem(name="Router", category="network", quantity=1),
    ]

    assert service._client_capacity(subset) == 4

    rows = [item.properties for item in subset]
    assert service._client_capacity(rows) == 4


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


def test_profile_builds_maximum_capacity_constraint_for_ga():
    service = AnalysisScopeProfileService()

    constraints = service.build_ga_constraints(
        "technical",
        target_units=3,
        max_target_units=3,
    )

    by_name = {constraint["name"]: constraint for constraint in constraints}
    assert by_name["client_capacity_required"]["operator"] == ">="
    assert by_name["client_capacity_required"]["bound"] == 3.0
    assert by_name["client_capacity_required_max"]["operator"] == "<="
    assert by_name["client_capacity_required_max"]["bound"] == 3.0
    assert by_name["client_capacity_required_max"]["label"] == "Максимум клиентских мест"


def test_ga_respects_maximum_places_and_selected_items_on_large_technical_pool():
    from application.services.genetic_optimization_service import GeneticOptimizationService
    from infrastructure.repositories.in_memory_entity_repository import InMemoryEntityRepository

    repository = InMemoryEntityRepository(
        {
            "server": [
                {
                    "name": f"Server {index}",
                    "quantity": 1,
                    "price": 1000.0 + index,
                    "ram_gb": 64 + index,
                    "cpu_cores": 16 + index,
                    "storage_gb": 512,
                    "max_power": 200,
                }
                for index in range(12)
            ],
            "client": [
                {
                    "name": f"Client {index}",
                    "quantity": 1,
                    "price": 100.0 + index,
                    "ram_gb": 8,
                    "cpu_cores": 4,
                    "storage_gb": 256,
                    "max_power": 50,
                    "client_seats": 3,
                }
                for index in range(12)
            ],
            "network": [
                {
                    "name": f"Network {index}",
                    "quantity": 1,
                    "price": 50.0 + index,
                    "lan_ports": 8 + index,
                    "lan_speed_mbps": 1000,
                    "max_power": 15,
                }
                for index in range(12)
            ],
        }
    )
    service = GeneticOptimizationService(repository)

    summary = service.run(
        analysis_scope="technical",
        target_units=3,
        max_target_units=3,
        max_selected_items=3,
        required_categories=("server", "client", "network"),
        max_budget=50_000.0,
        ga_params={
            "pop_size": 12,
            "generations": 8,
            "mutation_rate": 0.04,
            "elite": 1,
            "seed": 9,
            "stagnation_limit": 20,
            "max_random_attempts": 80,
            "quality_check_enabled": False,
        },
    )

    assert summary["status"] == "ok"
    assert summary["totals"]["selected_count"] <= 3
    assert summary["totals"]["client_seats"] <= 3
    assert summary["selected_by_category"]["server"]
    assert summary["selected_by_category"]["client"]
    assert summary["selected_by_category"]["network"]
    assert all(constraint["passed"] for constraint in summary["constraints"])
