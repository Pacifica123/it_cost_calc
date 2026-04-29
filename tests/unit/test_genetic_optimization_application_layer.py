from application.services.genetic_optimization_service import GeneticOptimizationService
from application.use_cases.run_genetic_optimization import RunGeneticOptimizationUseCase
from infrastructure.repositories.in_memory_entity_repository import InMemoryEntityRepository


def _quality(subset):
    return sum(item.properties.get("quality", 0) for item in subset)


def _cost(subset):
    return sum(item.properties.get("total_cost", 0) for item in subset)


def test_service_converts_runtime_rows_to_generic_candidates_preserving_custom_properties():
    repository = InMemoryEntityRepository(
        {
            "server": [
                {
                    "name": "Server A",
                    "quantity": 1,
                    "price": 100.0,
                    "quality": 7,
                    "risk": 0.2,
                }
            ],
            "client": [
                {
                    "name": "Client B",
                    "quantity": 3,
                    "price": 20.0,
                    "quality": 4,
                    "risk": 0.1,
                }
            ],
        }
    )
    service = GeneticOptimizationService(repository)

    candidates = service.build_candidates(categories=("server", "client"))

    assert len(candidates) == 2
    assert candidates[0].properties["source_category"] == "server"
    assert candidates[0].properties["quality"] == 7
    assert candidates[0].properties["total_cost"] == 100.0
    assert candidates[1].properties["total_cost"] == 60.0
    assert candidates[1].properties["risk"] == 0.1


def test_service_runs_generic_ga_and_returns_ui_export_friendly_summary():
    repository = InMemoryEntityRepository(
        {
            "server": [
                {"name": "Server A", "quantity": 1, "price": 100.0, "quality": 10},
                {"name": "Server B", "quantity": 1, "price": 180.0, "quality": 12},
            ],
            "client": [
                {"name": "Client A", "quantity": 4, "price": 25.0, "quality": 8},
            ],
            "network": [
                {"name": "Router A", "quantity": 1, "price": 35.0, "quality": 6},
            ],
        }
    )
    service = GeneticOptimizationService(repository)

    summary = service.run(
        categories=("server", "client", "network"),
        criteria=[
            {"name": "quality", "func": _quality, "direction": "max"},
            {"name": "capital_cost", "func": _cost, "direction": "min"},
        ],
        constraints=[
            {"name": "min_quality", "func": _quality, "operator": ">=", "bound": 18},
        ],
        weights=[0.7, 0.3],
        max_budget=220.0,
        required_categories=("server", "client"),
        ga_params={
            "pop_size": 4,
            "generations": 3,
            "mutation_rate": 0.02,
            "elite": 1,
            "seed": 7,
            "stagnation_limit": 20,
            "max_random_attempts": 40,
        },
    )

    assert summary["status"] == "ok"
    assert summary["error"] is None
    assert summary["parameters"]["candidate_count"] == 4
    assert summary["parameters"]["criterion_names"] == ["quality", "capital_cost"]
    assert summary["totals"]["capital_cost"] <= 220.0
    assert summary["selected_by_category"]["server"]
    assert summary["selected_by_category"]["client"]
    assert "genetic_optimization" in summary["export_payload"]
    assert summary["export_payload"]["genetic_optimization"]["best_score"] == summary["ga_result"]["best_agg"]
    assert all(constraint["passed"] for constraint in summary["constraints"])


def test_use_case_delegates_to_configured_service():
    class DummyService:
        def __init__(self):
            self.received = None

        def run(self, **options):
            self.received = options
            return {"status": "ok", "options": options}

    service = DummyService()
    use_case = RunGeneticOptimizationUseCase(service)

    result = use_case.execute(max_budget=1000, ga_params={"seed": 1})

    assert result["status"] == "ok"
    assert service.received == {"max_budget": 1000, "ga_params": {"seed": 1}}


