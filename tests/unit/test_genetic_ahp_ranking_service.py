from application.services.genetic_ahp_ranking_service import GeneticAhpRankingService
from application.services.genetic_optimization_service import GeneticOptimizationService
from application.use_cases.run_genetic_ahp_ranking import RunGeneticAhpRankingUseCase
from domain.optimization.ga import run_ga_mvp
from domain.optimization.mech import Item
from infrastructure.repositories.in_memory_entity_repository import InMemoryEntityRepository


def _sum_prop(prop):
    return lambda subset: sum(item.properties.get(prop, 0) for item in subset)


def _quality(subset):
    return sum(item.properties.get("quality", 0) for item in subset)


def _cost(subset):
    return sum(item.properties.get("total_cost", 0) for item in subset)


def test_ga_returns_top_candidate_solutions_for_downstream_ahp():
    result = run_ga_mvp(
        {
            "items": [
                Item(name="A", quality=6, price=30),
                Item(name="B", quality=10, price=60),
                Item(name="C", quality=7, price=25),
            ],
            "criteria": [
                {"name": "quality", "func": _sum_prop("quality"), "direction": "max"},
                {"name": "price", "func": _sum_prop("price"), "direction": "min"},
            ],
            "constraints": [
                {"name": "budget", "func": _sum_prop("price"), "operator": "<=", "bound": 90},
            ],
            "weights": [0.7, 0.3],
            "pop_size": 10,
            "generations": 6,
            "mutation_rate": 0.03,
            "elite": 1,
            "seed": 9,
            "stagnation_limit": 20,
            "top_solutions_limit": 5,
        }
    )

    assert result["error"] is None
    assert result["top_solutions"]
    assert result["top_solutions"][0]["rank"] == 1
    assert result["top_solutions"][0]["agg"] >= result["top_solutions"][-1]["agg"]
    assert "normalized_scores_by_criterion" in result["top_solutions"][0]


def test_genetic_ahp_service_ranks_ga_candidate_solutions():
    repository = InMemoryEntityRepository(
        {
            "server": [
                {"name": "Server A", "quantity": 1, "price": 100.0, "quality": 10},
                {"name": "Server B", "quantity": 1, "price": 140.0, "quality": 14},
            ],
            "client": [
                {"name": "Client A", "quantity": 3, "price": 20.0, "quality": 5},
                {"name": "Client B", "quantity": 4, "price": 28.0, "quality": 7},
            ],
            "network": [
                {"name": "Router A", "quantity": 1, "price": 35.0, "quality": 6},
            ],
        }
    )
    genetic_service = GeneticOptimizationService(repository)
    service = GeneticAhpRankingService(genetic_service)

    result = service.run(
        categories=("server", "client", "network"),
        criteria=[
            {"name": "quality", "func": _quality, "direction": "max"},
            {"name": "capital_cost", "func": _cost, "direction": "min"},
        ],
        constraints=[{"name": "min_quality", "func": _quality, "operator": ">=", "bound": 18}],
        weights=[0.6, 0.4],
        ahp_weights=[0.6, 0.4],
        max_budget=260.0,
        required_categories=("server", "client"),
        ga_params={
            "pop_size": 12,
            "generations": 8,
            "mutation_rate": 0.02,
            "elite": 1,
            "seed": 5,
            "stagnation_limit": 20,
            "max_random_attempts": 80,
            "top_solutions_limit": 4,
        },
    )

    assert result["status"] == "ok"
    report = result["ahp_report"]
    assert report["status"] == "ok"
    assert report["candidate_count"] >= 1
    assert report["criteria"]["names"] == ["quality", "capital_cost"]
    assert report["criteria"]["weights_source"] == "direct_ahp_weights"
    assert report["final"]["winner_id"]
    assert report["final"]["ranking"][0]["selected_items"]
    assert "genetic_ahp_ranking" in result["export_payload"]


def test_genetic_ahp_use_case_delegates_to_service():
    class DummyService:
        def __init__(self):
            self.received = None

        def run(self, **options):
            self.received = options
            return {"status": "ok", "ahp_report": {"status": "ok"}}

    service = DummyService()
    use_case = RunGeneticAhpRankingUseCase(service)

    result = use_case.execute(ahp_top_limit=3)

    assert result["status"] == "ok"
    assert service.received == {"ahp_top_limit": 3}
