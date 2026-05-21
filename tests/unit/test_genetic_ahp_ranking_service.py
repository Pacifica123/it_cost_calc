import pytest

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


def _assert_matrix_close(actual, expected, *, abs_tol=1e-9):
    assert len(actual) == len(expected)
    for actual_row, expected_row in zip(actual, expected):
        assert actual_row == pytest.approx(expected_row, abs=abs_tol)


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
    assert "ga_ahp_agreement" in result["export_payload"]
    assert "independent_assessment" in result["export_payload"]


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


def test_genetic_ahp_default_mode_uses_independent_candidate_metrics_with_anchor_bounds():
    class DummyGeneticService:
        def run(self, **options):
            return {
                "status": "ok",
                "candidate_solutions": [
                    {
                        "rank": 1,
                        "score": 0.66,
                        "selected_items": [{"name": "A"}],
                        "totals": {"capital_cost": 900.0},
                        "raw_scores_by_criterion": {"performance": 90.0, "cost": 900.0},
                        "directed_scores_by_criterion": {"performance": 90.0, "cost": -900.0},
                        "normalized_scores_by_criterion": {"performance": 1.0, "cost": 0.0},
                    },
                    {
                        "rank": 2,
                        "score": 0.64,
                        "selected_items": [{"name": "B"}],
                        "totals": {"capital_cost": 500.0},
                        "raw_scores_by_criterion": {"performance": 80.0, "cost": 500.0},
                        "directed_scores_by_criterion": {"performance": 80.0, "cost": -500.0},
                        "normalized_scores_by_criterion": {"performance": 0.8, "cost": 0.5},
                    },
                    {
                        "rank": 3,
                        "score": 0.57,
                        "selected_items": [{"name": "C"}],
                        "totals": {"capital_cost": 300.0},
                        "raw_scores_by_criterion": {"performance": 60.0, "cost": 300.0},
                        "directed_scores_by_criterion": {"performance": 60.0, "cost": -300.0},
                        "normalized_scores_by_criterion": {"performance": 0.6, "cost": 0.7},
                    },
                ],
                "ga_result": {
                    "criteria_metadata": [
                        {"name": "performance", "direction": "max"},
                        {"name": "cost", "direction": "min"},
                    ],
                    "criterion_names": ["performance", "cost"],
                    "criterion_directions": ["max", "min"],
                    "normalization_mins": [0.0, -1000.0],
                    "normalization_maxs": [100.0, 0.0],
                    "weights": [0.7, 0.3],
                },
                "export_payload": {"genetic_optimization": {"status": "ok"}},
            }

    service = GeneticAhpRankingService(DummyGeneticService())

    result = service.run(ahp_top_limit=3, saaty_cap=False)

    report = result["ahp_report"]
    assert report["status"] == "ok"
    assert report["assessment_mode"] == "independent_candidate_metrics"
    assert report["criteria"]["weights_source"] == "ga_weights"
    assert report["criteria"]["normalization_scope"] == ["ga_search_space", "ga_search_space"]

    _assert_matrix_close(
        report["alternatives"]["value_matrix"],
        [
            [90.0, 80.0, 60.0],
            [900.0, 500.0, 300.0],
        ],
    )
    _assert_matrix_close(
        report["alternatives"]["utility_matrix"],
        [
            [0.9, 0.8, 0.6],
            [0.1, 0.5, 0.7],
        ],
    )
    assert report["independent_assessment"]["ga_scores"] == [0.66, 0.64, 0.57]
    assert len(report["independent_assessment"]["ahp_scores"]) == 3
    assert report["independent_assessment"]["candidate_pool"][0]["name"] == "A"
    assert "genetic_ahp_ranking" in result["export_payload"]
    assert result["export_payload"]["ga_ahp_agreement"] == report["agreement"]
    assert result["export_payload"]["independent_assessment"] == report["independent_assessment"]
