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




def test_service_can_exclude_category_as_hard_filter():
    repository = InMemoryEntityRepository(
        {
            "server": [{"name": "Server A", "quantity": 1, "price": 100.0, "quality": 10}],
            "client": [{"name": "Client A", "quantity": 1, "price": 10.0, "quality": 1}],
            "network": [{"name": "Router A", "quantity": 1, "price": 30.0, "quality": 5}],
        }
    )
    service = GeneticOptimizationService(repository)

    summary = service.run(
        categories=("server", "client", "network"),
        criteria=[{"name": "quality", "func": _quality, "direction": "max"}],
        weights=[1.0],
        required_categories=("server",),
        excluded_categories=("client",),
        ga_params={
            "pop_size": 6,
            "generations": 4,
            "mutation_rate": 0.02,
            "elite": 1,
            "seed": 11,
            "stagnation_limit": 20,
            "max_random_attempts": 60,
        },
    )

    assert summary["status"] == "ok"
    assert "server" in summary["selected_by_category"]
    assert "client" not in summary["selected_by_category"]
    excluded = [
        constraint
        for constraint in summary["constraints"]
        if constraint["name"] == "excluded_category_client"
    ]
    assert excluded and excluded[0]["passed"] is True


def test_runtime_candidates_normalize_equipment_metrics_for_profile_criteria():
    repository = InMemoryEntityRepository(
        {
            "client": [
                {
                    "name": "Workstation",
                    "quantity": 2,
                    "price": 100.0,
                    "ram_score": 16,
                    "cpu_score": 8,
                    "storage_gb": 512,
                    "energy": 120,
                }
            ]
        }
    )
    service = GeneticOptimizationService(repository)

    candidate = service.build_candidates(categories=("client",))[0]

    assert candidate.properties["ram_gb"] == 16
    assert candidate.properties["cpu_cores"] == 8
    assert candidate.properties["storage_gb"] == 512
    assert candidate.properties["max_power"] == 120


def test_cost_tie_breaker_prefers_cheaper_candidate_only_when_quality_is_close():
    repository = InMemoryEntityRepository({"server": []})

    def fake_runner(params):
        expensive = {
            "mask": (1, 0),
            "items": [{"name": "A", "total_cost": 1000.0}],
            "raw_scores": [1.0],
            "normalized_scores": [1.0],
            "raw_scores_by_criterion": {"quality": 1.0},
            "directed_scores_by_criterion": {"quality": 1.0},
            "normalized_scores_by_criterion": {"quality": 1.0},
            "agg": 1.0,
            "criteria": [{"name": "quality", "raw_value": 1.0, "normalized_value": 1.0, "weight": 1.0}],
            "constraints": [],
            "rank": 1,
        }
        cheap = {
            "mask": (0, 1),
            "items": [{"name": "B", "total_cost": 100.0}],
            "raw_scores": [0.99],
            "normalized_scores": [0.99],
            "raw_scores_by_criterion": {"quality": 0.99},
            "directed_scores_by_criterion": {"quality": 0.99},
            "normalized_scores_by_criterion": {"quality": 0.99},
            "agg": 0.99,
            "criteria": [{"name": "quality", "raw_value": 0.99, "normalized_value": 0.99, "weight": 1.0}],
            "constraints": [],
            "rank": 2,
        }
        return {
            "error": None,
            "best_mask": expensive["mask"],
            "best_items": expensive["items"],
            "best_agg": expensive["agg"],
            "best_raw_scores": expensive["raw_scores"],
            "best_normalized_scores": expensive["normalized_scores"],
            "best_raw_scores_by_criterion": expensive["raw_scores_by_criterion"],
            "best_directed_scores_by_criterion": expensive["directed_scores_by_criterion"],
            "best_normalized_scores_by_criterion": expensive["normalized_scores_by_criterion"],
            "criteria": expensive["criteria"],
            "constraints": [],
            "top_solutions": [expensive, cheap],
            "criterion_names": ["quality"],
            "constraints_metadata": [],
        }

    service = GeneticOptimizationService(repository, ga_runner=fake_runner)

    summary = service.run(
        categories=("server",),
        criteria=[{"name": "quality", "func": _quality, "direction": "max"}],
        weights=[1.0],
        max_budget=1200.0,
        cost_tie_breaker_tolerance=0.02,
    )

    assert summary["selected_items"][0]["name"] == "B"
    assert summary["ga_result"]["cost_tie_breaker"]["status"] == "replaced"
    assert summary["ga_result"]["cost_tie_breaker"]["budget_savings"] == 1100.0


def test_cost_tie_breaker_does_not_replace_when_quality_gap_is_large():
    repository = InMemoryEntityRepository({"server": []})

    def fake_runner(params):
        expensive = {
            "mask": (1, 0),
            "items": [{"name": "A", "total_cost": 1000.0}],
            "raw_scores": [1.0],
            "normalized_scores": [1.0],
            "raw_scores_by_criterion": {"quality": 1.0},
            "directed_scores_by_criterion": {"quality": 1.0},
            "normalized_scores_by_criterion": {"quality": 1.0},
            "agg": 1.0,
            "criteria": [{"name": "quality", "raw_value": 1.0, "normalized_value": 1.0, "weight": 1.0}],
            "constraints": [],
            "rank": 1,
        }
        cheap = {
            "mask": (0, 1),
            "items": [{"name": "B", "total_cost": 100.0}],
            "raw_scores": [0.80],
            "normalized_scores": [0.80],
            "raw_scores_by_criterion": {"quality": 0.80},
            "directed_scores_by_criterion": {"quality": 0.80},
            "normalized_scores_by_criterion": {"quality": 0.80},
            "agg": 0.80,
            "criteria": [{"name": "quality", "raw_value": 0.80, "normalized_value": 0.80, "weight": 1.0}],
            "constraints": [],
            "rank": 2,
        }
        return {
            "error": None,
            "best_mask": expensive["mask"],
            "best_items": expensive["items"],
            "best_agg": expensive["agg"],
            "best_raw_scores": expensive["raw_scores"],
            "best_normalized_scores": expensive["normalized_scores"],
            "best_raw_scores_by_criterion": expensive["raw_scores_by_criterion"],
            "best_directed_scores_by_criterion": expensive["directed_scores_by_criterion"],
            "best_normalized_scores_by_criterion": expensive["normalized_scores_by_criterion"],
            "criteria": expensive["criteria"],
            "constraints": [],
            "top_solutions": [expensive, cheap],
            "criterion_names": ["quality"],
            "constraints_metadata": [],
        }

    service = GeneticOptimizationService(repository, ga_runner=fake_runner)

    summary = service.run(
        categories=("server",),
        criteria=[{"name": "quality", "func": _quality, "direction": "max"}],
        weights=[1.0],
        max_budget=1200.0,
        cost_tie_breaker_tolerance=0.02,
    )

    assert summary["selected_items"][0]["name"] == "A"
    assert summary["ga_result"]["cost_tie_breaker"]["status"] == "unchanged"
