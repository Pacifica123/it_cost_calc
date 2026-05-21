from application.services.genetic_ahp_ranking_service import GeneticAhpRankingService


class _DummyGeneticService:
    def __init__(self, candidates):
        self._candidates = candidates

    def run(self, **options):
        return {
            "status": "ok",
            "candidate_solutions": self._candidates,
            "ga_result": {
                "criteria_metadata": [{"name": "cost", "direction": "min"}],
                "criterion_names": ["cost"],
                "criterion_directions": ["min"],
                "normalization_mins": [-1000.0],
                "normalization_maxs": [0.0],
                "weights": [1.0],
            },
            "export_payload": {"genetic_optimization": {"status": "ok"}},
        }


def _candidate(rank, score, cost):
    return {
        "rank": rank,
        "score": score,
        "selected_items": [{"name": f"Config {rank}"}],
        "totals": {"capital_cost": cost},
        "raw_scores_by_criterion": {"cost": cost},
        "directed_scores_by_criterion": {"cost": -cost},
        "normalized_scores_by_criterion": {"cost": max(0.0, 1.0 - cost / 1000.0)},
    }


def test_agreement_report_marks_ahp_winner_outside_ga_top_5_as_conflict():
    service = GeneticAhpRankingService(
        _DummyGeneticService(
            [
                _candidate(rank=1, score=0.95, cost=900.0),
                _candidate(rank=2, score=0.90, cost=800.0),
                _candidate(rank=3, score=0.85, cost=700.0),
                _candidate(rank=4, score=0.80, cost=600.0),
                _candidate(rank=5, score=0.75, cost=500.0),
                _candidate(rank=6, score=0.70, cost=100.0),
            ]
        )
    )

    result = service.run(ahp_top_limit=6, saaty_cap=False)

    report = result["ahp_report"]
    agreement = report["agreement"]
    assert agreement["status"] == "conflict"
    assert agreement["status_label"] == "конфликт оценок"
    assert agreement["winner_ga_rank"] == 6
    assert agreement["winner_rank_delta"] == -5
    assert agreement["max_rank_delta"] == 5
    assert agreement["top_3_overlap"] == 0
    assert agreement["warnings"]
    assert agreement["rank_deltas"][-1]["id"] == "GA-6"
    assert agreement["rank_deltas"][-1]["ahp_rank"] == 1
    assert agreement["winner_comparison"]["same_winner"] is False


def test_agreement_report_marks_close_rankings_as_high_agreement():
    service = GeneticAhpRankingService(
        _DummyGeneticService(
            [
                _candidate(rank=1, score=0.95, cost=100.0),
                _candidate(rank=2, score=0.90, cost=200.0),
                _candidate(rank=3, score=0.85, cost=300.0),
            ]
        )
    )

    result = service.run(ahp_top_limit=3, saaty_cap=False)

    agreement = result["ahp_report"]["agreement"]
    assert agreement["status"] == "high"
    assert agreement["top_3_overlap"] == 3
    assert agreement["max_rank_delta"] == 0
    assert agreement["winner_comparison"]["same_winner"] is True


def test_agreement_report_warns_about_inactive_constant_criteria():
    class ConstantCriteriaGeneticService:
        def run(self, **options):
            return {
                "status": "ok",
                "candidate_solutions": [
                    {
                        "rank": 1,
                        "score": 0.9,
                        "selected_items": [{"name": "A"}],
                        "raw_scores_by_criterion": {
                            "category_coverage": 4.0,
                            "client_capacity": 6.0,
                            "total_power_watts": 0.0,
                            "capital_cost": 500.0,
                        },
                        "directed_scores_by_criterion": {
                            "category_coverage": 4.0,
                            "client_capacity": 6.0,
                            "total_power_watts": -0.0,
                            "capital_cost": -500.0,
                        },
                    },
                    {
                        "rank": 2,
                        "score": 0.8,
                        "selected_items": [{"name": "B"}],
                        "raw_scores_by_criterion": {
                            "category_coverage": 4.0,
                            "client_capacity": 4.0,
                            "total_power_watts": 0.0,
                            "capital_cost": 400.0,
                        },
                        "directed_scores_by_criterion": {
                            "category_coverage": 4.0,
                            "client_capacity": 4.0,
                            "total_power_watts": -0.0,
                            "capital_cost": -400.0,
                        },
                    },
                ],
                "ga_result": {
                    "criteria_metadata": [
                        {"name": "category_coverage", "direction": "max"},
                        {"name": "client_capacity", "direction": "max"},
                        {"name": "total_power_watts", "direction": "min"},
                        {"name": "capital_cost", "direction": "min"},
                    ],
                    "criterion_names": [
                        "category_coverage",
                        "client_capacity",
                        "total_power_watts",
                        "capital_cost",
                    ],
                    "criterion_directions": ["max", "max", "min", "min"],
                    "normalization_mins": [3.0, 3.0, -0.0, -1000.0],
                    "normalization_maxs": [4.0, 6.0, -0.0, 0.0],
                    "weights": [0.35, 0.25, 0.15, 0.25],
                },
                "export_payload": {"genetic_optimization": {"status": "ok"}},
            }

    service = GeneticAhpRankingService(ConstantCriteriaGeneticService())

    result = service.run(ahp_top_limit=2)

    report = result["ahp_report"]
    diagnostics = report["criterion_diagnostics"]
    inactive_names = [row["criterion"] for row in diagnostics["inactive_criteria"]]
    assert diagnostics["status"] == "has_inactive_criteria"
    assert inactive_names == ["category_coverage", "total_power_watts"]
    assert diagnostics["inactive_criteria_count"] == 2
    assert diagnostics["active_criteria_count"] == 2
    assert (
        "Критерий total_power_watts не влияет на ранжирование: у всех кандидатов значение 0."
        in diagnostics["warnings"]
    )
    assert report["agreement"]["criterion_diagnostics"] == diagnostics
    assert report["agreement"]["warnings"][:2] == diagnostics["warnings"]


def test_winner_interpretation_explains_score_and_metric_tradeoff():
    class TradeoffGeneticService:
        def run(self, **options):
            return {
                "status": "ok",
                "candidate_solutions": [
                    {
                        "rank": 1,
                        "score": 0.85,
                        "selected_items": [{"name": "Powerful"}],
                        "totals": {"capital_cost": 500000.0},
                        "raw_scores_by_criterion": {
                            "client_capacity": 6.0,
                            "capital_cost": 500000.0,
                        },
                        "directed_scores_by_criterion": {
                            "client_capacity": 6.0,
                            "capital_cost": -500000.0,
                        },
                    },
                    {
                        "rank": 2,
                        "score": 0.78,
                        "selected_items": [{"name": "Budget"}],
                        "totals": {"capital_cost": 400000.0},
                        "raw_scores_by_criterion": {
                            "client_capacity": 4.0,
                            "capital_cost": 400000.0,
                        },
                        "directed_scores_by_criterion": {
                            "client_capacity": 4.0,
                            "capital_cost": -400000.0,
                        },
                    },
                ],
                "ga_result": {
                    "criteria_metadata": [
                        {"name": "client_capacity", "direction": "max"},
                        {"name": "capital_cost", "direction": "min"},
                    ],
                    "criterion_names": ["client_capacity", "capital_cost"],
                    "criterion_directions": ["max", "min"],
                    "normalization_mins": [3.0, -600000.0],
                    "normalization_maxs": [6.0, -350000.0],
                    "weights": [0.3, 0.7],
                },
                "export_payload": {"genetic_optimization": {"status": "ok"}},
            }

    service = GeneticAhpRankingService(TradeoffGeneticService())

    result = service.run(ahp_top_limit=2, saaty_cap=False)

    interpretation = result["ahp_report"]["agreement"]["winner_interpretation"]
    assert interpretation["ahp_winner_id"] == "GA-2"
    assert interpretation["ga_winner_id"] == "GA-1"
    assert interpretation["ahp_score_delta_percent"] > 0
    assert interpretation["ga_score_delta_percent"] < 0
    assert "по AHP score" in interpretation["summary"]
    assert "по GA score" in interpretation["summary"]
    assert "стоимости" in interpretation["summary"]
    assert "клиентских мест" in interpretation["summary"]
    assert result["export_payload"]["ga_ahp_agreement"]["winner_interpretation"] == interpretation
