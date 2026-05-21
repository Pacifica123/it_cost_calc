import pytest

from application.services.genetic_ahp_ranking_service import GeneticAhpRankingService


class _DummyGeneticService:
    def __init__(self, candidates, *, weight=1.0):
        self._candidates = candidates
        self._weight = weight

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
                "weights": [self._weight],
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


def test_hybrid_assessment_normalizes_ga_and_ahp_scores_before_combining():
    service = GeneticAhpRankingService(
        _DummyGeneticService(
            [
                _candidate(rank=1, score=10.0, cost=900.0),
                _candidate(rank=2, score=8.0, cost=500.0),
                _candidate(rank=3, score=6.0, cost=100.0),
            ]
        )
    )

    result = service.run(ahp_top_limit=3, saaty_cap=False)

    report = result["ahp_report"]
    hybrid = report["hybrid_assessment"]
    assert hybrid["status"] == "ok"
    assert hybrid["method"] == "weighted_normalized_ga_ahp_score"
    assert hybrid["lambda"] == pytest.approx(0.5)
    assert hybrid["lambda_label"] == "нейтральный компромисс"
    assert hybrid["normalization"]["ga"] == "minmax_over_candidate_pool"
    assert hybrid["normalization"]["ahp"] == "minmax_over_candidate_pool"

    by_id = {row["id"]: row for row in hybrid["ranking"]}
    assert by_id["GA-1"]["ga_score_normalized"] == pytest.approx(1.0)
    assert by_id["GA-3"]["ga_score_normalized"] == pytest.approx(0.0)
    assert by_id["GA-3"]["ahp_score_normalized"] == pytest.approx(1.0)

    for row in hybrid["ranking"]:
        expected = 0.5 * row["ga_score_normalized"] + 0.5 * row["ahp_score_normalized"]
        assert row["hybrid_score"] == pytest.approx(expected)
        assert row["ga_contribution"] == pytest.approx(0.5 * row["ga_score_normalized"])
        assert row["ahp_contribution"] == pytest.approx(0.5 * row["ahp_score_normalized"])
        assert row["score_disagreement"] == pytest.approx(
            abs(row["ga_score_normalized"] - row["ahp_score_normalized"])
        )

    assert result["export_payload"]["hybrid_assessment"] == hybrid


def test_hybrid_assessment_warns_when_ga_and_ahp_are_in_conflict():
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

    assert result["ahp_report"]["agreement"]["status"] == "conflict"
    hybrid = result["ahp_report"]["hybrid_assessment"]
    assert any("GA и AHP находятся в конфликте" in warning for warning in hybrid["warnings"])
    assert hybrid["winner"] is not None
    assert len(hybrid["ranking"]) == 6


def test_hybrid_assessment_handles_constant_score_series_without_crashing():
    service = GeneticAhpRankingService(
        _DummyGeneticService(
            [
                _candidate(rank=1, score=0.5, cost=500.0),
                _candidate(rank=2, score=0.5, cost=500.0),
            ]
        )
    )

    result = service.run(ahp_top_limit=2, saaty_cap=False)

    hybrid = result["ahp_report"]["hybrid_assessment"]
    assert hybrid["status"] == "ok"
    assert hybrid["normalization"]["ga_details"]["status"] == "constant_score"
    assert hybrid["normalization"]["ahp_details"]["status"] == "constant_score"
    assert [row["hybrid_score"] for row in hybrid["ranking"]] == pytest.approx([0.5, 0.5])
    assert any("GA-score не различает кандидатов" in warning for warning in hybrid["warnings"])
    assert any("AHP-score не различает кандидатов" in warning for warning in hybrid["warnings"])
