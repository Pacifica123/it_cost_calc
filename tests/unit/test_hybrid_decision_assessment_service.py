import pytest

from application.services.hybrid_decision_assessment_service import HybridDecisionAssessmentService
from shared.constants import ANALYSIS_SCOPE_TECHNICAL


def _candidate(candidate_id, *, rank, ga_score, cost=100.0):
    return {
        "id": candidate_id,
        "name": candidate_id,
        "scope": ANALYSIS_SCOPE_TECHNICAL,
        "components": [],
        "totals": {"capital_cost": cost},
        "metrics": {"ga_score": ga_score},
        "source": "ga",
        "metadata": {"rank": rank},
    }


def test_hybrid_assessment_waits_for_standalone_ahp_result():
    service = HybridDecisionAssessmentService()

    result = service.assess([_candidate("GA-1", rank=1, ga_score=0.9)])

    assert result["status"] == "incomplete"
    assert result["missing_methods"] == ["ahp"]
    assert "самостоятельный AHP" in result["reason"]
    assert result["ranking"][0]["comment"] == "ожидает AHP-результат"


def test_hybrid_assessment_combines_ga_ahp_and_pareto_status():
    service = HybridDecisionAssessmentService()
    candidates = [
        _candidate("GA-1", rank=1, ga_score=0.80, cost=120.0),
        _candidate("GA-2", rank=2, ga_score=0.60, cost=90.0),
    ]
    ahp_report = {"final": {"ranking": [("GA-2", 0.70), ("GA-1", 0.30)]}}
    pareto_report = {
        "final_nondominated": ["GA-2"],
        "ranking": [
            {"id": "GA-2", "name": "GA-2", "weighted_sum": 5.0, "nondominated": True},
            {"id": "GA-1", "name": "GA-1", "weighted_sum": 4.0, "nondominated": False},
        ],
    }

    result = service.assess(
        candidates,
        ahp_report=ahp_report,
        pareto_report=pareto_report,
        lambda_value=0.25,
        source_label="top-2 GA",
    )

    assert result["status"] == "ok"
    assert result["method"] == "scoped_pool_ga_ahp_with_pareto_status"
    assert result["winner_id"] == "GA-2"
    assert result["ranking"][0]["pareto_status"] == "недоминируемая"
    assert result["ranking"][0]["hybrid_score"] == pytest.approx(0.75)
    assert result["source_label"] == "top-2 GA"


def test_hybrid_assessment_warns_when_winner_is_pareto_dominated():
    service = HybridDecisionAssessmentService()
    candidates = [
        _candidate("GA-1", rank=1, ga_score=0.95),
        _candidate("GA-2", rank=2, ga_score=0.10),
    ]
    ahp_report = {"final": {"ranking": [("GA-1", 0.9), ("GA-2", 0.1)]}}
    pareto_report = {
        "ranking": [
            {"id": "GA-1", "nondominated": False},
            {"id": "GA-2", "nondominated": True},
        ],
        "final_nondominated": ["GA-2"],
    }

    result = service.assess(candidates, ahp_report=ahp_report, pareto_report=pareto_report)

    assert result["status"] == "ok"
    assert result["winner_id"] == "GA-1"
    assert result["winner"]["pareto_status"] == "доминируемая"
    assert any("Pareto" in warning and "доминируемый" in warning for warning in result["warnings"])
