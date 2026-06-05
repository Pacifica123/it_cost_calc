from domain.decision.criteria_importance.analysis import (
    load_default_budgeting_case,
    run_importance_pipeline,
)


def test_criteria_importance_pipeline_returns_ranked_result():
    result = run_importance_pipeline(load_default_budgeting_case())

    assert result["winner_id"]
    assert result["ranking"]
    assert result["ranking"][0]["id"] == result["winner_id"]
    assert "criterion_multiplicity" in result["analysis_model"]


def test_criteria_importance_accepts_single_candidate_pool_boundary():
    case_data = {
        "name": "single",
        "criteria": [
            {"id": "score", "name": "Оценка"},
            {"id": "support", "name": "Сопровождение"},
        ],
        "alternatives": [{"id": "GA-1", "name": "Единственный GA-кандидат"}],
        "scores": {"GA-1": {"score": 4.0, "support": 5.0}},
        "relations": [],
    }

    result = run_importance_pipeline(case_data)

    assert result["winner_id"] == "GA-1"
    assert result["final_nondominated"] == ["GA-1"]
    assert len(result["ranking"]) == 1
    assert result["ranking"][0]["nondominated"] is True
