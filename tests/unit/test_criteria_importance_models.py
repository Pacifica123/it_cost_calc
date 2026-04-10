from it_cost_calc.domain.decision.criteria_importance.analysis import (
    load_default_budgeting_case,
    run_importance_pipeline,
)


def test_criteria_importance_pipeline_returns_ranked_result():
    result = run_importance_pipeline(load_default_budgeting_case())

    assert result["winner_id"]
    assert result["ranking"]
    assert result["ranking"][0]["id"] == result["winner_id"]
    assert "criterion_multiplicity" in result["analysis_model"]
