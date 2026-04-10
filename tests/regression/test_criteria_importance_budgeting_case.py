import json
from pathlib import Path

from domain.decision.criteria_importance.analysis import (
    load_default_budgeting_case,
    run_importance_pipeline,
)


def test_default_budgeting_case_matches_expected_regression_snapshot():
    root = Path(__file__).resolve().parents[2]
    expected_path = root / "tests" / "regression" / "fixtures" / "criteria_importance_expected.json"

    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    report = run_importance_pipeline(load_default_budgeting_case())

    assert report["winner_id"] == expected["winner_id"]
    assert report["winner_name"] == expected["winner_name"]
    assert [item["id"] for item in report["ranking"]] == expected["ranking_ids"]
    assert report["weighted_sum"] == expected["weighted_sum"]
    assert report["final_nondominated"] == expected["final_nondominated"]
    assert report["analysis_model"]["criterion_multiplicity"] == expected["criterion_multiplicity"]
