import json
from pathlib import Path

import pytest

from it_cost_calc.domain.decision.ahp import (
    DEFAULT_CONSTRAINTS,
    DEFAULT_EXPERT_MATRICES,
    DEFAULT_SOFT_CRITERIA,
    run_ahp_pipeline,
)


def test_ahp_sample_case_matches_expected_regression_snapshot():
    root = Path(__file__).resolve().parents[2]
    sample_path = root / "data" / "examples" / "ahp" / "test_confs.json"
    expected_path = root / "tests" / "regression" / "fixtures" / "ahp_sample_expected.json"

    payload = json.loads(sample_path.read_text(encoding="utf-8"))
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    report = run_ahp_pipeline(
        configurations=payload["configurations"],
        soft_criteria=DEFAULT_SOFT_CRITERIA,
        experts_criteria_matrices=DEFAULT_EXPERT_MATRICES,
        constraints=DEFAULT_CONSTRAINTS,
    )

    assert report["final"]["winner_id"] == expected["winner_id"]
    assert [item["id"] for item in report["passed"]] == expected["passed_ids"]
    assert [item["aggregate"]["id"] for item in report["removed_details"]] == expected[
        "removed_ids"
    ]
    assert [item[0] for item in report["final"]["ranking"]] == expected["ranking_ids"]
    assert report["criteria"]["CR"] == pytest.approx(expected["criteria_cr"])
    assert report["criteria"]["weights"] == pytest.approx(expected["criteria_weights"])
    assert report["final"]["normalized_final_scores"] == pytest.approx(
        expected["normalized_final_scores_by_passed_order"]
    )
    assert [item[1] for item in report["final"]["ranking"]] == pytest.approx(
        expected["ranking_scores"]
    )
