import json
from pathlib import Path

from it_cost_calc.domain.decision.ahp import (
    DEFAULT_CONSTRAINTS,
    DEFAULT_EXPERT_MATRICES,
    DEFAULT_SOFT_CRITERIA,
    run_ahp_pipeline,
)


def test_ahp_pipeline_returns_winner_for_sample_data():
    root = Path(__file__).resolve().parents[2]
    sample_path = root / "data" / "examples" / "ahp" / "test_confs.json"
    payload = json.loads(sample_path.read_text(encoding="utf-8"))
    configurations = payload.get("configurations", payload)

    report = run_ahp_pipeline(
        configurations=configurations,
        soft_criteria=DEFAULT_SOFT_CRITERIA,
        experts_criteria_matrices=DEFAULT_EXPERT_MATRICES,
        constraints=DEFAULT_CONSTRAINTS,
    )

    assert report["passed_count"] >= 1
    assert report["final"]["winner_id"]
    assert report["selection"]["selected_count"] >= 1
