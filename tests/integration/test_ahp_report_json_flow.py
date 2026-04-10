import json
from pathlib import Path

from domain.decision.ahp import (
    DEFAULT_CONSTRAINTS,
    DEFAULT_EXPERT_MATRICES,
    DEFAULT_SOFT_CRITERIA,
    run_ahp_pipeline,
)
from infrastructure.storage import JsonFileStorage


def test_ahp_sample_report_can_be_built_and_saved_to_json(tmp_path: Path):
    root = Path(__file__).resolve().parents[2]
    sample_path = root / "data" / "examples" / "ahp" / "test_confs.json"
    report_path = tmp_path / "ahp_report.json"

    payload = json.loads(sample_path.read_text(encoding="utf-8"))
    report = run_ahp_pipeline(
        configurations=payload["configurations"],
        soft_criteria=DEFAULT_SOFT_CRITERIA,
        experts_criteria_matrices=DEFAULT_EXPERT_MATRICES,
        constraints=DEFAULT_CONSTRAINTS,
    )

    JsonFileStorage().write(report_path, report)
    saved = json.loads(report_path.read_text(encoding="utf-8"))

    assert report_path.exists()
    assert report["final"]["winner_id"] == "E"
    assert saved["final"]["winner_id"] == "E"
    assert saved["selection"]["selected_ids"] == ["E", "B", "C", "D"]
