import json
from pathlib import Path

from domain.decision.ahp.aggregation import (
    aggregate_configuration,
    filter_hard_constraints,
)
from domain.decision.ahp.defaults import DEFAULT_CONSTRAINTS


def test_filter_hard_constraints_keeps_only_expected_sample_configurations():
    root = Path(__file__).resolve().parents[2]
    payload = json.loads(
        (root / "data" / "examples" / "ahp" / "test_confs.json").read_text(encoding="utf-8")
    )
    aggregates = [
        aggregate_configuration(configuration) for configuration in payload["configurations"]
    ]

    passed, removed = filter_hard_constraints(aggregates, DEFAULT_CONSTRAINTS)

    assert [item["id"] for item in passed] == ["B", "C", "D", "E"]
    assert [item["aggregate"]["id"] for item in removed] == ["A", "F"]
    assert all(item["reasons"] == ["not_enough_clients"] for item in removed)
