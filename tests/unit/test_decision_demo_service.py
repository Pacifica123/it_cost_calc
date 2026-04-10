import json
from pathlib import Path

from application.services.decision_demo_service import DecisionDemoDataService


def test_decision_demo_service_builds_consistent_payload_from_demo_fixture():
    root = Path(__file__).resolve().parents[2]
    fixture_path = root / "data" / "fixtures" / "demo_dataset.json"
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))

    service = DecisionDemoDataService()
    result = service.build(payload["entities"])

    assert len(result["configurations"]) == 5
    assert {criterion["id"] for criterion in result["criteria_case"]["criteria"]} == {
        "1",
        "2",
        "3",
        "4",
        "5",
    }
    assert len(result["criteria_case"]["alternatives"]) == len(result["configurations"])
    assert result["constraints"]["max_budget"] > 0
    assert result["constraints"]["max_energy"] > 0

    first_config = result["configurations"][0]
    assert first_config["devices"]
    assert all("lifespan" in device for device in first_config["devices"])
