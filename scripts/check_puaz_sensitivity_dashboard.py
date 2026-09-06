"""Standalone smoke check for the second PUAZ scenario/sensitivity export."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from infrastructure.exporters.sensitivity_dashboard_exporter import (  # noqa: E402
    build_decision_sensitivity_payload,
    export_decision_sensitivity_dashboard,
    export_decision_sensitivity_json,
)


def _report() -> dict:
    return {
        "project": {"title": "PUAZ sensitivity smoke"},
        "analysis_results": {
            "hybrid_assessment": {
                "by_scope": {
                    "technical": {
                        "lambda": 0.5,
                        "winner_id": "A",
                        "ranking": [
                            {
                                "id": "A",
                                "name": "A",
                                "ga_rank": 1,
                                "ahp_rank": 2,
                                "ga_score": 10.0,
                                "ahp_score": 0.2,
                                "pareto_status": "недоминируемая",
                                "totals": {"tco": {"total_ownership_cost": 120.0}},
                            },
                            {
                                "id": "B",
                                "name": "B",
                                "ga_rank": 2,
                                "ahp_rank": 1,
                                "ga_score": 5.0,
                                "ahp_score": 0.8,
                                "pareto_status": "недоминируемая",
                                "totals": {"tco": {"total_ownership_cost": 80.0}},
                            },
                        ],
                    }
                }
            }
        },
    }


def main() -> int:
    payload = build_decision_sensitivity_payload(_report())
    scope = payload["scopes"]["technical"]
    sweep = {row["lambda"]: row for row in scope["lambda_sweep"]}
    if sweep[0.0]["winner_id"] != "B" or sweep[1.0]["winner_id"] != "A":
        raise AssertionError("lambda extremes must expose AHP/GA winner switch")
    if not scope["baseline"]["matches_original_hybrid"]:
        raise AssertionError("baseline sensitivity recomputation must match stored Hybrid winner")

    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        html_path = export_decision_sensitivity_dashboard(
            _report(), root / "decision_sensitivity_dashboard.html"
        )
        json_path = export_decision_sensitivity_json(_report(), root / "decision_sensitivity.json")
        html = html_path.read_text(encoding="utf-8")
        required = ('id="lambdaRange"', 'id="budgetSelect"', 'id="heatmap"', "PUAZ")
        missing = [marker for marker in required if marker not in html]
        if missing:
            raise AssertionError(f"sensitivity HTML missing markers: {missing}")
        if "https://" in html or "http://" in html:
            raise AssertionError("sensitivity dashboard must have no external HTTP dependencies")
        parsed = json.loads(json_path.read_text(encoding="utf-8"))
        if parsed["schema_version"] != 1:
            raise AssertionError("unexpected sensitivity JSON schema")

    print("PUAZ sensitivity dashboard smoke: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
