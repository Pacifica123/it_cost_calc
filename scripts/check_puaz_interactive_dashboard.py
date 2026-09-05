"""Standalone smoke check for the PUAZ interactive DecisionReport dashboard."""

from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from infrastructure.exporters.interactive_dashboard_exporter import (  # noqa: E402
    build_interactive_dashboard_payload,
    export_interactive_dashboard,
)


def _candidate(candidate_id: str, rank: int, tco: float) -> dict:
    return {
        "id": candidate_id,
        "name": f"PUAZ {candidate_id}",
        "scope": "technical",
        "totals": {
            "capital_cost": tco * 0.7,
            "tco": {"annual_opex": tco * 0.08, "total_ownership_cost": tco},
        },
        "metrics": {"ga_score": 1.0 / rank},
        "metadata": {"rank": rank, "candidate_pool_source": "PUAZ smoke"},
    }


def main() -> int:
    report = {
        "project": {"title": "PUAZ smoke", "goal": "Проверка интерактивной визуализации"},
        "candidate_configurations": [
            _candidate("A", 1, 1000.0),
            _candidate("B", 2, 850.0),
        ],
        "winner_explanation": {"recommended": {"id": "A", "name": "PUAZ A", "method": "Hybrid"}},
        "analysis_results": {
            "ahp": {
                "by_scope": {
                    "technical": {
                        "final": {
                            "ranking": [
                                {"id": "A", "rank": 1, "score": 0.6},
                                {"id": "B", "rank": 2, "score": 0.4},
                            ]
                        }
                    }
                }
            },
            "hybrid_assessment": {
                "by_scope": {
                    "technical": {
                        "ranking": [
                            {"id": "A", "rank": 1, "hybrid_score": 0.8},
                            {"id": "B", "rank": 2, "hybrid_score": 0.7},
                        ]
                    }
                }
            },
        },
        "catalog_data_quality": {
            "summary": {
                "catalog_components_total": 2,
                "complete_metrics": 1,
                "incomplete_metrics": 1,
                "with_warnings": 1,
                "with_manual_overrides": 0,
            }
        },
        "warnings": [],
        "risks": [],
    }

    payload = build_interactive_dashboard_payload(report)
    candidates = {row["id"]: row for row in payload["candidates"]}
    if candidates["A"]["analysis_support"] != 100.0:
        raise AssertionError("expected rank leader to receive 100 diagnostic support")
    if candidates["B"]["analysis_support"] != 0.0:
        raise AssertionError("expected last candidate to receive 0 diagnostic support")

    with TemporaryDirectory() as tmp:
        path = export_interactive_dashboard(report, Path(tmp) / "decision_dashboard.html")
        html = path.read_text(encoding="utf-8")
        required = ('id="scopeFilter"', 'id="candidateFilter"', 'id="scatter"', "PUAZ A")
        missing = [marker for marker in required if marker not in html]
        if missing:
            raise AssertionError(f"dashboard HTML missing markers: {missing}")
        if "https://" in html or "http://" in html:
            raise AssertionError("dashboard must stay standalone without external HTTP dependencies")

    print("PUAZ interactive dashboard smoke: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
