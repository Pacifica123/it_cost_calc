from infrastructure.exporters.interactive_dashboard_exporter import (
    build_interactive_dashboard_html,
    build_interactive_dashboard_payload,
)


def _candidate(candidate_id: str, *, tco: float, scope: str = "technical") -> dict:
    return {
        "id": candidate_id,
        "name": f"Candidate {candidate_id}",
        "scope": scope,
        "components": [{"id": f"part-{candidate_id}"}],
        "totals": {
            "capital_cost": tco * 0.6,
            "tco": {"annual_opex": tco * 0.1, "total_ownership_cost": tco},
        },
        "metrics": {"ga_score": 1.0 / tco},
        "metadata": {"candidate_pool_source": "test pool"},
    }


def _report() -> dict:
    return {
        "title": "Test report",
        "project": {"title": "Test", "goal": "Compare candidates"},
        "candidate_configurations": [
            _candidate("A", tco=100.0),
            _candidate("B", tco=120.0),
            _candidate("C", tco=90.0),
        ],
        "winner_explanation": {"recommended": {"id": "A", "name": "Candidate A", "method": "Hybrid"}},
        "analysis_results": {
            "ahp": {
                "by_scope": {
                    "technical": {
                        "final": {
                            "ranking": [
                                {"id": "A", "rank": 1, "score": 0.5},
                                {"id": "B", "rank": 2, "score": 0.3},
                                {"id": "C", "rank": 3, "score": 0.2},
                            ]
                        }
                    }
                }
            },
            "hybrid_assessment": {
                "by_scope": {
                    "technical": {
                        "ranking": [
                            {"id": "A", "rank": 1, "hybrid_score": 0.9},
                            {"id": "C", "rank": 2, "hybrid_score": 0.7},
                            {"id": "B", "rank": 3, "hybrid_score": 0.6},
                        ]
                    }
                }
            },
        },
        "catalog_data_quality": {
            "summary": {
                "catalog_components_total": 4,
                "complete_metrics": 3,
                "incomplete_metrics": 1,
                "with_warnings": 1,
                "with_manual_overrides": 1,
            }
        },
        "warnings": ["demo warning"],
        "risks": [],
        "metadata": {"generated_at": "2026-09-05T00:00:00+00:00"},
    }


def test_dashboard_payload_uses_rank_based_diagnostic_support():
    payload = build_interactive_dashboard_payload(_report())
    by_id = {row["id"]: row for row in payload["candidates"]}

    assert by_id["A"]["analysis_support"] == 100.0
    assert by_id["B"]["analysis_support"] == 25.0
    assert by_id["C"]["analysis_support"] == 25.0
    assert by_id["A"]["recommended"] is True
    assert payload["catalog_quality"]["incomplete"] == 1
    assert "не заменяет GA" in payload["analysis_notes"][1]


def test_dashboard_html_is_standalone_and_contains_interactive_controls():
    html = build_interactive_dashboard_html(_report())

    assert "<!doctype html>" in html.lower()
    assert 'id="scopeFilter"' in html
    assert 'id="candidateFilter"' in html
    assert 'id="scatter"' in html
    assert "Candidate A" in html
    assert "https://" not in html
    assert "http://" not in html


def test_dashboard_keeps_same_candidate_id_separate_between_scopes():
    report = _report()
    report["candidate_configurations"] = [
        _candidate("A", tco=100.0, scope="technical"),
        _candidate("A", tco=50.0, scope="software"),
    ]
    report["winner_explanation"] = {
        "recommended": {"id": "A", "name": "Candidate A", "method": "Hybrid", "scope": "technical"}
    }
    report["analysis_results"] = {
        "ahp": {
            "by_scope": {
                "technical": {"final": {"ranking": [{"id": "A", "rank": 1, "score": 1.0}]}},
                "software": {"final": {"ranking": [{"id": "A", "rank": 1, "score": 1.0}]}},
            }
        }
    }

    payload = build_interactive_dashboard_payload(report)
    by_key = {row["key"]: row for row in payload["candidates"]}

    assert set(by_key) == {"technical::A", "software::A"}
    assert by_key["technical::A"]["recommended"] is True
    assert by_key["software::A"]["recommended"] is False
    assert list(by_key["technical::A"]["method_evidence"].values())[0]["scope"] == "technical"
    assert list(by_key["software::A"]["method_evidence"].values())[0]["scope"] == "software"
