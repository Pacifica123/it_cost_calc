import json

from infrastructure.exporters.ga_ahp_json_exporter import (
    build_ga_ahp_report_payload,
    export_ga_ahp_report_json,
)


def _sample_result():
    ahp_report = {
        "status": "ok",
        "assessment_mode": "independent_candidate_metrics",
        "agreement": {"status": "moderate", "summary": "Согласованность рассчитана."},
        "final": {
            "ranking": [
                {"id": "GA-1", "rank": 1, "name": "A", "score": 0.52},
                {"id": "GA-2", "rank": 2, "name": "B", "score": 0.48},
            ]
        },
        "independent_assessment": {
            "candidate_pool": [{"id": "GA-1", "name": "A"}, {"id": "GA-2", "name": "B"}],
            "criterion_values": {"GA-1": {"cost": 100.0}, "GA-2": {"cost": 120.0}},
            "criterion_utilities": {"GA-1": {"cost": 0.9}, "GA-2": {"cost": 0.8}},
        },
    }
    return {
        "status": "ok",
        "ahp_report": ahp_report,
        "export_payload": {
            "genetic_optimization": {
                "quality_check": {
                    "status": "ok",
                    "method": "exhaustive_enumeration",
                    "ga_best_matches_exact_best": True,
                }
            },
            "genetic_ahp_ranking": ahp_report,
            "ga_ahp_agreement": ahp_report["agreement"],
            "independent_assessment": ahp_report["independent_assessment"],
        },
    }


def test_build_ga_ahp_report_payload_exposes_key_sections():
    payload = build_ga_ahp_report_payload(_sample_result())

    assert payload["schema_version"] == 1
    assert payload["report_type"] == "ga_ahp_independent_assessment"
    assert payload["assessment_mode"] == "independent_candidate_metrics"
    assert payload["ga_ahp_agreement"]["status"] == "moderate"
    assert payload["ranking"][0]["id"] == "GA-1"
    assert payload["candidate_pool"][1]["name"] == "B"
    assert payload["ga_quality_check"]["ga_best_matches_exact_best"] is True
    assert payload["criterion_values"]["GA-1"]["cost"] == 100.0
    assert payload["criterion_utilities"]["GA-2"]["cost"] == 0.8


def test_export_ga_ahp_report_json_writes_utf8_file(tmp_path):
    path = export_ga_ahp_report_json(_sample_result(), tmp_path / "ga_ahp_report.json")

    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["ga_ahp_agreement"]["summary"] == "Согласованность рассчитана."
    assert data["ga_quality_check"]["method"] == "exhaustive_enumeration"
    assert data["independent_assessment"]["candidate_pool"][0]["id"] == "GA-1"
