from application.services.decision_report_service import DecisionReportService
from infrastructure.exporters.decision_report_exporter import (
    build_decision_report_csv_rows,
    build_decision_report_markdown,
)
from shared.constants import ANALYSIS_SCOPE_TECHNICAL


def _candidate(candidate_id: str, *, pool_source: str) -> dict:
    return {
        "id": candidate_id,
        "name": f"Candidate {candidate_id}",
        "scope": ANALYSIS_SCOPE_TECHNICAL,
        "components": [],
        "totals": {"capital_cost": 100.0, "tco": {"total_ownership_cost": 120.0}},
        "metrics": {"ga_score": 0.8},
        "source": "ga",
        "metadata": {
            "candidate_pool_source": pool_source,
            "candidate_pool_method": "ga",
        },
    }


def test_decision_report_exports_scoped_ga_ahp_pareto_hybrid_as_separate_methods():
    service = DecisionReportService()
    report = service.build_report(
        candidate_configurations=[_candidate("A", pool_source="top-1 GA-кандидатов")],
        genetic_result={"by_scope": {ANALYSIS_SCOPE_TECHNICAL: {"candidate_configurations": [_candidate("A", pool_source="top-1 GA-кандидатов")]}}},
        ahp_result={"by_scope": {ANALYSIS_SCOPE_TECHNICAL: {"final": {"ranking": [("A", 1.0)]}}}},
        criteria_importance_result={"by_scope": {ANALYSIS_SCOPE_TECHNICAL: {"final_nondominated": ["A"]}}},
        hybrid_assessment_result={
            "by_scope": {
                ANALYSIS_SCOPE_TECHNICAL: {
                    "status": "ok",
                    "winner": {"id": "A", "name": "Candidate A", "hybrid_score": 0.92},
                    "ranking": [{"id": "A", "name": "Candidate A", "hybrid_score": 0.92}],
                }
            }
        },
    )

    assert set(["genetic_optimization", "ahp", "criteria_importance", "hybrid_assessment"]).issubset(
        report["analysis_results"]
    )
    assert report["winner_explanation"]["recommended"]["method"].startswith("Hybrid")
    assert report["candidate_configurations"][0]["metadata"]["candidate_pool_source"] == "top-1 GA-кандидатов"

    csv_rows = build_decision_report_csv_rows(report)
    assert csv_rows[0]["candidate_pool_source"] == "top-1 GA-кандидатов"
    assert csv_rows[0]["candidate_pool_method"] == "ga"

    markdown = build_decision_report_markdown(report)
    assert "GA формирует общий пул" in markdown
    assert "AHP ранжирует" in markdown
    assert "Hybrid" in markdown
    assert "GA + AHP и финансовая модель" not in markdown
