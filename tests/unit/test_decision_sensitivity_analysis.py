from application.services.decision_sensitivity_analysis_service import (
    DecisionSensitivityAnalysisService,
)
from infrastructure.exporters.sensitivity_dashboard_exporter import (
    build_decision_sensitivity_html,
    build_decision_sensitivity_payload,
)


def _report() -> dict:
    return {
        "project": {"title": "Sensitivity test"},
        "analysis_results": {
            "hybrid_assessment": {
                "by_scope": {
                    "technical": {
                        "status": "ok",
                        "lambda": 0.5,
                        "winner_id": "A",
                        "ranking": [
                            {
                                "id": "A",
                                "name": "Fast expensive",
                                "ga_rank": 1,
                                "ahp_rank": 2,
                                "ga_score": 10.0,
                                "ahp_score": 0.2,
                                "pareto_status": "недоминируемая",
                                "totals": {"tco": {"total_ownership_cost": 120.0}},
                            },
                            {
                                "id": "B",
                                "name": "Balanced cheap",
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


def test_lambda_sensitivity_switches_between_ahp_and_ga_leaders():
    scope = DecisionSensitivityAnalysisService().build(_report())["scopes"]["technical"]
    by_lambda = {row["lambda"]: row for row in scope["lambda_sweep"]}

    assert by_lambda[0.0]["winner_id"] == "B"
    assert by_lambda[1.0]["winner_id"] == "A"
    assert scope["stability"]["switch_count"] == 1
    assert scope["baseline"]["matches_original_hybrid"] is True


def test_tco_ceiling_filters_only_exported_pool():
    scope = DecisionSensitivityAnalysisService().build(_report())["scopes"]["technical"]
    cell = next(
        row
        for row in scope["grid"]
        if row["tco_limit"] == 80.0 and row["lambda"] == 1.0
    )

    assert cell["winner_id"] == "B"
    assert cell["feasible_count"] == 1


def test_missing_tco_is_kept_without_limit_and_excluded_with_finite_limit():
    report = _report()
    report["analysis_results"]["hybrid_assessment"]["by_scope"]["technical"]["ranking"][0][
        "totals"
    ] = {}
    scope = DecisionSensitivityAnalysisService().build(report)["scopes"]["technical"]

    no_limit = next(row for row in scope["grid"] if row["tco_limit"] is None and row["lambda"] == 1.0)
    finite = next(row for row in scope["grid"] if row["tco_limit"] == 80.0 and row["lambda"] == 1.0)

    assert no_limit["feasible_count"] == 2
    assert finite["feasible_count"] == 1
    assert scope["warnings"]


def test_sensitivity_export_is_standalone_and_explains_method_boundaries():
    payload = build_decision_sensitivity_payload(_report())
    html = build_decision_sensitivity_html(_report())

    assert payload["method"]["formula"].startswith("H_i(lambda)")
    assert "не перезапускает GA/AHP" in payload["method"]["budget_interpretation"]
    assert 'id="lambdaRange"' in html
    assert 'id="budgetSelect"' in html
    assert 'id="heatmap"' in html
    assert "Fast expensive" in html
    assert "https://" not in html
    assert "http://" not in html


def test_exact_lambda_solver_finds_analytical_breakpoint_and_exact_share():
    scope = DecisionSensitivityAnalysisService().build(_report())["scopes"]["technical"]
    exact = scope["exact_lambda"]

    assert exact["line_model"] == "affine_in_lambda_on_fixed_pool"
    assert 0.5 in exact["breakpoints"]
    assert exact["stability"]["method"] == "exact_lambda_measure"
    assert exact["stability"]["switch_count"] == 1
    shares = {row["id"]: row["share_pct"] for row in exact["stability"]["winner_shares"]}
    assert shares == {"A": 50.0, "B": 50.0}


def test_tco_grid_uses_fixed_baseline_normalization_and_keeps_subset_mode_diagnostic():
    report = _report()
    report["analysis_results"]["hybrid_assessment"]["by_scope"]["technical"] = {
        "status": "ok",
        "lambda": 0.1,
        "winner_id": "D",
        "ranking": [
            {"id": "A", "ga_rank": 3, "ahp_rank": 2, "ga_score": 0.0, "ahp_score": 1.0, "totals": {"tco": {"total_ownership_cost": 1.0}}},
            {"id": "B", "ga_rank": 2, "ahp_rank": 3, "ga_score": 1.0, "ahp_score": 0.0, "totals": {"tco": {"total_ownership_cost": 1.0}}},
            {"id": "C", "ga_rank": 1, "ahp_rank": 4, "ga_score": 2.0, "ahp_score": 0.0, "totals": {"tco": {"total_ownership_cost": 2.0}}},
            {"id": "D", "ga_rank": 4, "ahp_rank": 1, "ga_score": 0.0, "ahp_score": 16.0, "totals": {"tco": {"total_ownership_cost": 1.0}}},
        ],
    }

    scope = DecisionSensitivityAnalysisService().build(report)["scopes"]["technical"]
    fixed = next(row for row in scope["grid"] if row["tco_limit"] == 1.0 and row["lambda"] == 0.1)
    subset = next(
        row
        for row in scope["subset_renormalization_grid"]
        if row["tco_limit"] == 1.0 and row["lambda"] == 0.1
    )

    assert fixed["ranking_ids"].index("A") < fixed["ranking_ids"].index("B")
    assert subset["ranking_ids"].index("B") < subset["ranking_ids"].index("A")
    assert scope["normalization"]["primary_tco_mode"] == "fixed_baseline_normalization"
    assert any("Subset-renormalization" in warning for warning in scope["warnings"])
