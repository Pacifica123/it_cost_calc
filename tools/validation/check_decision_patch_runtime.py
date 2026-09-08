#!/usr/bin/env python3
"""Runtime smoke checks for the decision-hypothesis hardening patch.

This is deliberately pytest-free so devctl can validate the patched services in
minimal project environments.  It exercises the service-level behavior changed
by the patch; the broader pytest suite remains a conditional check when the dev
extras are installed.
"""

from __future__ import annotations

from math import isclose
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _sensitivity_report() -> dict:
    return {
        "project": {"title": "runtime-smoke"},
        "analysis_results": {
            "hybrid_assessment": {
                "by_scope": {
                    "technical": {
                        "status": "ok",
                        "lambda": 0.1,
                        "winner_id": "D",
                        "ranking": [
                            {
                                "id": "A",
                                "ga_rank": 3,
                                "ahp_rank": 2,
                                "ga_score": 0.0,
                                "ahp_score": 1.0,
                                "totals": {"tco": {"total_ownership_cost": 1.0}},
                            },
                            {
                                "id": "B",
                                "ga_rank": 2,
                                "ahp_rank": 3,
                                "ga_score": 1.0,
                                "ahp_score": 0.0,
                                "totals": {"tco": {"total_ownership_cost": 1.0}},
                            },
                            {
                                "id": "C",
                                "ga_rank": 1,
                                "ahp_rank": 4,
                                "ga_score": 2.0,
                                "ahp_score": 0.0,
                                "totals": {"tco": {"total_ownership_cost": 2.0}},
                            },
                            {
                                "id": "D",
                                "ga_rank": 4,
                                "ahp_rank": 1,
                                "ga_score": 0.0,
                                "ahp_score": 16.0,
                                "totals": {"tco": {"total_ownership_cost": 1.0}},
                            },
                        ],
                    }
                }
            }
        },
    }


def check_sensitivity_runtime() -> None:
    from application.services.decision_sensitivity_analysis_service import (
        DecisionSensitivityAnalysisService,
    )

    scope = DecisionSensitivityAnalysisService().build(_sensitivity_report())["scopes"]["technical"]
    assert scope["normalization"]["primary_tco_mode"] == "fixed_baseline_normalization"
    assert scope["exact_lambda"]["line_model"] == "affine_in_lambda_on_fixed_pool"

    fixed = next(
        row for row in scope["grid"] if row["tco_limit"] == 1.0 and row["lambda"] == 0.1
    )
    subset = next(
        row
        for row in scope["subset_renormalization_grid"]
        if row["tco_limit"] == 1.0 and row["lambda"] == 0.1
    )
    assert fixed["ranking_ids"].index("A") < fixed["ranking_ids"].index("B")
    assert subset["ranking_ids"].index("B") < subset["ranking_ids"].index("A")


def check_hybrid_runtime() -> None:
    from application.services.hybrid_decision_assessment_service import (
        HybridDecisionAssessmentService,
    )

    service = HybridDecisionAssessmentService()
    raw = [-2.0, 1.0, 7.0, 11.0]
    transformed = [7.0 * value - 13.0 for value in raw]
    normalized_raw, _ = service._normalize_scores(raw)
    normalized_transformed, _ = service._normalize_scores(transformed)
    assert all(
        isclose(left, right, rel_tol=0.0, abs_tol=1e-12)
        for left, right in zip(normalized_raw, normalized_transformed)
    )


def check_npv_runtime() -> None:
    from application.services.tco_model_service import TCOModelService
    from domain.finance.npv import calculate_npv

    service = TCOModelService()
    threshold = service.break_even_annual_effect(
        investment=1000.0,
        annual_opex=200.0,
        discount_rate=0.10,
        horizon_years=5,
    )["annual_effect"]
    npv = calculate_npv(1000.0, 0.10, [threshold - 200.0] * 5)
    assert isclose(npv, 0.0, rel_tol=0.0, abs_tol=1e-9)

    basis = service.build_npv_inputs(
        {"initial_investment": 1000.0, "annual_opex": 200.0},
        horizon_years=5,
        annual_effect=0.0,
        discount_rate=0.10,
    )
    assert basis["annual_effect"] == 0.0
    assert basis["break_even_annual_effect"] > basis["annual_opex"]


def check_ahp_runtime() -> None:
    from domain.decision.ahp.math_utils import (
        ahp_priority_and_consistency,
        build_pairwise_matrix_from_scores,
    )

    exact = build_pairwise_matrix_from_scores([1.0, 10.0, 100.0], saaty_cap=False)
    _weights, _lambda_max, _ci, exact_cr = ahp_priority_and_consistency(exact)
    clipped = build_pairwise_matrix_from_scores([1.0, 10.0, 100.0], saaty_cap=True)
    _weights, _lambda_max, _ci, clipped_cr = ahp_priority_and_consistency(clipped)
    assert abs(float(exact_cr)) < 1e-10
    assert isclose(float(clipped_cr), 0.4834773101905522, rel_tol=1e-9, abs_tol=1e-12)


def main() -> int:
    checks = [
        ("sensitivity", check_sensitivity_runtime),
        ("hybrid", check_hybrid_runtime),
        ("npv", check_npv_runtime),
        ("ahp", check_ahp_runtime),
    ]
    for name, check in checks:
        check()
        print(f"PASS {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
