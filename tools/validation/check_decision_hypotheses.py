#!/usr/bin/env python3
"""Independent audit harness for decision-analysis hypotheses H1-H6.

This script is intentionally outside the application runtime.  H1-H5 are
checked with exact ``fractions.Fraction`` arithmetic.  H6 additionally imports
the current AHP math implementation to reproduce the clipping effect against
real project code.

Run from repository root:
    PYTHONPATH=src python tools/validation/check_decision_hypotheses.py
"""

from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
import sys
from typing import Callable, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _hybrid(ga: Fraction, ahp: Fraction, lam: Fraction) -> Fraction:
    return lam * ga + (1 - lam) * ahp


def _minmax(values: Iterable[Fraction]) -> list[Fraction]:
    values = list(values)
    lo = min(values)
    hi = max(values)
    if len(values) == 1:
        return [Fraction(1)]
    if hi == lo:
        return [Fraction(1, 2) for _ in values]
    return [(value - lo) / (hi - lo) for value in values]


def _pair_crossing(
    left_ga: Fraction,
    left_ahp: Fraction,
    right_ga: Fraction,
    right_ahp: Fraction,
) -> tuple[str, Fraction | None]:
    left_slope = left_ga - left_ahp
    right_slope = right_ga - right_ahp
    slope_delta = left_slope - right_slope
    intercept_delta = right_ahp - left_ahp
    if slope_delta == 0:
        return ("identical" if intercept_delta == 0 else "parallel", None)
    return ("single_crossing", intercept_delta / slope_delta)


def _rank_scores(
    rows: Mapping[str, tuple[Fraction, Fraction]],
    lam: Fraction,
) -> list[tuple[str, Fraction]]:
    ga_values = _minmax(ga for ga, _ in rows.values())
    ahp_values = _minmax(ahp for _, ahp in rows.values())
    scored = [
        (candidate_id, _hybrid(ga_values[index], ahp_values[index], lam))
        for index, candidate_id in enumerate(rows)
    ]
    return sorted(scored, key=lambda item: (item[1], item[0]), reverse=True)


def check_h1() -> dict[str, object]:
    relation, crossing = _pair_crossing(Fraction(1), Fraction(0), Fraction(0), Fraction(1))
    assert relation == "single_crossing"
    assert crossing == Fraction(1, 2)

    identical_relation, identical_crossing = _pair_crossing(
        Fraction(3, 4), Fraction(1, 4), Fraction(3, 4), Fraction(1, 4)
    )
    assert identical_relation == "identical"
    assert identical_crossing is None
    return {
        "status": "proved_by_exact_algebra_and_checked_example",
        "fixed_pool_affine": True,
        "example_crossing": "1/2",
        "pair_crossing_bound": "<=1 unless score lines are identical",
        "sampled_sweep_required_for_breakpoints": False,
    }


def check_h2() -> dict[str, object]:
    values = [Fraction(0), Fraction(1, 4), Fraction(1, 2), Fraction(3, 4), Fraction(1)]
    cases = 0
    for ga_a in values:
        for ga_b in values:
            for ahp_a in values:
                for ahp_b in values:
                    if ga_a < ga_b or ahp_a < ahp_b:
                        continue
                    for lam in values:
                        assert _hybrid(ga_a, ahp_a, lam) >= _hybrid(ga_b, ahp_b, lam)
                        cases += 1
    return {
        "status": "proved_by_convex_combination; finite_property_grid_passed",
        "checked_cases": cases,
        "numeric_reversal_under_unanimous_evidence": False,
    }


def check_h3() -> dict[str, object]:
    source = [Fraction(-2), Fraction(1), Fraction(7), Fraction(11)]
    transformed = [Fraction(7) * value - Fraction(13) for value in source]
    source_norm = _minmax(source)
    transformed_norm = _minmax(transformed)
    assert source_norm == transformed_norm
    return {
        "status": "proved_for_positive_affine_transform_on_fixed_nonconstant_pool",
        "normalized_values": [str(value) for value in source_norm],
        "candidate_context_invariant": False,
    }


def check_h4() -> dict[str, object]:
    # C is removed by the TCO ceiling.  It is not the full-pool Hybrid winner,
    # yet removing it changes the GA min-max range and reverses A/B under
    # subset renormalization.
    full = {
        "A": (Fraction(0), Fraction(1)),
        "B": (Fraction(1), Fraction(0)),
        "C": (Fraction(2), Fraction(0)),
        "D": (Fraction(0), Fraction(16)),
    }
    feasible = {key: value for key, value in full.items() if key != "C"}
    lam = Fraction(1, 10)

    full_rank = _rank_scores(full, lam)
    subset_rank = _rank_scores(feasible, lam)
    full_scores = dict(full_rank)
    subset_scores = dict(subset_rank)
    assert full_rank[0][0] == "D"
    assert full_scores["A"] > full_scores["B"]
    assert subset_scores["B"] > subset_scores["A"]

    # Fixed baseline normalization must retain the original A/B relationship.
    full_ids = list(full)
    ga_norm = dict(zip(full_ids, _minmax(value[0] for value in full.values())))
    ahp_norm = dict(zip(full_ids, _minmax(value[1] for value in full.values())))
    fixed_scores = {
        key: _hybrid(ga_norm[key], ahp_norm[key], lam)
        for key in feasible
    }
    assert fixed_scores["A"] > fixed_scores["B"]

    return {
        "status": "refuted",
        "lambda": "1/10",
        "removed_candidate": "C",
        "full_pool_winner": full_rank[0][0],
        "full_pool_A_gt_B": True,
        "subset_renormalized_B_gt_A": True,
        "fixed_baseline_A_gt_B": True,
        "interpretation": "feasibility_effect and normalization_context_effect must be separated",
    }


def check_h5() -> dict[str, object]:
    investment = Fraction(1000)
    annual_opex = Fraction(120)
    rate = Fraction(1, 10)
    horizon = 4
    annuity = sum((Fraction(1) / (1 + rate) ** year for year in range(1, horizon + 1)), Fraction(0))
    break_even_effect = annual_opex + investment / annuity
    npv = -investment + sum(
        ((break_even_effect - annual_opex) / (1 + rate) ** year for year in range(1, horizon + 1)),
        Fraction(0),
    )
    assert npv == 0
    return {
        "status": "proved_by_exact_inversion",
        "formula": "E*=O+I/A",
        "annuity_factor": str(annuity),
        "break_even_effect": str(break_even_effect),
        "npv_at_break_even": str(npv),
        "is_forecast_of_actual_effect": False,
    }


def check_h6() -> dict[str, object]:
    from domain.decision.ahp.math_utils import (
        ahp_priority_and_consistency,
        build_pairwise_matrix_from_scores,
    )

    scores = [1.0, 10.0, 100.0]
    exact_matrix = build_pairwise_matrix_from_scores(scores, saaty_cap=False)
    priorities, _lambda_max, _ci, exact_cr = ahp_priority_and_consistency(exact_matrix)
    expected = [1.0 / 111.0, 10.0 / 111.0, 100.0 / 111.0]
    assert all(abs(float(actual) - target) < 1e-10 for actual, target in zip(priorities, expected))
    assert abs(float(exact_cr)) < 1e-10

    clipped_matrix = build_pairwise_matrix_from_scores(scores, saaty_cap=True)
    _priorities, _lambda_max, _ci, clipped_cr = ahp_priority_and_consistency(clipped_matrix)
    assert abs(float(clipped_cr) - 0.4834773101905522) < 1e-9
    return {
        "status": "exact_ratio_proved; saaty_cap_consistency_claim_refuted",
        "exact_ratio_cr": float(exact_cr),
        "clipped_cr": float(clipped_cr),
        "clipping_can_create_consistency_distortion": True,
    }


def _run(name: str, check: Callable[[], dict[str, object]]) -> dict[str, object]:
    try:
        return {"ok": True, **check()}
    except Exception as exc:  # pragma: no cover - command-line reporting path
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    results = {
        "H1_EXACT_LAMBDA_GEOMETRY": _run("H1", check_h1),
        "H2_DOMINANCE_PRESERVATION": _run("H2", check_h2),
        "H3_SCALE_INVARIANCE": _run("H3", check_h3),
        "H4_TCO_ORDER_INVARIANCE": _run("H4", check_h4),
        "H5_NPV_INVERSION": _run("H5", check_h5),
        "H6_SCORE_DERIVED_AHP": _run("H6", check_h6),
    }
    lean_files = [str(path.relative_to(ROOT)) for path in ROOT.rglob("*.lean")]
    results["FORMAL_STATUS"] = {
        "lean_source_files": lean_files,
        "kernel_verified_by_this_tool": False,
        "note": "This audit uses algebra/property checks and current Python runtime, not Lean kernel verification.",
    }
    passed = all(bool(result.get("ok")) for key, result in results.items() if key.startswith("H"))
    payload = {"passed": passed, "results": results}
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
