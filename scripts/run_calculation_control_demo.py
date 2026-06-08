from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

EXPECTED = {
    "technical": {
        "ga_selected_names": ["T1 Сервер", "T2 Рабочее место", "T3 Сеть"],
        "ga_best_score": 1.0,
        "ahp_winner_name": "GA-кандидат 1",
        "pareto_winner_name": "GA-кандидат 1",
        "pareto_nondominated": ["GA-1"],
    },
    "software": {
        "ga_selected_names": ["S2 Pro"],
        "ga_best_score": 1.0,
        "ahp_winner_name": "GA-кандидат 1",
        "pareto_winner_name": "GA-кандидат 1",
        "pareto_nondominated": ["GA-1"],
    },
}


def _round(value: Any, digits: int = 6) -> float:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return 0.0


def _ga_params() -> dict[str, Any]:
    return {
        "pop_size": 24,
        "generations": 30,
        "mutation_rate": 0.04,
        "seed": 42,
        "top_solutions_limit": 8,
        "quality_check_enabled": True,
        "quality_check_max_items": 18,
        "max_random_attempts": 200,
    }


def _run_scope(presenter: Any, scope: str) -> dict[str, Any]:
    ga = presenter.run_genetic_optimization(
        scope=scope,
        max_budget=300.0,
        target_units=3.0,
        ga_params=_ga_params(),
    )
    ahp = presenter.run_ahp_for_candidate_pool(scope)
    pareto = presenter.run_pareto_for_candidate_pool(scope)

    return {
        "ga": {
            "selected_names": [str(item.get("name")) for item in ga.get("selected_items", [])],
            "best_score": _round((ga.get("ga_result") or {}).get("best_agg")),
            "best_cost": _round((ga.get("totals") or {}).get("capital_cost")),
            "quality_check_status": (ga.get("ga_result") or {}).get("quality_check", {}).get("status"),
            "exact_best_mask": list(
                (ga.get("ga_result") or {}).get("quality_check", {}).get("exact_best_mask", ())
            ),
            "candidate_scores": [
                {
                    "rank": candidate.get("rank"),
                    "score": _round(candidate.get("score")),
                    "names": [str(item.get("name")) for item in candidate.get("selected_items", [])],
                }
                for candidate in ga.get("candidate_solutions", [])
            ],
        },
        "ahp": {
            "winner_name": ahp.get("final", {}).get("winner_name"),
            "ranking": [
                {
                    "rank": row.get("rank"),
                    "name": row.get("name"),
                    "score": _round(row.get("score")),
                }
                for row in ahp.get("final", {}).get("ranking", [])
            ],
        },
        "pareto": {
            "winner_name": pareto.get("winner_name"),
            "final_nondominated": list(pareto.get("final_nondominated", [])),
            "ranking": [
                {
                    "id": row.get("id"),
                    "name": row.get("name"),
                    "weighted_sum": _round(row.get("weighted_sum")),
                    "nondominated": bool(row.get("nondominated")),
                }
                for row in pareto.get("ranking", [])
            ],
        },
    }


def _validate(summary: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for scope, expected in EXPECTED.items():
        actual = summary.get("scopes", {}).get(scope, {})
        ga = actual.get("ga", {})
        ahp = actual.get("ahp", {})
        pareto = actual.get("pareto", {})
        if ga.get("selected_names") != expected["ga_selected_names"]:
            errors.append(f"{scope}: GA selected_names mismatch")
        if ga.get("best_score") != expected["ga_best_score"]:
            errors.append(f"{scope}: GA best_score mismatch")
        if ga.get("quality_check_status") != "ok":
            errors.append(f"{scope}: GA quality_check_status is not ok")
        if ahp.get("winner_name") != expected["ahp_winner_name"]:
            errors.append(f"{scope}: AHP winner_name mismatch")
        if pareto.get("winner_name") != expected["pareto_winner_name"]:
            errors.append(f"{scope}: Pareto winner_name mismatch")
        if pareto.get("final_nondominated") != expected["pareto_nondominated"]:
            errors.append(f"{scope}: Pareto nondominated mismatch")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the tiny calculation-control demo for GA, AHP and Pareto."
    )
    parser.add_argument(
        "--profile",
        default="calculation_control",
        help="Demo profile id from data/fixtures/demo_profiles.json.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Return non-zero when expected answers do not match.",
    )
    args = parser.parse_args()

    from ui_qt.presenters.app_presenter import QtAppPresenter

    with tempfile.TemporaryDirectory() as temp_dir:
        presenter = QtAppPresenter(
            repo_root=ROOT,
            runtime_entities_path=Path(temp_dir) / "runtime_entities.json",
        )
        presenter.load_demo_profile(args.profile)
        summary = {
            "scenario_id": "ga_ahp_pareto_calculation_control_v1",
            "profile": args.profile,
            "scopes": {
                "technical": _run_scope(presenter, "technical"),
                "software": _run_scope(presenter, "software"),
            },
        }
    errors = _validate(summary)
    summary["status"] = "ok" if not errors else "failed"
    summary["validation_errors"] = errors
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not errors or not args.check_only else 1


if __name__ == "__main__":
    raise SystemExit(main())
