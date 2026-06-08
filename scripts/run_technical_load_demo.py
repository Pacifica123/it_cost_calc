from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

EXPECTED = {
    "profile": "technical_load_1000",
    "technical_rows": 1000,
    "server_rows": 334,
    "client_rows": 333,
    "network_rows": 333,
    "candidate_count": 1000,
    "decision_pool_size": 8,
}


def _round(value: Any, digits: int = 6) -> float:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return 0.0


def _ga_params(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "pop_size": int(args.population),
        "generations": int(args.generations),
        "mutation_rate": float(args.mutation_rate),
        "seed": int(args.seed),
        "top_solutions_limit": int(args.top_solutions),
        "quality_check_enabled": False,
        "max_random_attempts": int(args.max_random_attempts),
        "stagnation_limit": int(args.stagnation_limit),
    }


def _count_technical_rows(dataset: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    return {
        "server": len(dataset.get("server", [])),
        "client": len(dataset.get("client", [])),
        "network": len(dataset.get("network", [])),
    }


def _validate(summary: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    counts = summary.get("dataset", {}).get("technical_rows_by_category", {})
    if summary.get("profile") != EXPECTED["profile"]:
        errors.append("profile mismatch")
    if counts.get("server") != EXPECTED["server_rows"]:
        errors.append("server row count mismatch")
    if counts.get("client") != EXPECTED["client_rows"]:
        errors.append("client row count mismatch")
    if counts.get("network") != EXPECTED["network_rows"]:
        errors.append("network row count mismatch")
    if summary.get("dataset", {}).get("technical_rows") < EXPECTED["technical_rows"]:
        errors.append("technical row count is below 1000")
    ga = summary.get("ga", {})
    if ga.get("status") != "ok":
        errors.append("GA status is not ok")
    if ga.get("candidate_count") != EXPECTED["candidate_count"]:
        errors.append("GA candidate_count mismatch")
    if ga.get("pool_count") != EXPECTED["decision_pool_size"]:
        errors.append("GA decision pool size mismatch")
    if ga.get("selected_count", 0) <= 0:
        errors.append("GA selected no items")
    if ga.get("quality_check_status") != "skipped":
        errors.append("GA quality check must be skipped for the 1000-item load profile")
    if summary.get("ahp", {}).get("status") != "ok":
        errors.append("AHP status is not ok")
    if summary.get("pareto", {}).get("status") != "ok":
        errors.append("Pareto status is not ok")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the 1000-row technical load demo through GA/AHP/Pareto."
    )
    parser.add_argument(
        "--profile",
        default="technical_load_1000",
        help="Demo profile id from data/fixtures/demo_profiles.json.",
    )
    parser.add_argument("--budget", type=float, default=2_400_000.0)
    parser.add_argument("--min-units", type=float, default=3.0)
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--mutation-rate", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-solutions", type=int, default=8)
    parser.add_argument("--max-random-attempts", type=int, default=200)
    parser.add_argument("--stagnation-limit", type=int, default=12)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Return non-zero when load-demo invariants do not match.",
    )
    args = parser.parse_args()

    from ui_qt.presenters.app_presenter import QtAppPresenter

    timings: dict[str, float] = {}
    with tempfile.TemporaryDirectory() as temp_dir:
        presenter = QtAppPresenter(
            repo_root=ROOT,
            runtime_entities_path=Path(temp_dir) / "runtime_entities.json",
        )

        start = perf_counter()
        dataset = presenter.load_demo_profile(args.profile)
        timings["load_seconds"] = perf_counter() - start

        technical_counts = _count_technical_rows(dataset)
        technical_total = sum(technical_counts.values())

        start = perf_counter()
        ga_result = presenter.run_genetic_optimization(
            scope="technical",
            max_budget=float(args.budget),
            target_units=float(args.min_units),
            ga_params=_ga_params(args),
        )
        timings["ga_seconds"] = perf_counter() - start

        start = perf_counter()
        ahp_result = presenter.run_ahp_for_candidate_pool("technical")
        timings["ahp_seconds"] = perf_counter() - start

        start = perf_counter()
        pareto_result = presenter.run_pareto_for_candidate_pool("technical")
        timings["pareto_seconds"] = perf_counter() - start

        pool = presenter.get_candidate_pool_snapshot("technical")
        ga_core = ga_result.get("ga_result", {}) if isinstance(ga_result.get("ga_result"), dict) else {}
        summary = {
            "scenario_id": "technical_load_1000_v1",
            "profile": args.profile,
            "status": "ok",
            "dataset": {
                "technical_rows": technical_total,
                "technical_rows_by_category": technical_counts,
                "total_rows": sum(len(rows) for rows in dataset.values()),
            },
            "parameters": {
                "budget": float(args.budget),
                "min_units": float(args.min_units),
                "ga": _ga_params(args),
            },
            "timings": {key: _round(value) for key, value in timings.items()},
            "ga": {
                "status": ga_result.get("status"),
                "candidate_count": ga_result.get("parameters", {}).get("candidate_count"),
                "pool_count": len(pool.candidates),
                "selected_count": ga_result.get("totals", {}).get("selected_count"),
                "selected_cost": _round(ga_result.get("totals", {}).get("capital_cost")),
                "selected_client_seats": _round(ga_result.get("totals", {}).get("client_seats")),
                "best_score": _round(ga_core.get("best_agg")),
                "generations_ran": ga_core.get("generations_ran"),
                "termination_reason": ga_core.get("termination_reason"),
                "quality_check_status": ga_core.get("quality_check", {}).get("status"),
                "unique_feasible_solutions_seen": ga_core.get("unique_feasible_solutions_seen"),
            },
            "ahp": {
                "status": ahp_result.get("status"),
                "candidate_count": ahp_result.get("candidate_count"),
                "winner_name": ahp_result.get("final", {}).get("winner_name"),
            },
            "pareto": {
                "status": pareto_result.get("status"),
                "ranked_count": len(pareto_result.get("ranking", [])),
                "winner_name": pareto_result.get("winner_name"),
                "nondominated_count": len(pareto_result.get("final_nondominated", [])),
            },
        }

    errors = _validate(summary)
    summary["status"] = "ok" if not errors else "failed"
    summary["validation_errors"] = errors
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not errors or not args.check_only else 1


if __name__ == "__main__":
    raise SystemExit(main())
