from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _analytics_components(base_components: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extend the fixture in memory so C6 can check profile pools deterministically."""

    extras = [
        {
            "id": "component-tech-sales-workstation",
            "name": "Рабочая станция менеджера продаж",
            "scope": "technical",
            "component_type": "workstation",
            "origin": "demo",
            "strict_analysis_participation": True,
            "purchase_cost": 72000.0,
            "quantity": 1,
            "metrics": {
                "source_category": "client",
                "client_seats": 1,
                "max_power": 95.0,
                "performance_score": 0.62,
                "reliability_score": 0.76,
                "lifespan": 4.0,
            },
        },
        {
            "id": "component-tech-access-switch",
            "name": "Коммутатор доступа отдела продаж",
            "scope": "technical",
            "component_type": "network_device",
            "origin": "demo",
            "strict_analysis_participation": True,
            "purchase_cost": 33000.0,
            "quantity": 1,
            "metrics": {
                "source_category": "network",
                "max_power": 45.0,
                "performance_score": 0.54,
                "reliability_score": 0.79,
                "lifespan": 5.0,
            },
        },
        {
            "id": "component-software-office-license",
            "name": "Офисная лицензия для пользователя",
            "scope": "software",
            "component_type": "software_license",
            "origin": "demo",
            "strict_analysis_participation": True,
            "purchase_cost": 18500.0,
            "quantity": 5,
            "metrics": {
                "source_category": "licenses",
                "license_units": 5,
                "functionality_score": 0.64,
                "support_score": 0.7,
                "reliability_score": 0.74,
                "lifespan": 3.0,
            },
        },
    ]
    return [*base_components, *extras]


def _build_summary(fixture_path: Path) -> dict[str, Any]:
    from application.services.genetic_ahp_ranking_service import GeneticAhpRankingService
    from application.services.genetic_optimization_service import GeneticOptimizationService
    from application.services.solution_component_analytics_service import (
        SolutionComponentAnalyticsIntegrationService,
    )
    from domain.decision.ahp import (
        DEFAULT_CONSTRAINTS,
        DEFAULT_EXPERT_MATRICES,
        DEFAULT_SOFT_CRITERIA,
        run_ahp_pipeline,
    )

    fixture = _read_json(fixture_path)
    entities = fixture.get("entities", fixture)
    if not isinstance(entities, dict):
        raise ValueError("Fixture must contain an 'entities' object or be an entities mapping")

    components = _analytics_components(list(entities.get("solution_components", [])))
    analytics_entities = {"solution_components": components}
    service = SolutionComponentAnalyticsIntegrationService()
    technical_pool = service.build_profile_pool(components, analysis_scope="technical")
    software_pool = service.build_profile_pool(components, analysis_scope="software")

    technical_ahp = run_ahp_pipeline(
        configurations=technical_pool["ahp_configurations"],
        soft_criteria=("total_performance", "avg_reliability", "lifespan"),
        experts_criteria_matrices=(
            np.array([[1.0, 2.0, 3.0], [1 / 2, 1.0, 2.0], [1 / 3, 1 / 2, 1.0]]),
            np.array([[1.0, 3.0, 2.0], [1 / 3, 1.0, 1 / 2], [1 / 2, 2.0, 1.0]]),
        ),
        constraints={"people_match_tolerance": 1.0},
    )

    control_ahp = run_ahp_pipeline(
        configurations=json.loads(
            (ROOT / "data" / "examples" / "ahp" / "test_confs.json").read_text(
                encoding="utf-8"
            )
        )["configurations"],
        soft_criteria=DEFAULT_SOFT_CRITERIA,
        experts_criteria_matrices=DEFAULT_EXPERT_MATRICES,
        constraints=DEFAULT_CONSTRAINTS,
    )

    ga_service = GeneticOptimizationService()
    ga_result = ga_service.run(
        source=analytics_entities,
        analysis_scope="technical",
        categories=("server", "client", "network"),
        target_units=1,
        max_power=700.0,
        max_budget=320000.0,
        required_categories=("server", "client", "network"),
        min_selected_items=1,
        ga_params={
            "pop_size": 14,
            "generations": 14,
            "mutation_rate": 0.08,
            "elite": 1,
            "seed": 42,
            "stagnation_limit": 8,
            "top_solutions_limit": 4,
            "quality_check_enabled": False,
        },
    )
    hybrid_result = GeneticAhpRankingService(ga_service).run(
        source=analytics_entities,
        analysis_scope="technical",
        categories=("server", "client", "network"),
        target_units=1,
        max_power=700.0,
        max_budget=320000.0,
        required_categories=("server", "client", "network"),
        min_selected_items=1,
        ahp_top_limit=4,
        ga_params={
            "pop_size": 14,
            "generations": 14,
            "mutation_rate": 0.08,
            "elite": 1,
            "seed": 42,
            "stagnation_limit": 8,
            "top_solutions_limit": 4,
            "quality_check_enabled": False,
        },
    )

    return {
        "technical_pool": technical_pool,
        "software_pool": software_pool,
        "technical_ahp": technical_ahp,
        "control_ahp": control_ahp,
        "ga_result": ga_result,
        "hybrid_result": hybrid_result,
    }


def _contains_solution_component(items: list[dict[str, Any]]) -> bool:
    return any(item.get("source_format") == "solution_component" for item in items)


def _validate(summary: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    technical_pool = summary["technical_pool"]
    software_pool = summary["software_pool"]
    technical_ahp = summary["technical_ahp"]
    control_ahp = summary["control_ahp"]
    ga_result = summary["ga_result"]
    hybrid_result = summary["hybrid_result"]

    if technical_pool["ready_component_count"] < 3:
        errors.append("technical profile pool must include server, workstation and network SolutionComponent rows")
    if software_pool["ready_component_count"] < 2:
        errors.append("software profile pool must include subscription/license SolutionComponent rows")
    if not technical_pool["candidate_configurations"]:
        errors.append("technical pool must expose CandidateConfiguration alternatives")
    if not technical_pool["ahp_configurations"]:
        errors.append("technical pool must expose AHP configurations through CandidateConfiguration")
    if any(
        item.get("normalization_state") == "draft"
        for item in technical_pool["candidate_configurations"]
    ):
        errors.append("draft SolutionComponent entered CandidateConfiguration alternatives")
    if not any(row.get("scope") == "mixed" for row in technical_pool["excluded_components"]):
        errors.append("mixed bundle must be excluded from a profile-specific pool with reasons")
    if technical_ahp.get("passed_count", 0) < 1 or "winner_id" not in technical_ahp.get("final", {}):
        errors.append("AHP must rank SolutionComponent alternatives")
    if control_ahp.get("final", {}).get("winner_id") != "E":
        errors.append("control AHP example winner changed")
    if ga_result.get("status") != "ok":
        errors.append("GA must run with SolutionComponent candidates")
    if not _contains_solution_component(ga_result.get("selected_items", [])):
        errors.append("GA selected set must include SolutionComponent item properties")
    if not ga_result.get("candidate_configurations"):
        errors.append("GA result must expose CandidateConfiguration payloads")
    if hybrid_result.get("status") != "ok":
        errors.append("GA + AHP hybrid ranking must finish over the same GA candidate pool")
    if not hybrid_result.get("ahp_report", {}).get("candidate_configurations"):
        errors.append("GA + AHP must preserve CandidateConfiguration payloads")
    if not any(
        item.get("source_format") == "solution_component"
        for candidate in hybrid_result.get("ahp_report", {}).get("candidate_configurations", [])
        for item in candidate.get("components", [])
    ):
        errors.append("GA + AHP CandidateConfiguration payloads must retain SolutionComponent markers")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run SolutionComponent C6 analytics integration smoke scenario without pytest."
    )
    parser.add_argument(
        "--fixture",
        default=str(ROOT / "data" / "fixtures" / "demo_dataset.json"),
        help="Path to demonstration dataset JSON.",
    )
    parser.add_argument("--json", action="store_true", help="Print full JSON summary.")
    args = parser.parse_args()

    summary = _build_summary(Path(args.fixture))
    errors = _validate(summary)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print("SolutionComponent C6 analytics smoke: OK")
        print(f"technical_ready={summary['technical_pool']['ready_component_count']}")
        print(f"software_ready={summary['software_pool']['ready_component_count']}")
        print(f"ahp_winner={summary['technical_ahp']['final']['winner_id']}")
        print(f"ga_selected={len(summary['ga_result']['selected_items'])}")
        print(f"hybrid_candidates={summary['hybrid_result']['ahp_report']['candidate_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
