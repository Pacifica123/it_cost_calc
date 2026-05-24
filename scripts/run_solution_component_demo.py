from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _demo_components() -> list[dict]:
    return [
        {
            "id": "software-subscription-crm",
            "name": "CRM-подписка для отдела продаж",
            "scope": "software",
            "component_type": "software_subscription",
            "strict_analysis_participation": True,
            "monthly_cost": 12000,
            "metrics": {
                "source_category": "subscription_licenses",
                "license_units": 8,
                "functionality_score": 0.78,
                "support_score": 0.8,
            },
            "cost_assumptions": ["Стоимость задана как ежемесячная подписка."],
        },
        {
            "id": "technical-workstation-basic",
            "name": "Рабочая станция менеджера",
            "scope": "technical",
            "component_type": "workstation",
            "strict_analysis_participation": True,
            "purchase_cost": 85000,
            "quantity": 1,
            "metrics": {
                "source_category": "client",
                "client_seats": 1,
                "max_power": 280,
            },
        },
        {
            "id": "mixed-sales-kit",
            "name": "Комплект рабочего места отдела продаж",
            "scope": "mixed",
            "component_type": "bundle",
            "strict_analysis_participation": True,
            "purchase_cost": 98000,
            "implementation_cost": 15000,
            "metadata": {
                "children": ["software-subscription-crm", "technical-workstation-basic"],
            },
            "cost_assumptions": ["Смешанный комплект используется как агрегированная альтернатива."],
        },
        {
            "id": "incomplete-draft",
            "name": "Черновик без области и типа",
            "purchase_cost": 10000,
        },
        {
            "id": "legacy_infrastructure:note-1",
            "name": "Свободная запись старой ИТ-песочницы",
            "scope": "technical",
            "component_type": "server",
            "strict_analysis_participation": True,
            "purchase_cost": 100000,
            "metadata": {"_legacy_sandbox": True},
        },
    ]


def _build_summary() -> dict:
    from application.services.candidate_configuration_service import CandidateConfigurationService
    from domain import to_plain_data
    from application.services.solution_component_normalization_service import (
        SolutionComponentNormalizationService,
    )

    components = _demo_components()
    service = SolutionComponentNormalizationService()
    normalized = service.normalize_many(components)
    candidate = service.to_candidate_configuration(
        components,
        candidate_id="solution_component_demo",
        name="Демонстрационный набор SolutionComponent",
    )
    candidate_via_existing_service = CandidateConfigurationService().from_solution_components(
        components,
        candidate_id="solution_component_demo_existing_service",
        name="Демонстрационный набор через CandidateConfigurationService",
    )
    software_cost = service.to_cost_model(components[0])
    technical_energy_input = service.to_energy_input(
        components[1],
        default_hours_per_day=8,
        default_working_days=22,
    )
    decision_snapshot = service.to_decision_report_component_snapshot(components[0])

    return {
        "normalized": [component.to_dict() for component in normalized],
        "candidate": candidate.to_dict(),
        "candidate_via_existing_service": candidate_via_existing_service.to_dict(),
        "software_cost_model": to_plain_data(software_cost),
        "technical_energy_input": technical_energy_input,
        "decision_snapshot": decision_snapshot,
    }


def _validate(summary: dict) -> list[str]:
    errors: list[str] = []
    normalized = summary["normalized"]
    by_id = {item["id"]: item for item in normalized}

    for component_id in (
        "software-subscription-crm",
        "technical-workstation-basic",
        "mixed-sales-kit",
    ):
        if not by_id[component_id]["candidate_eligible"]:
            errors.append(f"{component_id} must be candidate_eligible")
        if not by_id[component_id]["tco_eligible"]:
            errors.append(f"{component_id} must be tco_eligible")

    if by_id["incomplete-draft"]["candidate_eligible"]:
        errors.append("incomplete draft must not be candidate_eligible")
    if not by_id["incomplete-draft"]["blocking_errors"]:
        errors.append("incomplete draft must keep blocking_errors")

    if by_id["legacy_infrastructure:note-1"]["candidate_eligible"]:
        errors.append("legacy sandbox row must not be auto-included")

    candidate = summary["candidate"]
    if len(candidate["components"]) != 3:
        errors.append(f"candidate must contain 3 ready components, got {len(candidate['components'])}")
    if candidate["totals"]["tco"]["component_count"] != 3:
        errors.append("candidate TCO must be attached for 3 ready components")

    cost_model = summary["software_cost_model"]
    if cost_model["subscription_cost"] <= 0:
        errors.append("software component must produce subscription_cost")

    energy_input = summary["technical_energy_input"]
    if not energy_input or energy_input["max_power"] <= 0:
        errors.append("technical component must produce energy input")

    if not summary["decision_snapshot"]["analysis_ready"]:
        errors.append("decision report snapshot must keep readiness flags")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run SolutionComponent C2 model/normalization smoke scenario without a test runner."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full JSON snapshot instead of a compact summary.",
    )
    args = parser.parse_args()

    summary = _build_summary()
    errors = _validate(summary)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        candidate = summary["candidate"]
        print("SolutionComponent C2 smoke: OK")
        print(f"normalized={len(summary['normalized'])}")
        print(f"candidate_components={len(candidate['components'])}")
        print(f"blocked={candidate['metadata']['blocked_component_ids']}")
        print(f"tco={candidate['totals']['tco']['total_ownership_cost']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
