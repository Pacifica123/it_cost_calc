from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _component_ids(candidate: dict) -> list[str]:
    result: list[str] = []
    for item in candidate.get("components", []) or []:
        if not isinstance(item, dict):
            continue
        component_id = item.get("source_component_id") or item.get("id")
        if component_id:
            result.append(str(component_id))
    return result


def _build_summary(fixture_path: Path) -> dict:
    from application.services.decision_report_service import DecisionReportService
    from application.services.solution_component_analytics_service import (
        SolutionComponentAnalyticsIntegrationService,
    )
    from application.services.solution_component_runtime_service import (
        SolutionComponentRuntimeService,
    )
    from infrastructure.exporters.decision_report_exporter import (
        build_decision_report_markdown,
        build_solution_component_csv_rows,
    )
    from infrastructure.repositories.in_memory_entity_repository import InMemoryEntityRepository
    from shared.constants import ANALYSIS_SCOPE_SOFTWARE, ANALYSIS_SCOPE_TECHNICAL

    fixture = _read_json(fixture_path)
    entities = fixture.get("entities", fixture)
    if not isinstance(entities, dict):
        raise ValueError("Fixture must contain an 'entities' object or be an entities mapping")

    repository = InMemoryEntityRepository(entities)
    runtime_service = SolutionComponentRuntimeService(repository)
    analytics_service = SolutionComponentAnalyticsIntegrationService()

    components = runtime_service.list_components()
    by_id = {component.id: component for component in components}
    candidate = runtime_service.build_candidate_configuration(
        candidate_id="c9_solution_component_defense_candidate",
    )
    candidate["name"] = "C9: демонстрационный набор SolutionComponent"
    financial_chain = runtime_service.build_financial_chain()
    technical_pool = analytics_service.build_profile_pool_from_entities(
        repository.entities,
        analysis_scope=ANALYSIS_SCOPE_TECHNICAL,
        candidate_prefix="c9",
    )
    software_pool = analytics_service.build_profile_pool_from_entities(
        repository.entities,
        analysis_scope=ANALYSIS_SCOPE_SOFTWARE,
        candidate_prefix="c9",
    )
    report = DecisionReportService().build_report(
        project={
            "id": "solution-component-c9-defense-demo",
            "title": "C9: защитный сценарий редактора SolutionComponent",
            "goal": "Показать путь от расширенного редактора компонентов до DecisionReport без второй модели данных.",
        },
        entities=repository.entities,
        cost_totals=candidate.get("totals", {}),
        candidate_configurations=[candidate]
        + list(technical_pool.get("candidate_configurations", []))
        + list(software_pool.get("candidate_configurations", [])),
        metadata={"source": "scripts/run_solution_component_c9_demo.py"},
    )
    component_report = report.get("solution_component_report", {})
    counts = component_report.get("counts", {}) if isinstance(component_report, dict) else {}
    markdown = build_decision_report_markdown(report)
    component_csv_rows = build_solution_component_csv_rows(report)
    candidate_component_ids = _component_ids(candidate)

    mixed = by_id.get("c9-mixed-server-hypervisor-setup")
    software = by_id.get("c9-software-subscription-support")
    peripheral = by_id.get("c9-peripheral-mfu-no-client-seat")
    draft = by_id.get("c9-draft-unclassified-training")

    return {
        "scenario_id": fixture.get("scenario_id"),
        "component_count": len(components),
        "strict_component_count": sum(1 for component in components if component.candidate_eligible),
        "draft_component_count": sum(1 for component in components if not component.candidate_eligible),
        "component_ids": sorted(by_id),
        "candidate_component_count": len(candidate_component_ids),
        "candidate_component_ids": sorted(candidate_component_ids),
        "candidate_tco_total": candidate.get("totals", {}).get("tco", {}).get("total_ownership_cost"),
        "financial_chain_strict_count": financial_chain.get("strict_financial_component_count"),
        "financial_chain_excluded_count": financial_chain.get("excluded_component_count"),
        "technical_pool_ready": technical_pool.get("ready_component_count"),
        "technical_pool_excluded": technical_pool.get("excluded_component_count"),
        "software_pool_ready": software_pool.get("ready_component_count"),
        "software_pool_excluded": software_pool.get("excluded_component_count"),
        "report_valid_component_count": counts.get("valid"),
        "report_excluded_component_count": counts.get("excluded_or_draft"),
        "report_linked_component_count": counts.get("linked_to_alternatives"),
        "component_csv_row_count": len(component_csv_rows),
        "checks": {
            "mixed_bundle_energy_eligible": bool(mixed and mixed.energy_eligible),
            "mixed_bundle_candidate_eligible": bool(mixed and mixed.candidate_eligible),
            "software_subscription_energy_excluded": bool(software and not software.energy_eligible),
            "software_subscription_candidate_eligible": bool(software and software.candidate_eligible),
            "peripheral_client_seats_zero": bool(
                peripheral and float(peripheral.metrics.get("client_seats", -1)) == 0.0
            ),
            "peripheral_energy_eligible": bool(peripheral and peripheral.energy_eligible),
            "draft_excluded_from_candidate": bool(
                draft and not draft.candidate_eligible and draft.id not in candidate_component_ids
            ),
            "draft_has_warnings_or_errors": bool(
                draft and (draft.validation_warnings or draft.blocking_errors)
            ),
            "decision_report_has_solution_components": isinstance(component_report, dict)
            and counts.get("total") == len(components),
            "markdown_mentions_non_effect_of_drafts": "Неполные компоненты не влияли" in markdown,
            "component_csv_covers_all_components": len(component_csv_rows) == len(components),
        },
    }


def _validate(summary: dict, invariants_payload: dict) -> list[str]:
    expected = invariants_payload.get("expected", invariants_payload)
    errors: list[str] = []

    for key in (
        "component_count",
        "candidate_component_count",
    ):
        expected_value = expected.get(key)
        if expected_value is not None and summary.get(key) != expected_value:
            errors.append(f"{key} mismatch: {summary.get(key)} != {expected_value}")

    for key in (
        "strict_component_count",
        "draft_component_count",
        "report_valid_component_count",
        "report_excluded_component_count",
    ):
        minimum = expected.get(f"{key}_min")
        if minimum is not None and int(summary.get(key) or 0) < int(minimum):
            errors.append(f"{key} is below expected minimum: {summary.get(key)} < {minimum}")

    component_ids = set(summary.get("component_ids", []))
    for component_id in expected.get("required_component_ids", []):
        if component_id not in component_ids:
            errors.append(f"required component is absent: {component_id}")

    checks = summary.get("checks", {})
    for check_name, expected_value in expected.get("required_checks", {}).items():
        if bool(checks.get(check_name)) != bool(expected_value):
            errors.append(f"{check_name} mismatch: {checks.get(check_name)} != {expected_value}")

    for check_name in (
        "mixed_bundle_candidate_eligible",
        "software_subscription_candidate_eligible",
        "peripheral_energy_eligible",
        "draft_has_warnings_or_errors",
        "markdown_mentions_non_effect_of_drafts",
        "component_csv_covers_all_components",
    ):
        if not checks.get(check_name):
            errors.append(f"{check_name} is false")

    if float(summary.get("candidate_tco_total") or 0.0) <= 0.0:
        errors.append("candidate_tco_total must be positive")
    if int(summary.get("financial_chain_strict_count") or 0) < 3:
        errors.append("financial chain must include strict components")
    if int(summary.get("financial_chain_excluded_count") or 0) < 1:
        errors.append("financial chain must report excluded draft")
    if int(summary.get("technical_pool_ready") or 0) < 1:
        errors.append("technical analytics pool must include profile-ready component")
    if int(summary.get("software_pool_ready") or 0) < 1:
        errors.append("software analytics pool must include profile-ready component")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run final C9 SolutionComponent defense smoke scenario without pytest."
    )
    parser.add_argument(
        "--fixture",
        default=str(ROOT / "data" / "fixtures" / "solution_component_c9_demo_dataset.json"),
        help="Path to C9 SolutionComponent demonstration fixture.",
    )
    parser.add_argument(
        "--invariants",
        default=str(ROOT / "data" / "fixtures" / "regression" / "solution_component_c9_invariants.json"),
        help="Path to C9 expected invariants JSON.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full JSON summary instead of a compact result.",
    )
    args = parser.parse_args()

    summary = _build_summary(Path(args.fixture))
    errors = _validate(summary, _read_json(Path(args.invariants)))
    summary["status"] = "ok" if not errors else "failed"
    summary["validation_errors"] = errors

    if args.json or errors:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print("SolutionComponent C9 defense smoke: OK")
        print(f"components={summary['component_count']}")
        print(f"strict={summary['strict_component_count']}")
        print(f"drafts={summary['draft_component_count']}")
        print(f"report_valid={summary['report_valid_component_count']}")
        print(f"report_excluded={summary['report_excluded_component_count']}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
