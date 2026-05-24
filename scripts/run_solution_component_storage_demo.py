from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

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


def _build_summary(fixture_path: Path) -> dict:
    from application.services.decision_report_service import DecisionReportService
    from application.services.solution_component_runtime_service import (
        SolutionComponentRuntimeService,
    )
    from infrastructure.exporters.decision_report_exporter import (
        build_decision_report_csv_rows,
        build_decision_report_json_payload,
        build_decision_report_markdown,
    )
    from infrastructure.repositories.in_memory_entity_repository import InMemoryEntityRepository
    from infrastructure.repositories.json_entity_repository import JsonEntityRepository
    from shared.constants import SOLUTION_COMPONENT_ENTITY

    fixture = _read_json(fixture_path)
    entities = fixture.get("entities", fixture)
    if not isinstance(entities, dict):
        raise ValueError("Fixture must contain an 'entities' object or be an entities mapping")

    repository = InMemoryEntityRepository(entities)
    service = SolutionComponentRuntimeService(repository)
    snapshot = service.export_snapshot()
    candidate = service.build_candidate_configuration(candidate_id="runtime_solution_components")

    with tempfile.TemporaryDirectory() as temp_dir:
        runtime_path = Path(temp_dir) / "runtime_entities.json"
        json_repository = JsonEntityRepository(runtime_path)
        json_service = SolutionComponentRuntimeService(json_repository)
        json_service.replace_components(repository.list(SOLUTION_COMPONENT_ENTITY))
        reloaded_repository = JsonEntityRepository(runtime_path)
        reloaded_service = SolutionComponentRuntimeService(reloaded_repository)
        reloaded_snapshot = reloaded_service.export_snapshot()

    report = DecisionReportService().build_report(
        project={
            "id": "solution-component-storage-demo",
            "title": "Проверка runtime-хранения SolutionComponent",
            "goal": "Проверить хранение, загрузку и экспорт компонентов редактора.",
        },
        entities=repository.entities,
        cost_totals=candidate.get("totals", {}),
        candidate_configurations=[candidate],
        metadata={"source": "scripts/run_solution_component_storage_demo.py"},
    )
    markdown = build_decision_report_markdown(report)
    json_payload = build_decision_report_json_payload(report)
    csv_rows = build_decision_report_csv_rows(report)

    return {
        "snapshot": snapshot,
        "reloaded_snapshot": reloaded_snapshot,
        "candidate": candidate,
        "report_component_count": len(report.get("components", [])),
        "report_solution_component_count": sum(
            1
            for component in report.get("components", [])
            if isinstance(component, dict)
            and component.get("source_format") == "solution_component"
        ),
        "report_warning_count": len(report.get("warnings", [])),
        "markdown_has_solution_component_section": "## 4. Компоненты редактора" in markdown,
        "json_report_type": json_payload.get("report_type"),
        "csv_row_count": len(csv_rows),
    }


def _validate(summary: dict) -> list[str]:
    errors: list[str] = []
    snapshot = summary["snapshot"]
    reloaded = summary["reloaded_snapshot"]
    candidate = summary["candidate"]

    if snapshot["component_count"] < 4:
        errors.append("fixture must contain at least 4 solution components")
    if snapshot["strict_component_count"] < 3:
        errors.append("at least 3 solution components must be strict/candidate eligible")
    if snapshot["draft_component_count"] < 1:
        errors.append("at least 1 solution component must remain a draft")
    if snapshot != reloaded:
        errors.append("stored and reloaded solution component snapshots differ")
    if len(candidate.get("components", [])) != snapshot["strict_component_count"]:
        errors.append("candidate must include strict components only")
    if any(row.get("candidate_eligible") for row in snapshot["components"] if row["editor_status"] == "draft"):
        errors.append("draft component became candidate eligible")
    if summary["report_solution_component_count"] != snapshot["component_count"]:
        errors.append("DecisionReport must keep all editor components with statuses")
    if not summary["markdown_has_solution_component_section"]:
        errors.append("Markdown export must include editor component section")
    if summary["csv_row_count"] < 1:
        errors.append("CSV candidate export must stay compatible")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run SolutionComponent C3 runtime storage smoke scenario without pytest."
    )
    parser.add_argument(
        "--fixture",
        default=str(ROOT / "data" / "fixtures" / "demo_dataset.json"),
        help="Path to demonstration dataset JSON.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full JSON snapshot instead of a compact summary.",
    )
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
        snapshot = summary["snapshot"]
        print("SolutionComponent C3 storage smoke: OK")
        print(f"components={snapshot['component_count']}")
        print(f"strict={snapshot['strict_component_count']}")
        print(f"drafts={snapshot['draft_component_count']}")
        print(f"candidate_components={len(summary['candidate']['components'])}")
        print(f"report_warnings={summary['report_warning_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
