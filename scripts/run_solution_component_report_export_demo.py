from __future__ import annotations

import argparse
import csv
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
    from application.services.solution_component_runtime_service import SolutionComponentRuntimeService
    from infrastructure.exporters.decision_report_exporter import (
        build_decision_report_csv_rows,
        build_decision_report_json_payload,
        build_decision_report_markdown,
        build_solution_component_csv_rows,
        export_decision_report_csv,
        export_decision_report_json,
        export_decision_report_markdown,
        export_solution_component_csv,
    )
    from infrastructure.repositories.in_memory_entity_repository import InMemoryEntityRepository
    from shared.constants import SOLUTION_COMPONENT_ENTITY

    fixture = _read_json(fixture_path)
    entities = fixture.get("entities", fixture)
    if not isinstance(entities, dict):
        raise ValueError("Fixture must contain an 'entities' object or be an entities mapping")

    repository = InMemoryEntityRepository(entities)
    runtime_service = SolutionComponentRuntimeService(repository)
    runtime_candidate = runtime_service.build_candidate_configuration(
        candidate_id="report_export_solution_components",
    )
    runtime_candidate["name"] = "Компоненты редактора для отчёта"
    report = DecisionReportService().build_report(
        project={
            "id": "solution-component-report-export-demo",
            "title": "Проверка DecisionReport и экспорта SolutionComponent",
            "goal": "Проверить объяснимый вывод компонентов редактора в JSON, Markdown и CSV.",
        },
        entities=repository.entities,
        cost_totals=runtime_candidate.get("totals", {}),
        candidate_configurations=[runtime_candidate],
        metadata={"source": "scripts/run_solution_component_report_export_demo.py"},
    )
    markdown = build_decision_report_markdown(report)
    json_payload = build_decision_report_json_payload(report)
    candidate_rows = build_decision_report_csv_rows(report)
    component_rows = build_solution_component_csv_rows(report)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        json_path = export_decision_report_json(report, temp_root / "decision_report.json")
        md_path = export_decision_report_markdown(report, temp_root / "decision_report.md")
        candidate_csv_path = export_decision_report_csv(
            report,
            temp_root / "decision_report_candidates.csv",
        )
        component_csv_path = export_solution_component_csv(
            report,
            temp_root / "decision_report_solution_components.csv",
        )
        exported_component_rows = _read_csv(component_csv_path)
        exported_candidate_rows = _read_csv(candidate_csv_path)
        exported_json = _read_json(json_path)
        exported_markdown = md_path.read_text(encoding="utf-8")

    solution_report = report.get("solution_component_report", {})
    counts = solution_report.get("counts", {}) if isinstance(solution_report, dict) else {}
    return {
        "component_count": counts.get("total"),
        "valid_component_count": counts.get("valid"),
        "excluded_component_count": counts.get("excluded_or_draft"),
        "linked_component_count": counts.get("linked_to_alternatives"),
        "report_has_solution_component_report": isinstance(solution_report, dict),
        "json_has_solution_components": isinstance(json_payload.get("solution_components"), dict),
        "json_keeps_full_report": isinstance(json_payload.get("report"), dict),
        "json_report_type": json_payload.get("report_type"),
        "markdown_has_role_explanation": "SolutionComponent используется как расширенный ввод" in markdown,
        "markdown_has_valid_table": "### 4.1. Валидные компоненты" in markdown,
        "markdown_has_excluded_table": "### 4.2. Исключённые компоненты" in markdown,
        "markdown_explains_drafts": "Неполные компоненты не влияли" in markdown,
        "candidate_csv_row_count": len(candidate_rows),
        "component_csv_row_count": len(component_rows),
        "component_csv_has_reasons": any(row.get("warnings_or_reasons") for row in component_rows),
        "component_csv_valid_linked": any(row.get("candidate_ids") for row in component_rows),
        "exported_component_csv_row_count": len(exported_component_rows),
        "exported_candidate_csv_row_count": len(exported_candidate_rows),
        "exported_json_has_solution_components": isinstance(
            exported_json.get("solution_components"),
            dict,
        ),
        "exported_markdown_matches": exported_markdown == markdown,
        "entity_component_count": len(repository.list(SOLUTION_COMPONENT_ENTITY)),
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def _validate(summary: dict) -> list[str]:
    errors: list[str] = []
    if summary.get("component_count", 0) < 4:
        errors.append("DecisionReport must include all SolutionComponent rows")
    if summary.get("valid_component_count", 0) < 3:
        errors.append("DecisionReport must include valid SolutionComponent rows")
    if summary.get("excluded_component_count", 0) < 1:
        errors.append("DecisionReport must include excluded/draft SolutionComponent rows")
    if summary.get("linked_component_count", 0) < 1:
        errors.append("valid SolutionComponent rows must be linked to CandidateConfiguration")
    for key in (
        "report_has_solution_component_report",
        "json_has_solution_components",
        "json_keeps_full_report",
        "markdown_has_role_explanation",
        "markdown_has_valid_table",
        "markdown_has_excluded_table",
        "markdown_explains_drafts",
        "component_csv_has_reasons",
        "component_csv_valid_linked",
        "exported_json_has_solution_components",
        "exported_markdown_matches",
    ):
        if not summary.get(key):
            errors.append(f"{key} is false")
    if summary.get("candidate_csv_row_count", 0) < 1:
        errors.append("candidate CSV export must stay compatible")
    if summary.get("component_csv_row_count") != summary.get("component_count"):
        errors.append("component CSV must contain every SolutionComponent row")
    if summary.get("exported_component_csv_row_count") != summary.get("component_csv_row_count"):
        errors.append("exported component CSV row count mismatch")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run SolutionComponent C7 DecisionReport/export smoke scenario without pytest."
    )
    parser.add_argument(
        "--fixture",
        default=str(ROOT / "data" / "fixtures" / "demo_dataset.json"),
        help="Path to demonstration dataset JSON.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full JSON summary instead of a compact result.",
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
        print("SolutionComponent C7 report/export smoke: OK")
        print(f"components={summary['component_count']}")
        print(f"valid={summary['valid_component_count']}")
        print(f"excluded={summary['excluded_component_count']}")
        print(f"component_csv_rows={summary['component_csv_row_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
