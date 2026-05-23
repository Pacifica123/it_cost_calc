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


def _validate_expected_invariants(payload: dict, expected_payload: dict) -> list[str]:
    expected = expected_payload.get("expected", expected_payload)
    invariants = payload.get("invariants", {})
    reproducibility = payload.get("reproducibility", {})
    errors: list[str] = []

    scope_counts = invariants.get("scope_counts", {})
    for scope in expected.get("required_scopes", []):
        if int(scope_counts.get(scope, 0)) <= 0:
            errors.append(f"required scope is absent in demo fixture: {scope}")

    component_counts = invariants.get("component_type_counts", {})
    for component_type in expected.get("required_component_types", []):
        if int(component_counts.get(component_type, 0)) <= 0:
            errors.append(f"required component_type is absent in demo fixture: {component_type}")

    candidates_by_scope = invariants.get("candidate_count_by_scope", {})
    for scope, minimum in expected.get("candidate_count_by_scope_min", {}).items():
        if int(candidates_by_scope.get(scope, 0)) < int(minimum):
            errors.append(
                f"candidate count for {scope} is below expected minimum: "
                f"{candidates_by_scope.get(scope, 0)} < {minimum}"
            )

    for key in (
        "combined_candidate_count",
        "tco_candidate_count",
        "decision_report_candidate_count",
    ):
        minimum = expected.get(f"{key}_min")
        if minimum is not None and int(invariants.get(key, 0)) < int(minimum):
            errors.append(f"{key} is below expected minimum: {invariants.get(key, 0)} < {minimum}")

    expected_seed = expected.get("ga_seed")
    actual_seed = reproducibility.get("ga", {}).get("seed")
    if expected_seed is not None and actual_seed != expected_seed:
        errors.append(f"GA seed mismatch: {actual_seed} != {expected_seed}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build and validate the stage-8 demo control scenario without pytest."
    )
    parser.add_argument(
        "--fixture",
        default=str(ROOT / "data" / "fixtures" / "demo_dataset.json"),
        help="Path to demonstration dataset JSON.",
    )
    parser.add_argument(
        "--invariants",
        default=str(ROOT / "data" / "fixtures" / "regression" / "demo_control_invariants.json"),
        help="Path to regression invariants JSON.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write the built scenario snapshot. Omitted by checks to avoid generated files.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate and print a short summary without writing a snapshot.",
    )
    args = parser.parse_args()

    from application.services.demo_control_scenario_service import DemoControlScenarioService

    fixture_path = Path(args.fixture)
    fixture_payload = _read_json(fixture_path)
    entities = fixture_payload.get("entities", fixture_payload)
    if not isinstance(entities, dict):
        raise ValueError("Fixture must contain an 'entities' object or be an entities mapping")

    service = DemoControlScenarioService()
    scenario = service.build(entities)
    validation = scenario.get("validation", {})
    errors = list(validation.get("errors", []))

    invariants_path = Path(args.invariants)
    if invariants_path.exists():
        expected_payload = _read_json(invariants_path)
        errors.extend(_validate_expected_invariants(scenario, expected_payload))

    summary = {
        "scenario_id": scenario.get("scenario_id"),
        "status": "ok" if not errors else "failed",
        "invariants": scenario.get("invariants", {}),
        "validation_errors": errors,
    }

    if args.output and not args.check_only:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(scenario, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summary["output"] = str(output_path)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
