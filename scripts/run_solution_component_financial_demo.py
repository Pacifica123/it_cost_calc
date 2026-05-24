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


def _build_summary(fixture_path: Path) -> dict:
    from application.services.npv_report_service import NPVReportService
    from application.services.solution_component_runtime_service import (
        SolutionComponentRuntimeService,
    )
    from application.use_cases.prepare_cost_summary import PrepareCostSummaryUseCase
    from application.services.cost_aggregation_service import CostAggregationService
    from application.services.equipment_service import EquipmentService
    from infrastructure.repositories.in_memory_entity_repository import InMemoryEntityRepository

    fixture = _read_json(fixture_path)
    entities = fixture.get("entities", fixture)
    if not isinstance(entities, dict):
        raise ValueError("Fixture must contain an 'entities' object or be an entities mapping")

    repository = InMemoryEntityRepository(entities)
    runtime_service = SolutionComponentRuntimeService(repository)
    electricity_profile = {
        "hours_per_day": 12.0,
        "working_days": 22.0,
        "cost_per_kwh": 6.5,
    }
    chain = runtime_service.build_financial_chain(
        horizon_months=36,
        horizon_years=5,
        annual_effect=0.0,
        discount_rate=0.12,
        electricity_profile=electricity_profile,
    )
    npv_report = NPVReportService().build_report_from_tco(
        chain["tco"],
        discount_rate=0.12,
        horizon_years=5,
        annual_effect=0.0,
    )
    totals = PrepareCostSummaryUseCase(
        CostAggregationService(),
        EquipmentService(repository),
        runtime_service,
    ).execute(electricity_profile=electricity_profile)
    return {
        "chain": chain,
        "npv_report": npv_report,
        "totals": totals,
    }


def _validate(summary: dict) -> list[str]:
    errors: list[str] = []
    chain = summary["chain"]
    tco = chain["tco"]
    npv_inputs = chain["npv_inputs"]
    rows = chain["component_financial_rows"]
    totals = summary["totals"]

    if chain["strict_financial_component_count"] < 3:
        errors.append("at least 3 strict SolutionComponent rows must enter financial chain")
    if tco["initial_investment"] <= 0:
        errors.append("SolutionComponent TCO must have positive initial investment")
    if tco["monthly_opex"] <= 0:
        errors.append("SolutionComponent TCO must have monthly OPEX from recurring costs or energy")
    if tco["electricity_monthly_cost"] <= 0:
        errors.append("technical SolutionComponent energy must be derived into monthly electricity cost")
    if not any(row.get("energy_source") == "derived_ready" for row in rows):
        errors.append("at least one component must expose derived_ready energy source")
    if any(row.get("scope") == "software" and row.get("energy_applicable") for row in rows):
        errors.append("software components must not require energy calculation")
    if npv_inputs["investment"] != tco["initial_investment"]:
        errors.append("NPV investment must come from TCO initial investment")
    if not npv_inputs.get("warnings"):
        errors.append("NPV inputs must warn when costs are present but annual effect is zero")
    if "solution_component_financial_chain" not in totals:
        errors.append("global cost summary must keep the SolutionComponent financial chain")
    if totals.get("total_capital", 0.0) < tco["total_capex"]:
        errors.append("global totals must include SolutionComponent CAPEX contribution")
    if summary["npv_report"].get("financial_basis", {}).get("investment") != tco["initial_investment"]:
        errors.append("NPV report must preserve the TCO-derived financial basis")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run SolutionComponent C5 financial integration smoke scenario without pytest."
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
        chain = summary["chain"]
        tco = chain["tco"]
        print("SolutionComponent C5 financial smoke: OK")
        print(f"strict_financial_components={chain['strict_financial_component_count']}")
        print(f"initial_investment={tco['initial_investment']:.2f}")
        print(f"monthly_opex={tco['monthly_opex']:.2f}")
        print(f"electricity_monthly_cost={tco['electricity_monthly_cost']:.2f}")
        print(f"npv_warning_count={len(chain['npv_inputs'].get('warnings', []))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
