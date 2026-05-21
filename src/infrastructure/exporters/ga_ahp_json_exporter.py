"""JSON export adapter for the combined GA + AHP report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from domain import to_plain_data

GA_AHP_REPORT_SCHEMA_VERSION = 1


def build_ga_ahp_report_payload(result: Mapping[str, Any]) -> dict[str, Any]:
    """Build a stable JSON payload from the combined GA + AHP result.

    The service already returns a rich in-memory ``export_payload``. This helper
    makes the user-facing JSON file explicit and duplicates the most important
    sections at the top level so the report is easy to inspect without knowing
    the internal result structure.
    """
    ahp_report = _mapping(result.get("ahp_report"))
    export_payload = _mapping(result.get("export_payload"))
    independent_assessment = _mapping(
        export_payload.get("independent_assessment")
        or ahp_report.get("independent_assessment")
    )
    agreement = _mapping(
        export_payload.get("ga_ahp_agreement")
        or ahp_report.get("agreement")
    )
    winner_interpretation = _mapping(
        agreement.get("winner_interpretation")
        or _mapping(agreement.get("winner_comparison")).get("interpretation")
    )
    final = _mapping(ahp_report.get("final"))
    genetic_optimization = _mapping(export_payload.get("genetic_optimization"))
    ga_quality_check = _mapping(genetic_optimization.get("quality_check"))

    payload = {
        "schema_version": GA_AHP_REPORT_SCHEMA_VERSION,
        "report_type": "ga_ahp_independent_assessment",
        "status": result.get("status"),
        "ahp_status": ahp_report.get("status"),
        "assessment_mode": ahp_report.get("assessment_mode"),
        "summary": agreement.get("summary"),
        "ahp_report": ahp_report,
        "export_payload": export_payload,
        "ga_ahp_agreement": agreement,
        "winner_interpretation": winner_interpretation,
        "independent_assessment": independent_assessment,
        "ga_quality_check": ga_quality_check,
        "ranking": final.get("ranking", []),
        "candidate_pool": independent_assessment.get("candidate_pool", []),
        "criterion_values": independent_assessment.get("criterion_values", {}),
        "criterion_utilities": independent_assessment.get("criterion_utilities", {}),
    }
    return to_plain_data(payload)


def export_ga_ahp_report_json(result: Mapping[str, Any], filename: str | Path) -> Path:
    """Write the combined GA + AHP report to a UTF-8 JSON file."""
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_ga_ahp_report_payload(result)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return path


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


__all__ = [
    "GA_AHP_REPORT_SCHEMA_VERSION",
    "build_ga_ahp_report_payload",
    "export_ga_ahp_report_json",
]
