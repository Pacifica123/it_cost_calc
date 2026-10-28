"""Регрессия контрольного примера ПУАЗ: значения дашборда совпадают с эталоном.

Тест проходит настоящую цепочку GA -> AHP -> Pareto -> Hybrid -> DecisionReport
на data/fixtures/puaz_dashboard_control_dataset.json и сравнивает то, что видит
пользователь дашборда, с data/fixtures/regression/puaz_dashboard_control_expected.json.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from infrastructure.exporters.interactive_dashboard_exporter import (
    build_interactive_dashboard_html,
    build_interactive_dashboard_payload,
)

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_puaz_dashboard_control.py"
EXPECTED = ROOT / "data" / "fixtures" / "regression" / "puaz_dashboard_control_expected.json"


def _load_script():
    spec = importlib.util.spec_from_file_location("run_puaz_dashboard_control", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_puaz_dashboard_control_matches_expected_values():
    script = _load_script()
    report = script.build_control_report()
    summary = script.summarize_dashboard(build_interactive_dashboard_payload(report))
    expected = json.loads(EXPECTED.read_text(encoding="utf-8"))["expected"]

    assert script.compare(summary, expected) == []


def test_puaz_dashboard_control_covers_every_dashboard_panel():
    script = _load_script()
    report = script.build_control_report()
    payload = build_interactive_dashboard_payload(report)
    rows = payload["candidates"]

    assert {row["scope"] for row in rows} == {"software", "technical"}
    assert all(
        {item["method"] for item in row["method_evidence"].values()} == {"GA", "AHP", "Pareto", "Hybrid"}
        for row in rows
    )
    assert any(row["annual_opex"] for row in rows if row["scope"] == "technical")
    assert any(row["annual_opex"] for row in rows if row["scope"] == "software")
    assert all(abs(row["capex"] + row["annual_opex"] - row["tco"]) < 0.01 for row in rows)
    quality = payload["catalog_quality"]
    assert quality["incomplete"] and quality["with_warnings"] and quality["with_manual_overrides"]
    assert sum(1 for row in rows if row["recommended"]) == 1

    html = build_interactive_dashboard_html(report)
    assert "http://" not in html and "https://" not in html
