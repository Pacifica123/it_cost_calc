"""Контрольный пример ПУАЗ для интерактивного дашборда DecisionReport.

Сценарий загружает data/fixtures/puaz_dashboard_control_dataset.json, проходит
настоящую расчётную цепочку приложения (GA -> AHP -> Pareto -> Hybrid -> TCO ->
DecisionReport) в областях ПО и ТО теми же вызовами, что и экраны GUI, и строит из отчёта модель данных дашборда.
Затем значения, которые видит пользователь дашборда, сверяются с эталоном
data/fixtures/regression/puaz_dashboard_control_expected.json.

    python -B scripts/run_puaz_dashboard_control.py --check-only
    python -B scripts/run_puaz_dashboard_control.py --output-dir data/generated/puaz_control

Режим --check-only ничего не пишет на диск. С --output-dir сохраняются
decision_report.json, decision_dashboard.html и сводка контрольного примера.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEFAULT_FIXTURE = ROOT / "data" / "fixtures" / "puaz_dashboard_control_dataset.json"
DEFAULT_EXPECTED = ROOT / "data" / "fixtures" / "regression" / "puaz_dashboard_control_expected.json"
SCOPES = ("technical", "software")
MONEY_DIGITS = 2
SUPPORT_DIGITS = 2


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _ga_input(reproducibility: Mapping[str, Any], scope: str) -> Any:
    """Те же поля, что пользователь вводит на экране GA (GaInput)."""

    from ui_qt.presenters.ga_presenter import GaInput

    form = reproducibility["ga_form_by_scope"][scope]
    return GaInput(
        population=int(reproducibility.get("ga_pop_size", 24)),
        generations=int(reproducibility.get("ga_generations", 30)),
        mutation_rate=float(reproducibility.get("ga_mutation_rate", 0.04)),
        seed=int(reproducibility.get("ga_seed", 42)),
        budget=float(form["budget"]),
        min_units=float(form["min_units"]),
        max_units=float(form["max_units"]),
    )


def build_control_report(fixture_path: Path = DEFAULT_FIXTURE) -> dict[str, Any]:
    """Run the application chain on the control fixture and return DecisionReport."""

    from ui_qt.presenters.app_presenter import QtAppPresenter
    from ui_qt.presenters.ga_presenter import GaPresenter

    fixture = _read_json(fixture_path)
    reproducibility = fixture.get("reproducibility", {})
    lambda_value = float(reproducibility.get("hybrid_lambda", 0.5))

    with tempfile.TemporaryDirectory() as temp_dir:
        presenter = QtAppPresenter(
            repo_root=ROOT,
            runtime_entities_path=Path(temp_dir) / "runtime_entities.json",
        )
        presenter.load_demo_dataset(fixture_path)
        for scope in SCOPES:
            # Тот же путь, что кнопка «Запустить GA» на экране ПО/ТО.
            GaPresenter(presenter, scope=scope).run(_ga_input(reproducibility, scope))
            presenter.run_ahp_for_candidate_pool(scope)
            presenter.run_pareto_for_candidate_pool(scope)
            presenter.run_hybrid_assessment(scope, lambda_value=lambda_value)
        # Тот же снимок, из которого экран «Экспорт» строит JSON/MD/CSV/HTML.
        return presenter._build_decision_report_snapshot()


def summarize_dashboard(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Reduce the dashboard payload to values a user reads from the screen."""

    candidates: dict[str, Any] = {}
    for row in payload.get("candidates", []):
        ranks = {
            str(item.get("method")): f"{item.get('rank')}/{item.get('candidate_count')}"
            for item in row.get("method_evidence", {}).values()
        }
        candidates[str(row["key"])] = {
            "name": row.get("name"),
            "scope": row.get("scope"),
            "recommended": bool(row.get("recommended")),
            "capex": _round(row.get("capex"), MONEY_DIGITS),
            "annual_opex": _round(row.get("annual_opex"), MONEY_DIGITS),
            "tco": _round(row.get("tco"), MONEY_DIGITS),
            "analysis_support": _round(row.get("analysis_support"), SUPPORT_DIGITS),
            "ranks": dict(sorted(ranks.items())),
        }

    kpis: dict[str, Any] = {}
    for scope in ("all", *SCOPES):
        rows = [
            row
            for row in payload.get("candidates", [])
            if scope == "all" or row.get("scope") == scope
        ]
        tcos = [row["tco"] for row in rows if row.get("tco") is not None]
        support = [row["analysis_support"] for row in rows if row.get("analysis_support") is not None]
        kpis[scope] = {
            "alternatives": len(rows),
            "tco_min": _round(min(tcos), MONEY_DIGITS) if tcos else None,
            "tco_max": _round(max(tcos), MONEY_DIGITS) if tcos else None,
            "average_support": _round(sum(support) / len(support), SUPPORT_DIGITS) if support else None,
        }

    recommended = payload.get("recommended", {})
    return {
        "recommended": {
            "id": recommended.get("id"),
            "name": recommended.get("name"),
            "method": recommended.get("method"),
        },
        "kpis": kpis,
        "catalog_quality": dict(payload.get("catalog_quality", {})),
        "candidates": dict(sorted(candidates.items())),
    }


def compare(actual: Mapping[str, Any], expected: Mapping[str, Any], path: str = "") -> list[str]:
    """Return human-readable differences between two summaries."""

    errors: list[str] = []
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        for key in expected:
            if key not in actual:
                errors.append(f"{path}{key}: отсутствует в результате")
                continue
            errors.extend(compare(actual[key], expected[key], f"{path}{key}."))
        for key in actual:
            if key not in expected:
                errors.append(f"{path}{key}: лишнее значение в результате")
        return errors
    if actual != expected:
        errors.append(f"{path.rstrip('.')}: {actual!r} != ожидалось {expected!r}")
    return errors


def _round(value: Any, digits: int) -> float | None:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fixture", default=str(DEFAULT_FIXTURE), help="Контрольный набор данных.")
    parser.add_argument("--expected", default=str(DEFAULT_EXPECTED), help="Эталон значений дашборда.")
    parser.add_argument("--check-only", action="store_true", help="Только сверка, без записи файлов.")
    parser.add_argument("--output-dir", default=None, help="Куда сохранить отчёт, HTML и сводку.")
    parser.add_argument(
        "--write-expected",
        action="store_true",
        help="Перезаписать эталон текущим результатом (только при осознанном изменении примера).",
    )
    args = parser.parse_args()

    from infrastructure.exporters.decision_report_exporter import export_decision_report_json
    from infrastructure.exporters.interactive_dashboard_exporter import (
        build_interactive_dashboard_payload,
        export_interactive_dashboard,
    )

    report = build_control_report(Path(args.fixture))
    summary = summarize_dashboard(build_interactive_dashboard_payload(report))

    expected_path = Path(args.expected)
    if args.write_expected and not args.check_only:
        expected_path.parent.mkdir(parents=True, exist_ok=True)
        expected_path.write_text(
            json.dumps(
                {"schema_version": 1, "scenario_id": "puaz_dashboard_control_v1", "expected": summary},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    errors = compare(summary, _read_json(expected_path).get("expected", {}))

    result: dict[str, Any] = {
        "scenario_id": "puaz_dashboard_control_v1",
        "status": "ok" if not errors else "failed",
        "recommended": summary["recommended"],
        "kpis": summary["kpis"],
        "catalog_quality": summary["catalog_quality"],
        "validation_errors": errors,
    }

    if args.output_dir and not args.check_only:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result["outputs"] = {
            "decision_report": str(export_decision_report_json(report, output_dir / "decision_report.json")),
            "dashboard": str(export_interactive_dashboard(report, output_dir / "decision_dashboard.html")),
        }
        summary_path = output_dir / "puaz_dashboard_control_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        result["outputs"]["summary"] = str(summary_path)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
