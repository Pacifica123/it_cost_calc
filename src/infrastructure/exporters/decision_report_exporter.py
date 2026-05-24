"""Export adapters for the unified DecisionReport."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from domain import to_plain_data

DECISION_REPORT_SCHEMA_VERSION = 1


def build_decision_report_json_payload(report: Mapping[str, Any]) -> dict[str, Any]:
    """Return the full machine-readable report payload."""
    metadata = _mapping(report.get("metadata"))
    return to_plain_data(
        {
            "schema_version": metadata.get("schema_version", DECISION_REPORT_SCHEMA_VERSION),
            "report_type": metadata.get("report_type", "it_solution_decision_report"),
            "report": dict(report),
        }
    )


def build_decision_report_markdown(report: Mapping[str, Any]) -> str:
    """Build a compact user-facing markdown report for diploma/demo use."""
    project = _mapping(report.get("project"))
    cost_model = _mapping(report.get("cost_model"))
    tco = _mapping(cost_model.get("tco"))
    winner = _mapping(report.get("winner_explanation"))
    recommended = _mapping(winner.get("recommended"))
    npv = _mapping(report.get("npv_interpretation"))
    candidates = _sequence(report.get("candidate_configurations"))
    components = _sequence(report.get("components"))
    editor_components = [
        component
        for component in components
        if isinstance(component, Mapping)
        and component.get("source_format") == "solution_component"
    ]
    warnings = [str(item) for item in _sequence(report.get("warnings"))]
    risks = _sequence(report.get("risks"))

    lines = [
        f"# {report.get('title') or project.get('title') or 'Итоговый отчёт выбора ИТ-решения'}",
        "",
        "## 1. Исходный проект выбора",
        f"- Цель: {project.get('goal', 'не указана')}",
        f"- Дата формирования: {project.get('generated_at', 'не указана')}",
        "",
        "## 2. Итоговая рекомендация",
        f"- Победитель: {recommended.get('name') or recommended.get('id') or 'не определён'}",
        f"- Согласованность методов: {winner.get('agreement', 'не определена')}",
        f"- Пояснение: {winner.get('explanation', 'недостаточно данных')}",
    ]
    for reason in _sequence(winner.get("reasons")):
        lines.append(f"- Причина: {reason}")

    lines.extend(
        [
            "",
            "## 3. Стоимость владения",
            f"- Начальные инвестиции: {_money(tco.get('initial_investment'))}",
            f"- Ежемесячные расходы: {_money(tco.get('monthly_opex'))}",
            f"- Годовые расходы: {_money(tco.get('annual_opex'))}",
            f"- TCO за период: {_money(tco.get('total_ownership_cost'))}",
            "",
            "## 4. Компоненты редактора",
        ]
    )
    if editor_components:
        lines.extend(_solution_component_table(editor_components[:10]))
        if len(editor_components) > 10:
            lines.append(f"\nПоказаны первые 10 компонентов редактора из {len(editor_components)}.")
    else:
        lines.append("Компоненты редактора SolutionComponent не переданы в отчёт.")

    lines.extend(["", "## 5. Пул альтернатив"])
    if candidates:
        lines.extend(_candidate_table(candidates[:10]))
        if len(candidates) > 10:
            lines.append(f"\nПоказаны первые 10 альтернатив из {len(candidates)}.")
    else:
        lines.append("Альтернативы не переданы в отчёт.")

    lines.extend(
        [
            "",
            "## 6. NPV-интерпретация",
            f"- Статус: {npv.get('status', 'не рассчитан')}",
            f"- Значение NPV: {_money(npv.get('npv'))}",
            f"- Пояснение: {npv.get('interpretation') or npv.get('explanation', 'не указано')}",
            "",
            "## 7. Риски и предупреждения",
        ]
    )
    if risks:
        for risk in risks[:10]:
            if isinstance(risk, Mapping):
                lines.append(
                    f"- {risk.get('description', risk.get('id', 'риск'))}: "
                    f"{risk.get('mitigation', 'требует уточнения')}"
                )
    elif warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("Критичных предупреждений не зафиксировано.")

    lines.extend(
        [
            "",
            "## 8. Сценарий защиты",
            "1. Пользователь вводит компоненты решения и связанные затраты.",
            "2. Компоненты нормализуются по области анализа и типу.",
            "3. Из компонентов формируется пул кандидатных конфигураций.",
            "4. AHP, GA, GA + AHP и финансовая модель анализируют один связанный набор данных.",
            "5. DecisionReport фиксирует победителя, ограничения, риски и воспроизводимые исходные данные.",
            "",
        ]
    )
    return "\n".join(lines)


def build_decision_report_csv_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return a compact tabular candidate summary for spreadsheet viewing."""
    rows: list[dict[str, Any]] = []
    for candidate in _sequence(report.get("candidate_configurations")):
        if not isinstance(candidate, Mapping):
            continue
        totals = _mapping(candidate.get("totals"))
        tco = _mapping(totals.get("tco"))
        metrics = _mapping(candidate.get("metrics"))
        rows.append(
            {
                "id": candidate.get("id"),
                "name": candidate.get("name"),
                "scope": candidate.get("scope"),
                "source": candidate.get("source"),
                "component_count": len(_sequence(candidate.get("components"))),
                "capital_cost": totals.get("capital_cost", totals.get("total_cost")),
                "monthly_opex": tco.get("monthly_opex", totals.get("monthly_opex")),
                "total_ownership_cost": tco.get(
                    "total_ownership_cost", totals.get("total_ownership_cost")
                ),
                "ga_score": metrics.get("ga_score"),
                "people": metrics.get("people"),
            }
        )
    return rows


def export_decision_report_json(report: Mapping[str, Any], filename: str | Path) -> Path:
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(build_decision_report_json_payload(report), file, ensure_ascii=False, indent=2)
        file.write("\n")
    return path


def export_decision_report_markdown(report: Mapping[str, Any], filename: str | Path) -> Path:
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_decision_report_markdown(report), encoding="utf-8")
    return path


def export_decision_report_csv(report: Mapping[str, Any], filename: str | Path) -> Path:
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = build_decision_report_csv_rows(report)
    fieldnames = [
        "id",
        "name",
        "scope",
        "source",
        "component_count",
        "capital_cost",
        "monthly_opex",
        "total_ownership_cost",
        "ga_score",
        "people",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _solution_component_table(components: Sequence[Any]) -> list[str]:
    lines = [
        "| ID | Компонент | Scope | Статус | Предупреждения |",
        "|---|---|---|---|---|",
    ]
    for component in components:
        if not isinstance(component, Mapping):
            continue
        warnings = [
            str(item)
            for item in _sequence(component.get("blocking_errors"))
            + _sequence(component.get("validation_warnings"))
            if item
        ]
        warning_text = "; ".join(warnings[:2]) or "—"
        lines.append(
            "| {id} | {name} | {scope} | {status} | {warnings} |".format(
                id=str(component.get("id", "")).replace("|", "\\|"),
                name=str(component.get("name", "")).replace("|", "\\|"),
                scope=str(component.get("scope") or "").replace("|", "\\|"),
                status=str(
                    component.get("normalization_state")
                    or component.get("editor_status")
                    or ""
                ).replace("|", "\\|"),
                warnings=warning_text.replace("|", "\\|"),
            )
        )
    return lines


def _candidate_table(candidates: Sequence[Any]) -> list[str]:
    lines = [
        "| ID | Альтернатива | Источник | TCO |",
        "|---|---|---|---:|",
    ]
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        totals = _mapping(candidate.get("totals"))
        tco = _mapping(totals.get("tco"))
        lines.append(
            "| {id} | {name} | {source} | {tco} |".format(
                id=candidate.get("id", ""),
                name=str(candidate.get("name", "")).replace("|", "\\|"),
                source=candidate.get("source", ""),
                tco=_money(tco.get("total_ownership_cost", totals.get("total_ownership_cost"))),
            )
        )
    return lines


def _money(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "не указано"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


__all__ = [
    "DECISION_REPORT_SCHEMA_VERSION",
    "build_decision_report_csv_rows",
    "build_decision_report_json_payload",
    "build_decision_report_markdown",
    "export_decision_report_csv",
    "export_decision_report_json",
    "export_decision_report_markdown",
]
