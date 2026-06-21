"""Export adapters for the unified DecisionReport."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from domain import to_plain_data

DECISION_REPORT_SCHEMA_VERSION = 1


def build_decision_report_json_payload(report: Mapping[str, Any]) -> dict[str, Any]:
    """Return the full machine-readable report payload.

    C7 keeps the legacy ``report`` envelope intact and additionally exposes a
    dedicated machine snapshot for SolutionComponent rows.  Consumers that do
    not know about the editor can continue reading ``report`` only, while newer
    tools can use ``solution_components`` directly.
    """
    metadata = _mapping(report.get("metadata"))
    solution_component_report = _solution_component_report(report)
    catalog_data_quality = _mapping(report.get("catalog_data_quality"))
    return to_plain_data(
        {
            "schema_version": metadata.get("schema_version", DECISION_REPORT_SCHEMA_VERSION),
            "report_type": metadata.get("report_type", "it_solution_decision_report"),
            "report": dict(report),
            "solution_components": solution_component_report,
            "catalog_data_quality": dict(catalog_data_quality),
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
    analysis_results = _mapping(report.get("analysis_results"))
    candidates = _sequence(report.get("candidate_configurations"))
    components_report = _solution_component_report(report)
    catalog_quality = _mapping(report.get("catalog_data_quality"))
    catalog_components = _sequence(catalog_quality.get("components"))
    valid_components = _sequence(components_report.get("valid_components"))
    excluded_components = _sequence(components_report.get("excluded_components"))
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
            "## 4. Компоненты редактора SolutionComponent",
            str(
                components_report.get("role_explanation")
                or "Редактор компонентов передаёт в отчёт нормализованные записи и причины исключения черновиков."
            ),
            "",
            "### 4.1. Валидные компоненты, участвующие в выборе",
        ]
    )
    if valid_components:
        lines.extend(_valid_solution_component_table(valid_components[:12]))
        if len(valid_components) > 12:
            lines.append(f"\nПоказаны первые 12 валидных компонентов из {len(valid_components)}.")
    else:
        lines.append("Валидные компоненты редактора не переданы или не прошли допуск в аналитику.")

    lines.extend(
        [
            "",
            "### 4.2. Исключённые компоненты и черновики",
        ]
    )
    if excluded_components:
        lines.extend(_excluded_solution_component_table(excluded_components[:12]))
        if len(excluded_components) > 12:
            lines.append(f"\nПоказаны первые 12 исключённых компонентов из {len(excluded_components)}.")
        lines.append(
            "\nНеполные компоненты не влияли на итоговый выбор, потому что не получили "
            "candidate_eligible/analysis_ready и не были связаны с CandidateConfiguration."
        )
    else:
        lines.append("Черновики и исключённые компоненты не обнаружены.")

    lines.extend(["", "### 4.3. Качество данных каталога"])
    if catalog_components:
        lines.extend(_catalog_quality_table(catalog_components[:12]))
        summary = _mapping(catalog_quality.get("summary"))
        lines.append(
            "\nКаталожных компонентов: {total}; полные метрики: {complete}; "
            "неполные: {incomplete}; с предупреждениями: {warnings}.".format(
                total=summary.get("catalog_components_total", len(catalog_components)),
                complete=summary.get("complete_metrics", 0),
                incomplete=summary.get("incomplete_metrics", 0),
                warnings=summary.get("with_warnings", 0),
            )
        )
    else:
        lines.append("Компоненты, импортированные из каталога, в отчёт не переданы.")

    lines.extend(["", "## 5. Пул альтернатив"])
    if candidates:
        lines.extend(_candidate_table(candidates[:10]))
        if len(candidates) > 10:
            lines.append(f"\nПоказаны первые 10 альтернатив из {len(candidates)}.")
    else:
        lines.append("Альтернативы не переданы в отчёт.")

    lines.extend(["", "## 6. Аналитические методы"])
    lines.extend(_analysis_methods_summary(analysis_results))

    lines.extend(
        [
            "",
            "## 7. NPV-интерпретация",
            f"- Статус: {npv.get('status', 'не рассчитан')}",
            f"- Значение NPV: {_money(npv.get('npv'))}",
            f"- Пояснение: {npv.get('interpretation') or npv.get('explanation', 'не указано')}",
            "",
            "## 8. Риски и предупреждения",
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
            "## 9. Сценарий защиты",
            "1. Пользователь вводит компоненты решения и связанные затраты.",
            "2. Компоненты нормализуются по области анализа и типу.",
            "3. GA формирует общий пул кандидатных конфигураций или он загружается из demo-сценария.",
            "4. AHP ранжирует этот пул, Pareto проверяет компромиссы, Hybrid показывает сводную рекомендацию.",
            "5. DecisionReport фиксирует победителя, ограничения, риски, компоненты редактора и воспроизводимые исходные данные.",
            "",
        ]
    )
    return "\n".join(lines)


def build_decision_report_csv_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return a compact tabular candidate summary for spreadsheet viewing.

    This keeps the pre-C7 candidate CSV shape compatible.  Component-specific
    fields are exported by ``build_solution_component_csv_rows`` instead of
    overloading this table.
    """
    rows: list[dict[str, Any]] = []
    for candidate in _sequence(report.get("candidate_configurations")):
        if not isinstance(candidate, Mapping):
            continue
        totals = _mapping(candidate.get("totals"))
        tco = _mapping(totals.get("tco"))
        metrics = _mapping(candidate.get("metrics"))
        metadata = _mapping(candidate.get("metadata"))
        technical_metrics = {
            key: totals[key]
            for key in (
                "total_ram_gb",
                "total_cpu_cores",
                "total_storage_gb",
                "total_power_watts",
                "lan_ports",
                "lan_speed_mbps",
                "wifi_total_mbps",
                "ipv6_support_count",
            )
            if key in totals
        }
        rows.append(
            {
                "id": candidate.get("id"),
                "name": candidate.get("name"),
                "scope": candidate.get("scope"),
                "source": candidate.get("source"),
                "candidate_pool_source": metadata.get("candidate_pool_source"),
                "candidate_pool_method": metadata.get("candidate_pool_method"),
                "component_count": len(_sequence(candidate.get("components"))),
                "capital_cost": totals.get("capital_cost", totals.get("total_cost")),
                "monthly_opex": tco.get("monthly_opex", totals.get("monthly_opex")),
                "total_ownership_cost": tco.get(
                    "total_ownership_cost", totals.get("total_ownership_cost")
                ),
                "ga_score": metrics.get("ga_score"),
                "people": metrics.get("people"),
                "technical_metrics": json.dumps(
                    technical_metrics,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "metric_warnings": "; ".join(
                    str(warning) for warning in _sequence(totals.get("metric_warnings"))
                ),
            }
        )
    return rows


def build_solution_component_csv_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return a dedicated component export for C7 without changing candidate CSV."""
    component_report = _solution_component_report(report)
    rows: list[dict[str, Any]] = []
    for section, rows_source in (
        ("valid", _sequence(component_report.get("valid_components"))),
        ("excluded_or_draft", _sequence(component_report.get("excluded_components"))),
    ):
        for component in rows_source:
            if not isinstance(component, Mapping):
                continue
            financial = _mapping(component.get("financial_contribution"))
            metrics = _mapping(component.get("metrics_used"))
            links = _sequence(component.get("candidate_links"))
            reasons = _sequence(component.get("exclusion_reasons")) or _sequence(component.get("warnings"))
            rows.append(
                {
                    "section": section,
                    "id": component.get("id"),
                    "name": component.get("name"),
                    "scope": component.get("scope"),
                    "component_type": component.get("component_type"),
                    "origin": component.get("origin"),
                    "status": component.get("status"),
                    "candidate_eligible": component.get("candidate_eligible"),
                    "analysis_ready": component.get("analysis_ready"),
                    "tco_eligible": component.get("tco_eligible"),
                    "one_time_costs": financial.get("one_time_costs"),
                    "monthly_opex": financial.get("monthly_opex"),
                    "annual_opex": financial.get("annual_opex"),
                    "total_first_year_cost": financial.get("total_first_year_cost"),
                    "candidate_ids": "; ".join(
                        str(link.get("candidate_id")) for link in links if isinstance(link, Mapping)
                    ),
                    "metrics_used": json.dumps(metrics, ensure_ascii=False, sort_keys=True),
                    "warnings_or_reasons": "; ".join(str(reason) for reason in reasons),
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
        "candidate_pool_source",
        "candidate_pool_method",
        "component_count",
        "capital_cost",
        "monthly_opex",
        "total_ownership_cost",
        "ga_score",
        "people",
        "technical_metrics",
        "metric_warnings",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def export_solution_component_csv(report: Mapping[str, Any], filename: str | Path) -> Path:
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = build_solution_component_csv_rows(report)
    fieldnames = [
        "section",
        "id",
        "name",
        "scope",
        "component_type",
        "origin",
        "status",
        "candidate_eligible",
        "analysis_ready",
        "tco_eligible",
        "one_time_costs",
        "monthly_opex",
        "annual_opex",
        "total_first_year_cost",
        "candidate_ids",
        "metrics_used",
        "warnings_or_reasons",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _solution_component_report(report: Mapping[str, Any]) -> Mapping[str, Any]:
    explicit = report.get("solution_component_report")
    if isinstance(explicit, Mapping):
        return explicit
    components = [
        component
        for component in _sequence(report.get("components"))
        if isinstance(component, Mapping) and component.get("source_format") == "solution_component"
    ]
    valid = []
    excluded = []
    for component in components:
        row = _fallback_component_row(component)
        if row.get("candidate_eligible") or row.get("analysis_ready"):
            valid.append(row)
        else:
            excluded.append(row)
    return {
        "schema_version": 1,
        "role_explanation": "SolutionComponent rows are reported from the component snapshots.",
        "counts": {
            "total": len(components),
            "valid": len(valid),
            "excluded_or_draft": len(excluded),
            "linked_to_alternatives": 0,
        },
        "valid_components": valid,
        "excluded_components": excluded,
        "alternative_links": [],
        "metrics_used_by_method": {},
        "export_notes": [],
    }


def _fallback_component_row(component: Mapping[str, Any]) -> dict[str, Any]:
    warnings = _sequence(component.get("blocking_errors")) + _sequence(component.get("validation_warnings"))
    financial = component.get("financial_contribution")
    if not isinstance(financial, Mapping):
        cost_model = _mapping(component.get("cost_model"))
        purchase = _number(cost_model.get("purchase_cost"))
        implementation = _number(cost_model.get("implementation_cost"))
        migration = _number(cost_model.get("migration_cost"))
        testing = _number(cost_model.get("testing_cost"))
        monthly = (
            _number(cost_model.get("subscription_cost"))
            + _number(cost_model.get("support_cost"))
            + _number(cost_model.get("electricity_cost"))
        )
        one_time = purchase + implementation + migration + testing
        financial = {
            "one_time_costs": one_time,
            "monthly_opex": monthly,
            "annual_opex": monthly * 12.0,
            "total_first_year_cost": one_time + monthly * 12.0,
        }
    return {
        "id": component.get("id"),
        "name": component.get("name"),
        "scope": component.get("scope"),
        "component_type": component.get("component_type"),
        "origin": component.get("origin"),
        "status": component.get("normalization_state") or component.get("editor_status"),
        "editor_status": component.get("editor_status"),
        "analysis_ready": bool(component.get("analysis_ready")),
        "candidate_eligible": bool(component.get("candidate_eligible")),
        "tco_eligible": bool(component.get("tco_eligible")),
        "financial_contribution": dict(financial),
        "metrics_used": dict(_mapping(component.get("metrics"))),
        "candidate_links": [],
        "warnings": [str(item) for item in warnings],
        "exclusion_reasons": [str(item) for item in warnings]
        or ["Компонент не связан с CandidateConfiguration."],
    }


def _valid_solution_component_table(components: Sequence[Any]) -> list[str]:
    lines = [
        "| ID | Компонент | Scope / тип | Источник | Стоимость 1-го года | Альтернативы | Метрики |",
        "|---|---|---|---|---:|---|---|",
    ]
    for component in components:
        if not isinstance(component, Mapping):
            continue
        financial = _mapping(component.get("financial_contribution"))
        links = _sequence(component.get("candidate_links"))
        metrics = _mapping(component.get("metrics_used"))
        lines.append(
            "| {id} | {name} | {scope} / {ctype} | {origin} | {cost} | {links} | {metrics} |".format(
                id=_md(component.get("id")),
                name=_md(component.get("name")),
                scope=_md(component.get("scope")),
                ctype=_md(component.get("component_type")),
                origin=_md(component.get("origin")),
                cost=_money(financial.get("total_first_year_cost")),
                links=_md(_link_text(links)),
                metrics=_md(_metrics_text(metrics)),
            )
        )
    return lines


def _excluded_solution_component_table(components: Sequence[Any]) -> list[str]:
    lines = [
        "| ID | Компонент | Статус | Причина исключения / warning |",
        "|---|---|---|---|",
    ]
    for component in components:
        if not isinstance(component, Mapping):
            continue
        reasons = _sequence(component.get("exclusion_reasons")) or _sequence(component.get("warnings"))
        lines.append(
            "| {id} | {name} | {status} | {reason} |".format(
                id=_md(component.get("id")),
                name=_md(component.get("name")),
                status=_md(component.get("status")),
                reason=_md("; ".join(str(item) for item in reasons[:3]) or "—"),
            )
        )
    return lines


def _candidate_table(candidates: Sequence[Any]) -> list[str]:
    lines = [
        "| ID | Альтернатива | Источник | Пул | TCO |",
        "|---|---|---|---|---:|",
    ]
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        totals = _mapping(candidate.get("totals"))
        tco = _mapping(totals.get("tco"))
        metadata = _mapping(candidate.get("metadata"))
        lines.append(
            "| {id} | {name} | {source} | {pool} | {tco} |".format(
                id=_md(candidate.get("id")),
                name=_md(candidate.get("name")),
                source=_md(candidate.get("source")),
                pool=_md(metadata.get("candidate_pool_source") or "—"),
                tco=_money(tco.get("total_ownership_cost", totals.get("total_ownership_cost"))),
            )
        )
    return lines


def _catalog_quality_table(components: Sequence[Any]) -> list[str]:
    lines = [
        "| Компонент | Тип | Источник | Метрики | Не заполнено | Диагностика |",
        "|---|---|---|---|---|---|",
    ]
    for component in components:
        if not isinstance(component, Mapping):
            continue
        metrics = _mapping(component.get("metrics"))
        missing = _sequence(component.get("missing_metrics"))
        warnings = _sequence(component.get("metric_warnings"))
        source = component.get("source") or "—"
        parse_source = component.get("parse_source")
        if parse_source:
            source = f"{source} / {parse_source}"
        lines.append(
            "| {name} | {ctype} | {source} | {metrics} | {missing} | {warnings} |".format(
                name=_md(component.get("name")),
                ctype=_md(component.get("component_type") or "—"),
                source=_md(source),
                metrics=_md(_metrics_text(metrics)),
                missing=_md(", ".join(str(field) for field in missing) or "—"),
                warnings=_md("; ".join(str(warning) for warning in warnings[:2]) or "—"),
            )
        )
    return lines


def _analysis_methods_summary(analysis_results: Mapping[str, Any]) -> list[str]:
    if not analysis_results:
        return ["Аналитические методы ещё не переданы в отчёт."]

    rows = [
        "| Метод | Роль в новой модели | Статус |",
        "|---|---|---|",
    ]
    method_roles = (
        ("genetic_optimization", "GA", "поиск допустимых кандидатов"),
        ("ahp", "AHP", "самостоятельное экспертное ранжирование общего пула"),
        ("criteria_importance", "Pareto", "проверка компромиссов и недоминируемых вариантов"),
        ("hybrid_assessment", "Hybrid", "сводная рекомендация и контроль расхождения методов"),
    )
    for key, label, role in method_roles:
        value = analysis_results.get(key)
        rows.append(f"| {label} | {role} | {_md(_method_status(value))} |")
    if analysis_results.get("genetic_ahp"):
        rows.append(
            "| legacy GA+AHP | совместимость старых экспортов; не основной UI-сценарий | сохранено в JSON |"
        )
    return rows


def _method_status(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "не рассчитано"
    by_scope = value.get("by_scope")
    if isinstance(by_scope, Mapping):
        parts = []
        for scope, payload in by_scope.items():
            if isinstance(payload, Mapping):
                status = payload.get("status") or payload.get("method") or "есть результат"
                parts.append(f"{scope}: {status}")
        return "; ".join(parts) or "не рассчитано"
    return str(value.get("status") or value.get("method") or "есть результат")


def _link_text(links: Sequence[Any]) -> str:
    candidate_ids = [
        str(link.get("candidate_id"))
        for link in links
        if isinstance(link, Mapping) and link.get("candidate_id")
    ]
    return "; ".join(candidate_ids) or "—"


def _metrics_text(metrics: Mapping[str, Any]) -> str:
    if not metrics:
        return "—"
    parts = []
    for key, value in list(metrics.items())[:5]:
        parts.append(f"{key}={value}")
    return "; ".join(parts)


def _money(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "не указано"


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _md(value: Any) -> str:
    return str(value if value is not None else "").replace("|", "\\|")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


__all__ = [
    "DECISION_REPORT_SCHEMA_VERSION",
    "build_decision_report_csv_rows",
    "build_decision_report_json_payload",
    "build_decision_report_markdown",
    "build_solution_component_csv_rows",
    "export_decision_report_csv",
    "export_decision_report_json",
    "export_decision_report_markdown",
    "export_solution_component_csv",
]
