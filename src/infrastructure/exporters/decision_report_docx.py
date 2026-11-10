"""Word (.docx) version of the decision report.

The body is the same Markdown report the Export screen already writes (all sections,
tables and the defence scenario), followed by optional sections that live outside the
DecisionReport snapshot: profitability and risk of the NPV, and the multi-year plans.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from infrastructure.exporters.decision_report_exporter import build_decision_report_markdown
from infrastructure.exporters.docx_writer import DocxDocument


def _rows(items: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> list[list[str]]:
    return [[str(item.get(key, "")) for key in keys] for item in items]


def build_decision_report_docx(
    report: Mapping[str, Any],
    *,
    risk_section: Mapping[str, Any] | None = None,
    plans_section: Mapping[str, Any] | None = None,
    extra_sections: Sequence[tuple[str, Mapping[str, Any]]] = (),
) -> DocxDocument:
    title = str(report.get("title") or "Итоговый отчёт выбора ИТ-решения")
    document = DocxDocument(title)
    document.add_markdown(build_decision_report_markdown(report))

    if risk_section:
        document.heading("Доходность и риск проекта", 1)
        document.table(
            ["Показатель", "Значение", "Вывод"],
            _rows(risk_section.get("metrics", []), ("name", "value", "note")),
            caption="Таблица – Показатели доходности",
        )
        if risk_section.get("verdict"):
            document.paragraph(str(risk_section["verdict"]))
        if risk_section.get("tornado"):
            document.table(
                ["Фактор", "Пессимистично", "Оптимистично", "Размах", "Точка безубыточности"],
                _rows(risk_section["tornado"], ("factor", "low", "high", "range", "break_even")),
                caption="Таблица – Анализ чувствительности NPV (торнадо)",
            )
        if risk_section.get("monte_carlo"):
            document.table(
                ["Показатель", "Значение"],
                _rows(risk_section["monte_carlo"], ("name", "value")),
                caption="Таблица – Результаты моделирования Монте-Карло",
            )

    if plans_section:
        document.heading("Планирование затрат по годам", 1)
        if plans_section.get("tco_summary"):
            document.paragraph(str(plans_section["tco_summary"]))
        if plans_section.get("tco_rows"):
            document.table(
                ["Год", "OPEX", "Энергия", "Итого за год", "Приведённо", "Накопленно"],
                _rows(plans_section["tco_rows"], ("year", "opex", "electricity", "total", "discounted", "cumulative")),
                caption="Таблица – Совокупная стоимость владения по годам",
            )
        if plans_section.get("amortization_summary"):
            document.paragraph(str(plans_section["amortization_summary"]))
        if plans_section.get("amortization_rows"):
            document.table(
                ["Позиция", "Вид", "Кол-во", "Стоимость", "Срок, мес", "В месяц"],
                _rows(plans_section["amortization_rows"], ("name", "kind", "quantity", "total", "months", "monthly")),
                caption="Таблица – Амортизация",
            )
        if plans_section.get("replacement_summary"):
            document.paragraph(str(plans_section["replacement_summary"]))
        if plans_section.get("replacement_rows"):
            document.table(
                ["Год", "Что покупать", "Сумма"],
                _rows(plans_section["replacement_rows"], ("year", "items", "total")),
                caption="Таблица – План замены оборудования",
            )

    for heading, section in extra_sections:
        document.heading(heading, 1)
        for paragraph in section.get("paragraphs", []):
            document.paragraph(str(paragraph))
        for table in section.get("tables", []):
            document.table(table.get("header", []), table.get("rows", []), caption=table.get("caption"))
    return document


def export_decision_report_docx(
    report: Mapping[str, Any],
    filename: str | Path,
    *,
    risk_section: Mapping[str, Any] | None = None,
    plans_section: Mapping[str, Any] | None = None,
    extra_sections: Sequence[tuple[str, Mapping[str, Any]]] = (),
) -> Path:
    document = build_decision_report_docx(
        report,
        risk_section=risk_section,
        plans_section=plans_section,
        extra_sections=extra_sections,
    )
    return document.save(filename)
