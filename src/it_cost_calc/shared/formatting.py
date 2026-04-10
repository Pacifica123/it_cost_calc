from __future__ import annotations


def format_money(value: float) -> str:
    return f"{value:,.2f}".replace(",", " ")


def format_cost_summary(totals: dict) -> str:
    lines = [
        f"Капитальные затраты: {format_money(totals['total_capital'])}",
        f"Операционные затраты (разовые): {format_money(totals['total_operational_one_time'])}",
        f"Операционные затраты (периодические): {format_money(totals['total_operational_monthly'])}",
        f"Стоимость электроэнергии: {format_money(totals['electricity_costs'])}",
    ]
    overall = (
        totals["total_capital"]
        + totals["total_operational_one_time"]
        + totals["total_operational_monthly"]
        + totals["electricity_costs"]
    )
    lines.append(f"Общая стоимость: {format_money(overall)}")
    return "\n".join(lines)
