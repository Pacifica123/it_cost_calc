"""Регрессионный тест демонстрационного сценария ГА.

Фиксирует ключевой эффект перехода на нормализованную fitness-функцию:
демо больше не должно выбирать одиночный дешёвый Switch_A вместо полноценной
конфигурации в рамках бюджета.
"""

from __future__ import annotations

from domain.optimization import pipeline_knapsack


def test_demo_pipeline_ga_does_not_regress_to_single_switch_solution():
    result = pipeline_knapsack.run_demo()
    ga_result = result["ga"]

    selected_items = ga_result["best_items"]
    selected_names = {item["name"] for item in selected_items}
    total_cost = sum(float(item.get("cost", 0.0)) for item in selected_items)
    total_perf = sum(float(item.get("perf", 0.0)) for item in selected_items)

    assert ga_result["error"] is None
    assert ga_result["fitness_scale"] == "normalized_0_1"
    assert total_cost <= pipeline_knapsack.MAX_BUDGET

    assert selected_names != {"Switch_A"}
    assert len(selected_items) > 1
    assert any(name.startswith("Server_") for name in selected_names)
    assert total_perf > 100
