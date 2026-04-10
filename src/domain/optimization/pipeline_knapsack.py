"""Демонстрационный пайплайн: mech -> GA -> сравнительный отчёт."""

from __future__ import annotations

import logging

import numpy as np

from .ga import run_ga_mvp
from .mech import Item, run as mech_run

logger = logging.getLogger(__name__)

server_hw = [
    Item(name="Server_A", weight=8, cost=4000, energy=500, perf=700),
    Item(name="Server_B", weight=6, cost=3500, energy=420, perf=600),
    Item(name="Server_C", weight=10, cost=5000, energy=550, perf=800),
]

network_hw = [
    Item(name="Switch_A", weight=1, cost=500, energy=50, perf=100),
    Item(name="Router_A", weight=1, cost=700, energy=80, perf=120),
    Item(name="Firewall_A", weight=2, cost=1200, energy=100, perf=200),
]

client_hw = [
    Item(name="Client_A", weight=2, cost=800, energy=100, perf=150),
    Item(name="Client_B", weight=3, cost=1000, energy=130, perf=180),
    Item(name="Client_C", weight=4, cost=1500, energy=160, perf=250),
]

items = server_hw + network_hw + client_hw
MAX_BUDGET = 10000


def budget_constraint(subset):
    return sum(item.properties.get("cost", 0) for item in subset)


def performance(subset):
    return sum(item.properties.get("perf", 0) for item in subset)


def total_energy(subset):
    return sum(item.properties.get("energy", 0) for item in subset)


def total_cost(subset):
    return sum(item.properties.get("cost", 0) for item in subset)


def run_demo() -> dict:
    pairwise = np.array([[1, 5, 3], [1 / 5, 1, 1 / 2], [1 / 3, 2, 1]])
    criteria_funcs = [performance, total_energy, total_cost]
    optimize_directions = [1, -1, -1]

    mech_params = {
        "items": items,
        "max_constraint": MAX_BUDGET,
        "constraint_func": budget_constraint,
        "criteria_funcs": criteria_funcs,
        "optimize_directions": optimize_directions,
        "pairwise_matrix": pairwise,
        "exclude_empty": True,
    }

    logger.info("Запуск демонстрационного пайплайна knapsack")
    mech_out = mech_run(mech_params)
    start_pop = mech_out.get("pareto", [])
    logger.info("Жадный алгоритм подготовил %s стартовых конфигураций", len(start_pop))

    ga_params = {
        "items": items,
        "max_constraint": MAX_BUDGET,
        "constraint_func": budget_constraint,
        "criteria_funcs": criteria_funcs,
        "optimize_directions": optimize_directions,
        "pairwise_matrix": pairwise,
        "pop_size": 50,
        "generations": 100,
        "mutation_rate": 0.05,
        "elite": 1,
        "seed": 42,
        "stagnation_limit": 40,
    }
    ga_result = run_ga_mvp(ga_params)
    logger.info("Демонстрационный пайплайн завершён. best_agg=%s", ga_result.get("best_agg"))
    return {"mech": mech_out, "ga": ga_result}


if __name__ == "__main__":
    from infrastructure.logging import configure_logging

    configure_logging(level="INFO")
    result = run_demo()
    logger.info("Итог демонстрационного пайплайна: %s", result["ga"])
