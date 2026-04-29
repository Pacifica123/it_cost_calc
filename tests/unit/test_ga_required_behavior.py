"""Обязательные проверки поведения универсального ядра ГА.

Тесты фиксируют именно свойства генетического алгоритма, которые важны для
дальнейшей дипломной демонстрации: успешный запуск, воспроизводимость,
невозможные ограничения, нормализацию, направления критериев и соблюдение
бюджета.
"""

from __future__ import annotations

import math
from typing import Any

from domain.optimization.ga import run_ga_mvp
from domain.optimization.mech import Item


def _sum_prop(prop_name: str):
    def _inner(subset: list[Item]) -> float:
        return sum(float(item.properties.get(prop_name, 0.0)) for item in subset)

    return _inner


def _count_selected(subset: list[Item]) -> int:
    return len(subset)


def _demo_items() -> list[Item]:
    return [
        Item(name="FastWasteful", performance=100, cost=90, energy=95),
        Item(name="BalancedWorkstation", performance=90, cost=40, energy=20),
        Item(name="CheapSlow", performance=20, cost=10, energy=5),
    ]


def _ga_params(**overrides: Any) -> dict[str, Any]:
    params: dict[str, Any] = {
        "items": _demo_items(),
        "criteria": [
            {
                "name": "performance",
                "func": _sum_prop("performance"),
                "direction": "max",
            },
            {"name": "cost", "func": _sum_prop("cost"), "direction": "min"},
            {"name": "energy", "func": _sum_prop("energy"), "direction": "min"},
        ],
        "constraints": [
            {
                "name": "budget",
                "func": _sum_prop("cost"),
                "operator": "<=",
                "bound": 100,
            },
            {
                "name": "min_selected",
                "func": _count_selected,
                "operator": ">=",
                "bound": 1,
            },
            {
                "name": "single_item_case",
                "func": _count_selected,
                "operator": "<=",
                "bound": 1,
            },
        ],
        "weights": [0.5, 0.25, 0.25],
        "pop_size": 12,
        "generations": 10,
        "mutation_rate": 0.0,
        "elite": 1,
        "seed": 42,
        "stagnation_limit": 20,
        "max_random_attempts": 200,
        "top_solutions_limit": 5,
    }
    params.update(overrides)
    return params


def test_ga_successfully_runs_and_returns_explainable_result():
    result = run_ga_mvp(_ga_params())

    assert result["error"] is None
    assert result["best_items"]
    assert result["best_mask"]
    assert result["generations_ran"] > 0
    assert result["history"]
    assert result["termination_reason"] in {"max_generations", "stagnation"}
    assert result["fitness_scale"] == "normalized_0_1"
    assert result["top_solutions"]


def test_ga_is_reproducible_with_same_seed():
    params = _ga_params(seed=777, mutation_rate=0.15, generations=15, pop_size=16)

    first = run_ga_mvp(params)
    second = run_ga_mvp(params)

    assert first["error"] is None
    assert second["error"] is None
    assert first["best_mask"] == second["best_mask"]
    assert math.isclose(first["best_agg"], second["best_agg"], rel_tol=0.0, abs_tol=1e-12)
    assert [row["generation_best_score"] for row in first["history"]] == [
        row["generation_best_score"] for row in second["history"]
    ]
    assert [solution["mask"] for solution in first["top_solutions"]] == [
        solution["mask"] for solution in second["top_solutions"]
    ]


def test_ga_reports_impossible_constraints_without_selecting_items():
    params = _ga_params(
        constraints=[
            {
                "name": "impossible_budget",
                "func": _sum_prop("cost"),
                "operator": "<=",
                "bound": 1,
            },
            {
                "name": "min_selected",
                "func": _count_selected,
                "operator": ">=",
                "bound": 1,
            },
        ],
        max_random_attempts=10,
    )

    result = run_ga_mvp(params)

    assert result["termination_reason"] == "no_feasible_solution"
    assert result["generations_ran"] == 0
    assert result["best_items"] == []
    assert result["best_mask"] == tuple()
    assert result["top_solutions"] == []
    assert result["error"]


def test_ga_uses_normalized_scores_for_fitness():
    result = run_ga_mvp(_ga_params())

    assert result["fitness_scale"] == "normalized_0_1"
    assert len(result["normalization_mins"]) == 3
    assert len(result["normalization_maxs"]) == 3
    assert all(0.0 <= value <= 1.0 for value in result["best_normalized_scores"])
    assert all(
        0.0 <= value <= 1.0
        for value in result["best_normalized_scores_by_criterion"].values()
    )
    assert 0.0 <= result["best_agg"] <= 1.0

    raw_weighted_sum = sum(
        result["best_directed_scores_by_criterion"][name] * weight
        for name, weight in zip(result["criterion_names"], result["weights"])
    )
    assert not math.isclose(result["best_agg"], raw_weighted_sum, rel_tol=0.0, abs_tol=1e-12)


def test_ga_respects_max_and_min_criterion_directions():
    result = run_ga_mvp(_ga_params())

    selected_names = {item["name"] for item in result["best_items"]}
    assert selected_names == {"BalancedWorkstation"}

    criteria_by_name = {criterion["name"]: criterion for criterion in result["criteria"]}
    assert criteria_by_name["performance"]["direction"] == "max"
    assert criteria_by_name["cost"]["direction"] == "min"
    assert criteria_by_name["energy"]["direction"] == "min"

    # Проверяем, что результат не свёлся ни к самому производительному, но дорогому
    # варианту, ни к самому дешёвому, но слабому варианту.
    assert result["best_raw_scores_by_criterion"]["performance"] == 90.0
    assert result["best_raw_scores_by_criterion"]["cost"] == 40.0
    assert result["best_raw_scores_by_criterion"]["energy"] == 20.0


def test_ga_result_never_violates_budget_constraint():
    result = run_ga_mvp(_ga_params())

    total_cost = sum(float(item["cost"]) for item in result["best_items"])
    assert total_cost <= 100

    constraints_by_name = {constraint["name"]: constraint for constraint in result["constraints"]}
    assert constraints_by_name["budget"]["passed"] is True
    assert constraints_by_name["budget"]["value"] == total_cost
