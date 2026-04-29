import numpy as np

from domain.optimization.ga import run_ga_mvp
from domain.optimization.mech import Item


def _count(subset):
    return len(subset)


def _performance(subset):
    return sum(item.properties.get("performance", 0) for item in subset)


def _cost(subset):
    return sum(item.properties.get("cost", 0) for item in subset)


def test_ga_fitness_uses_normalized_scores_instead_of_raw_scales():
    """Высокая стоимость не должна подавлять производительность только из-за масштаба."""
    items = [
        Item(name="CheapSlow", performance=10, cost=10),
        Item(name="FastExpensive", performance=100, cost=1000),
    ]

    result = run_ga_mvp(
        {
            "items": items,
            "max_constraint": 1,
            "constraint_func": _count,
            "criteria_funcs": [_performance, _cost],
            "optimize_directions": [1, -1],
            "pairwise_matrix": np.array([[1, 5], [1 / 5, 1]], dtype=float),
            "pop_size": 6,
            "generations": 5,
            "mutation_rate": 0,
            "elite": 1,
            "seed": 7,
            "stagnation_limit": 10,
        }
    )

    assert result["error"] is None
    assert result["best_items"][0]["name"] == "FastExpensive"
    assert result["fitness_scale"] == "normalized_0_1"
    assert result["best_normalized_scores"] == [1.0, 0.0]
    assert 0.0 <= result["best_agg"] <= 1.0

    raw_weighted_sum = float(np.dot(result["best_raw_scores"], result["weights"]))
    assert raw_weighted_sum < 0
    assert result["best_agg"] != raw_weighted_sum


def test_ga_result_exposes_normalization_bounds_and_history_scores():
    result = run_ga_mvp(
        {
            "items": [
                Item(name="A", performance=5, cost=50),
                Item(name="B", performance=10, cost=100),
            ],
            "max_constraint": 1,
            "constraint_func": _count,
            "criteria_funcs": [_performance, _cost],
            "optimize_directions": [1, -1],
            "pairwise_matrix": np.array([[1, 1], [1, 1]], dtype=float),
            "pop_size": 4,
            "generations": 3,
            "mutation_rate": 0,
            "elite": 1,
            "seed": 1,
            "stagnation_limit": 10,
        }
    )

    assert result["normalization_mins"] == [5.0, -100.0]
    assert result["normalization_maxs"] == [10.0, -50.0]
    assert all(0.0 <= value <= 1.0 for value in result["best_normalized_scores"])
    assert "generation_best_normalized_scores" in result["history"][0]
    assert all(
        0.0 <= value <= 1.0
        for value in result["history"][0]["generation_best_normalized_scores"]
    )
