import numpy as np
import pytest

from domain.optimization.ga import run_ga_mvp
from domain.optimization.mech import Item


def _weight(subset):
    return sum(item.properties.get("weight", 0) for item in subset)


def _value(subset):
    return sum(item.properties.get("value", 0) for item in subset)


def _base_params(**overrides):
    params = {
        "items": [
            Item(name="A", weight=1, value=3),
            Item(name="B", weight=2, value=5),
            Item(name="C", weight=3, value=7),
        ],
        "max_constraint": 3,
        "constraint_func": _weight,
        "criteria_funcs": [_value],
        "optimize_directions": [1],
        "pairwise_matrix": np.array([[1]], dtype=float),
        "pop_size": 8,
        "generations": 5,
        "mutation_rate": 0.05,
        "elite": 1,
        "seed": 42,
        "stagnation_limit": 10,
    }
    params.update(overrides)
    return params


def test_ga_returns_history_and_termination_reason():
    result = run_ga_mvp(_base_params())

    assert result["termination_reason"] == "max_generations"
    assert result["error"] is None
    assert result["generations_ran"] == 5
    assert len(result["history"]) == 5
    assert result["history"][0]["generation"] == 1
    assert result["history"][0]["feasible_count"] > 0
    assert "mean_score" in result["history"][0]
    assert result["best_items"]


def test_ga_reports_no_feasible_solution_without_random_mask_hang():
    result = run_ga_mvp(
        _base_params(
            items=[Item(name="TooHeavy", weight=10, value=100)],
            max_constraint=1,
            pop_size=4,
            generations=10,
            max_random_attempts=5,
        )
    )

    assert result["termination_reason"] == "no_feasible_solution"
    assert result["generations_ran"] == 0
    assert result["best_mask"] == tuple()
    assert result["best_items"] == []
    assert result["history"] == []
    assert "не найдено" in result["error"].lower()


def test_ga_reports_empty_item_list_as_no_feasible_solution():
    result = run_ga_mvp(_base_params(items=[]))

    assert result["termination_reason"] == "no_feasible_solution"
    assert result["generations_ran"] == 0
    assert result["best_items"] == []
    assert "пуст" in result["error"].lower()


def test_ga_stops_by_stagnation_limit():
    result = run_ga_mvp(
        _base_params(
            items=[Item(name="Only", weight=1, value=3)],
            max_constraint=1,
            pop_size=1,
            generations=20,
            mutation_rate=0,
            elite=1,
            stagnation_limit=1,
        )
    )

    assert result["termination_reason"] == "stagnation"
    assert result["generations_ran"] == 2
    assert len(result["history"]) == 2


@pytest.mark.parametrize(
    "override, message",
    [
        ({"pop_size": 0}, "pop_size"),
        ({"generations": 0}, "generations"),
        ({"mutation_rate": 1.5}, "mutation_rate"),
        ({"elite": 3, "pop_size": 2}, "elite"),
        ({"optimize_directions": [1, -1]}, "optimize_directions"),
    ],
)
def test_ga_validates_runtime_parameters(override, message):
    with pytest.raises(ValueError, match=message):
        run_ga_mvp(_base_params(**override))
