import numpy as np

import domain.optimization.ga as ga
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


def test_default_heuristic_init_does_not_call_knapsack_mechanic(monkeypatch):
    def forbidden_mech_run(_params):
        raise AssertionError("KnapsackMechanic must not be required for default GA init")

    monkeypatch.setattr(ga, "mech_run", forbidden_mech_run)

    result = run_ga_mvp(_base_params())

    assert result["error"] is None
    assert result["init_mode"] == "heuristic_init"
    assert result["mech_used_for_initialization"] is False
    assert result["mech_seed_count"] == 0
    assert result["heuristic_seed_count"] > 0


def test_random_init_is_available_without_knapsack_mechanic(monkeypatch):
    def forbidden_mech_run(_params):
        raise AssertionError("KnapsackMechanic must not be called in random_init mode")

    monkeypatch.setattr(ga, "mech_run", forbidden_mech_run)

    result = run_ga_mvp(_base_params(init_mode="random_init", max_random_attempts=100))

    assert result["error"] is None
    assert result["init_mode"] == "random_init"
    assert result["mech_used_for_initialization"] is False
    assert result["mech_seed_count"] == 0


def test_pareto_seeded_init_uses_knapsack_mechanic_as_optional_seed(monkeypatch):
    original_mech_run = ga.mech_run
    called = {"value": False}

    def spy_mech_run(params):
        called["value"] = True
        return original_mech_run(params)

    monkeypatch.setattr(ga, "mech_run", spy_mech_run)

    result = run_ga_mvp(_base_params(init_mode="pareto_seeded_init"))

    assert called["value"] is True
    assert result["error"] is None
    assert result["init_mode"] == "pareto_seeded_init"
    assert result["mech_used_for_initialization"] is True
    assert result["mech_seed_count"] > 0


def test_invalid_init_mode_is_rejected():
    with np.testing.assert_raises_regex(ValueError, "init_mode"):
        run_ga_mvp(_base_params(init_mode="full_enum_init"))
