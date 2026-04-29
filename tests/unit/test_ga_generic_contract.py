import numpy as np

import domain.optimization.ga as ga
from domain.optimization.ga import run_ga_mvp
from domain.optimization.mech import Item


def _sum_prop(prop):
    return lambda subset: sum(item.properties.get(prop, 0) for item in subset)


def _count_selected(subset):
    return len(subset)


def test_ga_accepts_generic_criteria_and_multiple_constraints_without_it_domain(monkeypatch):
    def forbidden_mech_run(_params):
        raise AssertionError("Generic GA contract must not require KnapsackMechanic")

    monkeypatch.setattr(ga, "mech_run", forbidden_mech_run)

    items = [
        Item(name="Option A", quality=7, price=50, risk=0.5, delivery_days=7),
        Item(name="Option B", quality=11, price=80, risk=0.2, delivery_days=12),
        Item(name="Option C", quality=5, price=20, risk=0.1, delivery_days=3),
    ]

    result = run_ga_mvp(
        {
            "items": items,
            "criteria": [
                {"name": "quality", "func": _sum_prop("quality"), "direction": "max"},
                {"name": "price", "func": _sum_prop("price"), "direction": "min"},
                {"name": "risk", "func": _sum_prop("risk"), "direction": "min"},
            ],
            "constraints": [
                {"name": "budget", "func": _sum_prop("price"), "operator": "<=", "bound": 100},
                {"name": "max_delivery", "func": _sum_prop("delivery_days"), "operator": "<=", "bound": 15},
                {"name": "at_least_two_options", "func": _count_selected, "operator": ">=", "bound": 2},
            ],
            "weights": [0.6, 0.25, 0.15],
            "pop_size": 10,
            "generations": 8,
            "mutation_rate": 0.05,
            "elite": 1,
            "seed": 11,
            "stagnation_limit": 20,
        }
    )

    assert result["error"] is None
    assert result["weights_source"] == "direct"
    assert result["criterion_names"] == ["quality", "price", "risk"]
    assert result["criterion_directions"] == ["max", "min", "min"]
    selected_price = sum(item["price"] for item in result["best_items"])
    selected_delivery = sum(item["delivery_days"] for item in result["best_items"])
    assert selected_price <= 100
    assert selected_delivery <= 15
    assert len(result["best_items"]) >= 2
    assert all(detail["passed"] for detail in result["constraints"])
    assert {detail["name"] for detail in result["criteria"]} == {"quality", "price", "risk"}
    assert result["mech_used_for_initialization"] is False


def test_ga_supports_predicate_constraints_and_equal_default_weights():
    items = [
        Item(name="Alpha", score=3, category="a"),
        Item(name="Beta", score=8, category="b"),
        Item(name="Gamma", score=2, category="c"),
    ]

    result = run_ga_mvp(
        {
            "items": items,
            "criteria": [{"name": "score", "func": _sum_prop("score"), "direction": "max"}],
            "constraints": [
                {
                    "name": "contains_b",
                    "predicate": lambda subset: any(
                        item.properties.get("category") == "b" for item in subset
                    ),
                }
            ],
            "pop_size": 6,
            "generations": 5,
            "mutation_rate": 0,
            "elite": 1,
            "seed": 5,
            "stagnation_limit": 10,
        }
    )

    assert result["error"] is None
    assert result["weights_source"] == "equal_default"
    assert result["weights"] == [1.0]
    assert result["constraints"][0]["name"] == "contains_b"
    assert result["constraints"][0]["passed"] is True
    assert any(item["name"] == "Beta" for item in result["best_items"])


def test_legacy_constraint_and_criteria_contract_still_works():
    items = [
        Item(name="A", weight=1, value=3),
        Item(name="B", weight=2, value=5),
        Item(name="C", weight=5, value=20),
    ]

    result = run_ga_mvp(
        {
            "items": items,
            "max_constraint": 3,
            "constraint_func": _sum_prop("weight"),
            "criteria_funcs": [_sum_prop("value")],
            "optimize_directions": [1],
            "pairwise_matrix": np.array([[1]], dtype=float),
            "pop_size": 8,
            "generations": 5,
            "mutation_rate": 0,
            "elite": 1,
            "seed": 42,
            "stagnation_limit": 10,
        }
    )

    assert result["error"] is None
    assert result["weights_source"] == "ahp_pairwise_matrix"
    assert result["constraints"][0]["name"] == "legacy_constraint"
    assert result["constraints"][0]["value"] <= 3
    assert result["best_items"]
    assert "best_raw_scores" in result


def test_pareto_seeded_init_falls_back_for_generic_multi_constraint_contract(monkeypatch):
    def forbidden_mech_run(_params):
        raise AssertionError("Generic multi-constraint GA must not call KnapsackMechanic")

    monkeypatch.setattr(ga, "mech_run", forbidden_mech_run)

    result = run_ga_mvp(
        {
            "items": [Item(name="A", quality=5, price=10), Item(name="B", quality=7, price=20)],
            "criteria": [{"name": "quality", "func": _sum_prop("quality"), "direction": "max"}],
            "constraints": [
                {"name": "budget", "func": _sum_prop("price"), "operator": "<=", "bound": 30},
                {"name": "min_count", "func": _count_selected, "operator": ">=", "bound": 1},
            ],
            "weights": [1.0],
            "init_mode": "pareto_seeded_init",
            "pop_size": 6,
            "generations": 3,
            "mutation_rate": 0,
            "elite": 1,
            "seed": 4,
            "stagnation_limit": 10,
        }
    )

    assert result["error"] is None
    assert result["init_mode"] == "pareto_seeded_init"
    assert result["mech_used_for_initialization"] is False
    assert result["heuristic_seed_count"] > 0
