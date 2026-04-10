import numpy as np

from it_cost_calc.domain.optimization import Item, run


def test_mech_wrapper_keeps_run_entrypoint_working():
    items = [
        Item(weight=1, value=3, energy=3),
        Item(weight=2, value=5, energy=4),
        Item(weight=3, value=6, energy=7),
    ]

    def weight(subset):
        return sum(item.properties.get("weight", 0) for item in subset)

    def value(subset):
        return sum(item.properties.get("value", 0) for item in subset)

    def energy(subset):
        return sum(item.properties.get("energy", 0) for item in subset)

    result = run(
        {
            "items": items,
            "max_constraint": 3,
            "constraint_func": weight,
            "criteria_funcs": [value, energy],
            "optimize_directions": [1, -1],
            "pairwise_matrix": np.array([[1, 2], [1 / 2, 1]], dtype=float),
        }
    )

    assert "weights" in result
    assert isinstance(result["pareto"], list)
