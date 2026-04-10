from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def weight_constraint(subset):
    return sum(item.properties.get("weight", 0) for item in subset)


def perf(subset):
    return sum(
        item.properties.get("CPU", 0) * 0.5 + item.properties.get("RAM", 0) * 0.3 for item in subset
    )


def energy(subset):
    return sum(item.properties.get("energy", 0) for item in subset)


def main() -> None:
    from it_cost_calc.domain.optimization.ga import run_ga_mvp
    from it_cost_calc.domain.optimization.mech import Item
    from it_cost_calc.infrastructure.logging import configure_logging

    configure_logging(repo_root=ROOT)
    items = [
        Item(weight=3, CPU=4, RAM=2, energy=5),
        Item(weight=2, CPU=3, RAM=3, energy=4),
        Item(weight=1, CPU=2, RAM=1, energy=3),
        Item(weight=4, CPU=5, RAM=4, energy=6),
    ]

    params = {
        "items": items,
        "max_constraint": 5,
        "constraint_func": weight_constraint,
        "criteria_funcs": [perf, energy],
        "optimize_directions": [1, -1],
        "pairwise_matrix": np.array([[1, 3], [1 / 3, 1]]),
        "pop_size": 40,
        "generations": 200,
        "mutation_rate": 0.05,
        "elite": 1,
        "seed": 42,
        "stagnation_limit": 40,
    }

    result = run_ga_mvp(params)
    print(result)


if __name__ == "__main__":
    main()
