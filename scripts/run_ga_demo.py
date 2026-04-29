from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

OUTPUT_PATH = ROOT / "data" / "examples" / "optimization" / "ga_demo_output.json"


def weight_constraint(subset):
    return sum(item.properties.get("weight", 0) for item in subset)


def perf(subset):
    return sum(
        item.properties.get("CPU", 0) * 0.5 + item.properties.get("RAM", 0) * 0.3
        for item in subset
    )


def energy(subset):
    return sum(item.properties.get("energy", 0) for item in subset)


def main() -> dict:
    from domain.optimization.ga import run_ga_mvp
    from domain.optimization.mech import Item
    from domain.optimization.mech_utils import to_json_serializable
    from infrastructure.logging import configure_logging

    configure_logging(repo_root=ROOT)
    items = [
        Item(name="Server_A", weight=3, CPU=4, RAM=2, energy=5),
        Item(name="Server_B", weight=2, CPU=3, RAM=3, energy=4),
        Item(name="Client_A", weight=1, CPU=2, RAM=1, energy=3),
        Item(name="Server_C", weight=4, CPU=5, RAM=4, energy=6),
    ]

    params = {
        "items": items,
        "max_constraint": 5,
        "constraint_func": weight_constraint,
        "criteria": [
            {"name": "performance", "func": perf, "direction": "max"},
            {"name": "energy", "func": energy, "direction": "min"},
        ],
        "pairwise_matrix": np.array([[1, 3], [1 / 3, 1]]),
        "pop_size": 40,
        "generations": 200,
        "mutation_rate": 0.05,
        "elite": 1,
        "seed": 42,
        "stagnation_limit": 40,
        "top_solutions_limit": 5,
    }

    result = run_ga_mvp(params)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scenario": "ga_mvp_weight_performance_energy",
        "description": (
            "Демонстрационный запуск универсального ядра ГА: ограничение по весу, "
            "максимизация производительности и минимизация энергии."
        ),
        "result": result,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(to_json_serializable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(to_json_serializable(result), ensure_ascii=False, indent=2))
    print(f"\nРезультат сохранён: {OUTPUT_PATH}")
    return result


if __name__ == "__main__":
    main()
