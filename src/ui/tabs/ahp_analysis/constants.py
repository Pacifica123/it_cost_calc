import numpy as np

DEFAULT_SOFT_CRITERIA = [
    "avg_reliability",
    "total_performance",
    "total_cost",
    "total_energy",
    "lifespan",
]

M1 = np.array(
    [
        [1, 3, 5, 4, 3],
        [1 / 3, 1, 3, 2, 2],
        [1 / 5, 1 / 3, 1, 1 / 2, 1 / 3],
        [1 / 4, 1 / 2, 2, 1, 1 / 2],
        [1 / 3, 1 / 2, 3, 2, 1],
    ],
    dtype=float,
)
M2 = np.array(
    [
        [1, 2, 4, 3, 2],
        [1 / 2, 1, 2, 2, 1],
        [1 / 4, 1 / 2, 1, 1 / 2, 1 / 3],
        [1 / 3, 1 / 2, 2, 1, 1 / 2],
        [1 / 2, 1, 3, 2, 1],
    ],
    dtype=float,
)

DEFAULT_EXPERT_MATRICES = [M1, M2]
DEFAULT_CONSTRAINTS = {
    "max_budget": 6000.0,
    "max_energy": 700.0,
    "people_match_tolerance": 0.25,
}


SOFT_CRITERIA_LABELS = {
    "avg_reliability": "Надежность",
    "total_performance": "Производительность",
    "total_cost": "Совокупная стоимость",
    "total_energy": "Энергопотребление",
    "lifespan": "Срок службы / запас устойчивости",
}

SOFT_CRITERIA_SHORT_LABELS = {
    "avg_reliability": "Надежность",
    "total_performance": "Производит.",
    "total_cost": "Стоимость",
    "total_energy": "Энергия",
    "lifespan": "Срок службы",
}
