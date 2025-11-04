# run_mvp_example.py
from mech import Item
from ga import run_ga_mvp
import numpy as np

items = [
    Item(weight=3, CPU=4, RAM=2, energy=5),
    Item(weight=2, CPU=3, RAM=3, energy=4),
    Item(weight=1, CPU=2, RAM=1, energy=3),
    Item(weight=4, CPU=5, RAM=4, energy=6)
]

def weight_constraint(subset):
    return sum(it.properties.get("weight",0) for it in subset)

def perf(subset):
    return sum(it.properties.get("CPU",0)*0.5 + it.properties.get("RAM",0)*0.3 for it in subset)

def energy(subset):
    return sum(it.properties.get("energy",0) for it in subset)

params = {
    "items": items,
    "max_constraint": 5,
    "constraint_func": weight_constraint,
    "criteria_funcs": [perf, energy],
    "optimize_directions": [1, -1],
    "pairwise_matrix": np.array([[1,3],[1/3,1]]),
    "pop_size": 40,
    "generations": 200,
    "mutation_rate": 0.05,
    "elite": 1,
    "seed": 42,
    "stagnation_limit": 40
}

res = run_ga_mvp(params)
print(res)
