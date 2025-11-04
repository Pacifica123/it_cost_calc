# pipeline_knapsack.py
"""
Полный минимальный пайплайн:
1. объединяет все типы оборудования в items
2. запускает mech (жадный) -> получает начальную популяцию
3. запускает GA (га-мвп)
4. выводит сравнительный отчёт
"""

import numpy as np
from mech import Item, run as mech_run
from ga import run_ga_mvp

# --- 1. Определяем набор оборудования ---
server_hw = [
    Item(name="Server_A", weight=8, cost=4000, energy=500, perf=700),
    Item(name="Server_B", weight=6, cost=3500, energy=420, perf=600),
    Item(name="Server_C", weight=10, cost=5000, energy=550, perf=800),
]

network_hw = [
    Item(name="Switch_A", weight=1, cost=500, energy=50, perf=100),
    Item(name="Router_A", weight=1, cost=700, energy=80, perf=120),
    Item(name="Firewall_A", weight=2, cost=1200, energy=100, perf=200),
]

client_hw = [
    Item(name="Client_A", weight=2, cost=800, energy=100, perf=150),
    Item(name="Client_B", weight=3, cost=1000, energy=130, perf=180),
    Item(name="Client_C", weight=4, cost=1500, energy=160, perf=250),
]

items = server_hw + network_hw + client_hw

# --- 2. Определяем критерии и ограничения ---
MAX_BUDGET = 10000
MAX_ENERGY = 1000

def budget_constraint(subset):
    return sum(it.properties.get("cost", 0) for it in subset)

def energy_constraint(subset):
    return sum(it.properties.get("energy", 0) for it in subset)

# комбинированное ограничение: и по бюджету, и по энергопотреблению
def combined_constraint(subset):
    return budget_constraint(subset) + energy_constraint(subset) * 2  # просто взвешиваем, MVP

def performance(subset):
    return sum(it.properties.get("perf", 0) for it in subset)

def total_energy(subset):
    return sum(it.properties.get("energy", 0) for it in subset)

def total_cost(subset):
    return sum(it.properties.get("cost", 0) for it in subset)

pairwise = np.array([
    [1, 5, 3],
    [1/5, 1, 1/2],
    [1/3, 2, 1]
])

criteria_funcs = [performance, total_energy, total_cost]
optimize_directions = [1, -1, -1]  # хотим максимум производительности, минимум энергии и цены

# --- 3. Прогоняем mech (жадный) ---
mech_params = {
    "items": items,
    "max_constraint": MAX_BUDGET,  # допустим ограничение по бюджету
    "constraint_func": budget_constraint,
    "criteria_funcs": criteria_funcs,
    "optimize_directions": optimize_directions,
    "pairwise_matrix": pairwise,
    "exclude_empty": True
}

print("\n=== Жадный алгоритм (mech) ===")
mech_out = mech_run(mech_params)

start_pop = mech_out.get("pareto", [])
print(f"Получено {len(start_pop)} конфигураций начальной популяции.")
for i, conf in enumerate(start_pop, 1):
    print(f"--- Конфигурация {i} ---")
    for it in conf["items"]:
        print(" ", it)
    print("  raw_scores:", conf["raw_scores"], "agg_score:", conf.get("agg_score"))

# --- 4. Прогоняем GA ---
def mech_seed_from_mech():
    from ga_mvp import make_mask_from_mech_itemlist
    seeds = []
    for p in start_pop:
        mask = make_mask_from_mech_itemlist(items, p["items"])
        seeds.append(mask)
    return seeds

ga_params = {
    "items": items,
    "max_constraint": MAX_BUDGET,
    "constraint_func": budget_constraint,
    "criteria_funcs": criteria_funcs,
    "optimize_directions": optimize_directions,
    "pairwise_matrix": pairwise,
    "pop_size": 50,
    "generations": 100,
    "mutation_rate": 0.05,
    "elite": 1,
    "seed": 42,
    "stagnation_limit": 40
}

print("\n=== Генетический алгоритм (GA) ===")
ga_result = run_ga_mvp(ga_params)

# --- 5. Сравнение ---
print("\n=== Сравнение результатов ===")
print("Начальная популяция:", len(start_pop))
print("Лучшее решение после ГА:")
for it in ga_result["best_items"]:
    print(" ", it)
print("Суммарная производительность:", ga_result["best_raw_scores"][0])
print("Суммарное энергопотребление:", ga_result["best_raw_scores"][1])
print("Суммарная стоимость:", ga_result["best_raw_scores"][2])
print("Итоговый агрегированный фитнесс:", ga_result["best_agg"])
