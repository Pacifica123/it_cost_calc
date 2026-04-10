"""Минимальный генетический алгоритм для задачи о рюкзаке."""

from __future__ import annotations

import logging
import random
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from .mech import Item, run as mech_run

logger = logging.getLogger(__name__)


def subset_from_mask(items: List[Item], mask: Tuple[int, ...]) -> List[Item]:
    return [item for bit, item in zip(mask, items) if bit]


def mask_to_tuple(mask: Tuple[int, ...] | None) -> Tuple[int, ...]:
    if mask is None:
        return tuple()
    return tuple(int(bool(x)) for x in mask)


def make_mask_from_mech_itemlist(
    items: List[Item], mech_items_props: List[Dict]
) -> Tuple[int, ...]:
    """Сопоставляет список свойств из mech с исходным набором Item."""
    mask = [0] * len(items)
    for mech_props in mech_items_props:
        for index, item in enumerate(items):
            match = True
            for key, value in mech_props.items():
                if item.properties.get(key) != value:
                    match = False
                    break
            if match:
                mask[index] = 1
    return tuple(mask)


def run_ga_mvp(params: Dict[str, Any]) -> Dict[str, Any]:
    items: List[Item] = params["items"]
    max_constraint = params["max_constraint"]
    constraint_func: Callable[[List[Item]], float] = params["constraint_func"]
    criteria_funcs = params["criteria_funcs"]
    optimize_directions = params["optimize_directions"]
    pairwise_matrix = params["pairwise_matrix"]

    pop_size = int(params.get("pop_size", 50))
    generations = int(params.get("generations", 200))
    mutation_rate = float(params.get("mutation_rate", 0.02))
    elite = int(params.get("elite", 1))
    seed = params.get("seed", None)
    stagnation_limit = int(params.get("stagnation_limit", 50))

    logger.info(
        "Запуск GA MVP: items=%s, pop_size=%s, generations=%s, mutation_rate=%s",
        len(items),
        pop_size,
        generations,
        mutation_rate,
    )
    rnd = random.Random(seed)

    mech_params = {
        "items": items,
        "max_constraint": max_constraint,
        "constraint_func": constraint_func,
        "criteria_funcs": criteria_funcs,
        "optimize_directions": optimize_directions,
        "pairwise_matrix": pairwise_matrix,
        "exclude_empty": True,
    }
    mech_out = mech_run(mech_params)
    weights = np.array(mech_out.get("weights", [1.0] * len(criteria_funcs)), dtype=float)
    logger.debug("GA получил веса из mech: %s", weights.tolist())

    init_pop = []
    for solution in mech_out.get("pareto", []):
        mask = make_mask_from_mech_itemlist(items, solution.get("items", []))
        if any(mask):
            init_pop.append(mask)

    def random_mask() -> Tuple[int, ...]:
        while True:
            mask = tuple(rnd.choice([0, 1]) for _ in items)
            if sum(mask) == 0:
                continue
            if constraint_func(subset_from_mask(items, mask)) <= max_constraint:
                return mask

    attempts = 0
    while len(init_pop) < pop_size:
        init_pop.append(random_mask())
        attempts += 1
        if attempts > pop_size * 5:
            logger.warning("Ранний выход при доборе стартовой популяции GA")
            break

    population = init_pop[:pop_size]
    logger.info("Стартовая популяция GA подготовлена: %s решений", len(population))

    def raw_scores_for_mask(mask: Tuple[int, ...]) -> np.ndarray:
        subset = subset_from_mask(items, mask)
        raw = []
        for func, direction in zip(criteria_funcs, optimize_directions):
            value = func(subset)
            try:
                if hasattr(value, "centroid"):
                    scalar = float(value.centroid())
                elif hasattr(value, "low") and hasattr(value, "high"):
                    scalar = float((value.low + value.high) / 2.0)
                else:
                    scalar = float(value)
            except Exception:
                logger.exception(
                    "Не удалось привести значение критерия к float, будет использован 0.0"
                )
                scalar = 0.0
            raw.append(direction * scalar)
        return np.array(raw, dtype=float)

    def agg_fitness_from_raw(raw_vec: np.ndarray) -> float:
        return float(np.dot(raw_vec, weights))

    def one_point_crossover(
        left: Tuple[int, ...], right: Tuple[int, ...]
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        if len(left) <= 1:
            return left, right
        point = rnd.randrange(1, len(left))
        child_a = left[:point] + right[point:]
        child_b = right[:point] + left[point:]
        return tuple(child_a), tuple(child_b)

    def mutate(mask: Tuple[int, ...]) -> Tuple[int, ...]:
        mutable_mask = list(mask)
        for index in range(len(mutable_mask)):
            if rnd.random() < mutation_rate:
                mutable_mask[index] = 1 - mutable_mask[index]
        while constraint_func(subset_from_mask(items, mutable_mask)) > max_constraint and any(
            mutable_mask
        ):
            ones = [idx for idx, bit in enumerate(mutable_mask) if bit]
            if not ones:
                break
            drop = rnd.choice(ones)
            mutable_mask[drop] = 0
        if sum(mutable_mask) == 0:
            for index in rnd.sample(range(len(mutable_mask)), len(mutable_mask)):
                mutable_mask[index] = 1
                if constraint_func(subset_from_mask(items, mutable_mask)) <= max_constraint:
                    break
                mutable_mask[index] = 0
        return tuple(int(x) for x in mutable_mask)

    best = None
    best_score = -1e99
    stagnation = 0
    gen = 0

    for gen in range(1, generations + 1):
        raw_list = [raw_scores_for_mask(mask) for mask in population]
        agg_list = [agg_fitness_from_raw(raw) for raw in raw_list]

        gen_best_idx = int(np.argmax(agg_list))
        gen_best_mask = population[gen_best_idx]
        gen_best_score = agg_list[gen_best_idx]

        if gen == 1 or gen_best_score > best_score + 1e-12:
            best = gen_best_mask
            best_score = gen_best_score
            stagnation = 0
        else:
            stagnation += 1

        if gen % max(1, generations // 10) == 0 or gen == 1 or gen == generations:
            logger.debug(
                "GA progress gen=%s/%s best_score=%.6f gen_best=%.6f stagnation=%s",
                gen,
                generations,
                best_score,
                gen_best_score,
                stagnation,
            )

        if stagnation >= stagnation_limit:
            logger.info("GA ранне остановлен по stagnation_limit=%s", stagnation_limit)
            break

        sorted_idx = sorted(range(len(population)), key=lambda idx: agg_list[idx], reverse=True)
        next_pop = [population[idx] for idx in sorted_idx[:elite]]

        def tournament_select(
            current_population: List[Tuple[int, ...]] = population,
            current_agg_list: List[float] = agg_list,
        ) -> Tuple[int, ...]:
            candidates = rnd.sample(range(len(current_population)), 3)
            best_idx_local = max(candidates, key=current_agg_list.__getitem__)
            return current_population[best_idx_local]

        while len(next_pop) < pop_size:
            parent_a = tournament_select()
            parent_b = tournament_select()
            child_a, child_b = one_point_crossover(parent_a, parent_b)
            child_a = mutate(child_a)
            if len(next_pop) < pop_size:
                next_pop.append(child_a)
            if len(next_pop) < pop_size:
                child_b = mutate(child_b)
                next_pop.append(child_b)

        population = next_pop

    best_subset = subset_from_mask(items, best) if best is not None else []
    best_raw = raw_scores_for_mask(best) if best is not None else np.zeros(len(criteria_funcs))
    best_agg = agg_fitness_from_raw(best_raw) if best is not None else None
    result = {
        "best_mask": mask_to_tuple(best),
        "best_items": [item.properties for item in best_subset],
        "best_raw_scores": best_raw.tolist(),
        "best_agg": float(best_agg) if best_agg is not None else None,
        "generations_ran": gen,
    }
    logger.info("GA завершён. Лучший агрегированный score=%s", result["best_agg"])
    return result
