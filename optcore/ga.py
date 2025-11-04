# ga.py
"""
Минимальный GA (MVP) для задачи о рюкзаке.
Требует наличия mech.run(params).
Всё максимально прямо: init_from_mech -> fitness -> stop -> crossover -> mutation -> repeat.
"""

from typing import List, Tuple, Callable, Dict, Any
import random
import numpy as np
from mech import run as mech_run, Item  # предполагаем, что mech.py в той же папке

# ---- утилиты ----
def subset_from_mask(items: List[Item], mask: Tuple[int, ...]) -> List[Item]:
    return [it for bit, it in zip(mask, items) if bit]

def mask_to_tuple(mask):
    return tuple(int(bool(x)) for x in mask)

def make_mask_from_mech_itemlist(items: List[Item], mech_items_props: List[Dict]) -> Tuple[int, ...]:
    """
    Очень простая процедура сопоставления: если все ключи/значения объекта mech_items_props j
    содержатся в properties исходного Item -> считаем, что это тот item.
    (Работает для однозначных простых примеров; MVP — никаких хитростей.)
    """
    mask = [0] * len(items)
    for j_props in mech_items_props:
        for i, it in enumerate(items):
            # совпадение, если все пары (k,v) из j_props присутствуют и равны в it.properties
            match = True
            for k, v in j_props.items():
                if it.properties.get(k) != v:
                    match = False
                    break
            if match:
                mask[i] = 1
    return tuple(mask)

# ---- GA core MVP ----
def run_ga_mvp(params: Dict[str, Any]) -> Dict[str, Any]:
    # Ожидаемые параметры (минимум)
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

    rnd = random.Random(seed)

    # 1) Получаем начальную популяцию из mech
    mech_params = {
        "items": items,
        "max_constraint": max_constraint,
        "constraint_func": constraint_func,
        "criteria_funcs": criteria_funcs,
        "optimize_directions": optimize_directions,
        "pairwise_matrix": pairwise_matrix,
        "exclude_empty": True
    }
    mech_out = mech_run(mech_params)
    # AHP weights mech вернул в ключе "weights" — используем их как фитнесс-веса
    weights = np.array(mech_out.get("weights", [1.0]*len(criteria_funcs)), dtype=float)

    init_pop = []
    # Если mech вернул парето-решения — используем их
    for p in mech_out.get("pareto", []):
        mask = make_mask_from_mech_itemlist(items, p.get("items", []))
        if any(mask):
            init_pop.append(mask)

    # Если mech не дал достаточно — дополним случайными допустимыми
    def random_mask():
        while True:
            mask = tuple(rnd.choice([0,1]) for _ in items)
            # допускаем пустое? нет, только допустимые непустые
            if sum(mask) == 0:
                continue
            if constraint_func(subset_from_mask(items, mask)) <= max_constraint:
                return mask

    # заполнение популяции
    i = 0
    while len(init_pop) < pop_size:
        init_pop.append(random_mask())
        i += 1
        if i > pop_size*5:
            break  # защита от бесконечного цикла

    population = init_pop[:pop_size]

    # ---- fitness: raw per-criterion -> apply directions -> aggregate via weights ----
    def raw_scores_for_mask(mask):
        subset = subset_from_mask(items, mask)
        raw = []
        for f, d in zip(criteria_funcs, optimize_directions):
            v = f(subset)
            # если fuzzy-like объект, берем .low/.high? MVP: попытка привести к float
            try:
                if hasattr(v, "centroid"):
                    val = float(v.centroid())
                elif hasattr(v, "low") and hasattr(v, "high"):
                    val = float((v.low + v.high)/2.0)
                else:
                    val = float(v)
            except Exception:
                val = 0.0
            raw.append(d * val)  # приводим к "чем больше — тем лучше"
        return np.array(raw, dtype=float)

    def agg_fitness_from_raw(raw_vec):
        # простая взвешенная сумма
        return float(np.dot(raw_vec, weights))

    # ---- genetic operators: one-point crossover, bitflip mutation ----
    def one_point_crossover(a: Tuple[int,...], b: Tuple[int,...]) -> Tuple[Tuple[int,...], Tuple[int,...]]:
        if len(a) <= 1:
            return a, b
        pt = rnd.randrange(1, len(a))
        ca = a[:pt] + b[pt:]
        cb = b[:pt] + a[pt:]
        return tuple(ca), tuple(cb)

    def mutate(mask: Tuple[int,...]) -> Tuple[int,...]:
        m = list(mask)
        for i in range(len(m)):
            if rnd.random() < mutation_rate:
                m[i] = 1 - m[i]
        # ensure constraint: если превысили — простейшая починка: удаляем случайные единицы пока не ок
        while constraint_func(subset_from_mask(items, m)) > max_constraint and any(m):
            ones = [idx for idx,bit in enumerate(m) if bit]
            if not ones:
                break
            drop = rnd.choice(ones)
            m[drop] = 0
        # не допускаем пустой набор
        if sum(m) == 0:
            # попробуем включить один случайный предмет, если влезает
            for idx in rnd.sample(range(len(m)), len(m)):
                m[idx] = 1
                if constraint_func(subset_from_mask(items, m)) <= max_constraint:
                    break
                else:
                    m[idx] = 0
        return tuple(int(x) for x in m)

    # ---- main loop ----
    best = None
    best_score = -1e99
    stagn = 0

    for gen in range(1, generations+1):
        # 2) оценка приспособленности
        raw_list = [raw_scores_for_mask(m) for m in population]
        agg_list = [agg_fitness_from_raw(r) for r in raw_list]

        # find best in gen
        gen_best_idx = int(np.argmax(agg_list))
        gen_best_mask = population[gen_best_idx]
        gen_best_score = agg_list[gen_best_idx]

        # 3) стоп-критерий (simple): generations or stagnation
        if gen == 1 or gen_best_score > best_score + 1e-12:
            best = gen_best_mask
            best_score = gen_best_score
            stagn = 0
        else:
            stagn += 1

        # debug print (по вашему — пока что принтить)
        if gen % max(1, generations//10) == 0 or gen == 1 or gen == generations:
            print(f"[GA MVP] gen {gen}/{generations} best_score={best_score:.6f} gen_best={gen_best_score:.6f} stagn={stagn}")

        if stagn >= stagnation_limit:
            print(f"[GA MVP] ранняя остановка: stagnation_limit={stagnation_limit}")
            break

        # 4) если не остановились — produce next generation
        # сортируем по фитнесс для элитизма
        sorted_idx = sorted(range(len(population)), key=lambda i: agg_list[i], reverse=True)
        next_pop = [population[i] for i in sorted_idx[:elite]]  # элита переносится напрямую

        # турнир/roulette можно, но делаем простую: топ-к + случайные для кроссовера
        # простая селекция: выбираем родителей методом турнирного отбора (k=3)
        def tournament_select():
            k = 3
            cand = rnd.sample(range(len(population)), k)
            best_i = max(cand, key=lambda i: agg_list[i])
            return population[best_i]

        while len(next_pop) < pop_size:
            p1 = tournament_select()
            p2 = tournament_select()
            c1, c2 = one_point_crossover(p1, p2)
            c1 = mutate(c1)
            if len(next_pop) < pop_size:
                next_pop.append(c1)
            if len(next_pop) < pop_size:
                c2 = mutate(c2)
                next_pop.append(c2)

        population = next_pop

    # 5) вернуть/вывести решение (лучшее)
    best_subset = subset_from_mask(items, best) if best is not None else []
    best_raw = raw_scores_for_mask(best) if best is not None else np.zeros(len(criteria_funcs))
    best_agg = agg_fitness_from_raw(best_raw) if best is not None else None
    result = {
        "best_mask": mask_to_tuple(best),
        "best_items": [it.properties for it in best_subset],
        "best_raw_scores": best_raw.tolist(),
        "best_agg": float(best_agg) if best_agg is not None else None,
        "generations_ran": gen
    }
    print("[GA MVP] finished. best_agg=", result["best_agg"])
    return result
