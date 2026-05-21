"""Универсальное ядро генетического алгоритма для выбора подмножества item-ов.

Модуль не привязан к конкретной предметной области. Предметный смысл
критериев и ограничений задаётся извне через callable-функции и спецификации.
Старый контракт ``constraint_func`` / ``max_constraint`` / ``criteria_funcs``
сохранён для обратной совместимости.
"""

from __future__ import annotations

import logging
import math
import operator
import random
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np

from .mech import Item, run as mech_run

logger = logging.getLogger(__name__)

TERMINATION_MAX_GENERATIONS = "max_generations"
TERMINATION_STAGNATION = "stagnation"
TERMINATION_NO_FEASIBLE = "no_feasible_solution"

INIT_MODE_RANDOM = "random_init"
INIT_MODE_HEURISTIC = "heuristic_init"
INIT_MODE_PARETO_SEEDED = "pareto_seeded_init"
DEFAULT_INIT_MODE = INIT_MODE_HEURISTIC
ALLOWED_INIT_MODES = {INIT_MODE_RANDOM, INIT_MODE_HEURISTIC, INIT_MODE_PARETO_SEEDED}

Mask = Tuple[int, ...]
CriterionSpec = Dict[str, Any]
ConstraintSpec = Dict[str, Any]

_OPERATORS: Dict[str, Callable[[float, float], bool]] = {
    "<=": operator.le,
    "=<": operator.le,
    "lt_eq": operator.le,
    "le": operator.le,
    ">=": operator.ge,
    "=>": operator.ge,
    "gt_eq": operator.ge,
    "ge": operator.ge,
    "<": operator.lt,
    "lt": operator.lt,
    ">": operator.gt,
    "gt": operator.gt,
    "==": operator.eq,
    "=": operator.eq,
    "eq": operator.eq,
    "!=": operator.ne,
    "ne": operator.ne,
}


class _ConstraintEvaluationError(ValueError):
    """Внутренняя ошибка оценки ограничения."""


def subset_from_mask(items: List[Item], mask: Sequence[int]) -> List[Item]:
    return [item for bit, item in zip(mask, items) if bit]


def mask_to_tuple(mask: Sequence[int] | None) -> Mask:
    if mask is None:
        return tuple()
    return tuple(int(bool(x)) for x in mask)


def make_mask_from_mech_itemlist(items: List[Item], mech_items_props: List[Dict]) -> Mask:
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
                break
    return tuple(mask)


def _as_finite_float(value: Any, name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _as_positive_int(value: Any, name: str) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if numeric < 1:
        raise ValueError(f"{name} must be >= 1")
    return numeric


def _callable_name(func: Callable[..., Any], fallback: str) -> str:
    return getattr(func, "__name__", None) or fallback


def _direction_to_int(direction: Any, name: str) -> int:
    if isinstance(direction, str):
        normalized = direction.strip().lower()
        if normalized in {"max", "maximize", "maximise", "+", "1"}:
            return 1
        if normalized in {"min", "minimize", "minimise", "-", "-1"}:
            return -1
        raise ValueError(f"{name} direction must be 'max', 'min', 1 or -1")

    try:
        numeric = int(direction)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} direction must be 'max', 'min', 1 or -1") from exc
    if numeric not in (-1, 1):
        raise ValueError(f"{name} direction must be 'max', 'min', 1 or -1")
    return numeric


def _direction_label(direction: int) -> str:
    return "max" if direction == 1 else "min"


def _normalize_weights(weights: Sequence[Any], criteria_count: int) -> np.ndarray:
    vector = np.array(weights, dtype=float)
    if vector.shape != (criteria_count,):
        raise ValueError("weights must match number of criteria")
    if not np.all(np.isfinite(vector)):
        raise ValueError("weights must contain only finite numbers")
    if np.any(vector < 0):
        raise ValueError("weights must be non-negative")
    total = float(vector.sum())
    if total <= 1e-12:
        raise ValueError("weights sum must be greater than zero")
    return vector / total


def _criteria_from_params(params: Dict[str, Any]) -> List[CriterionSpec]:
    """Возвращает универсальные спецификации критериев.

    Новый контракт принимает ``criteria``:
    ``{"name": "quality", "func": callable, "direction": "max"}``.
    Старый контракт ``criteria_funcs`` + ``optimize_directions`` преобразуется
    в те же внутренние спецификации.
    """
    if "criteria" in params and params.get("criteria") is not None:
        criteria_input = list(params.get("criteria") or [])
        if not criteria_input:
            raise ValueError("criteria must contain at least one criterion")

        criteria: List[CriterionSpec] = []
        for index, spec in enumerate(criteria_input):
            fallback_name = f"criterion_{index + 1}"
            if callable(spec):
                func = spec
                name = _callable_name(func, fallback_name)
                direction = 1
            elif isinstance(spec, dict):
                func = spec.get("func", spec.get("function"))
                if not callable(func):
                    raise ValueError(f"criteria[{index}].func must be callable")
                name = str(spec.get("name") or _callable_name(func, fallback_name))
                direction = _direction_to_int(spec.get("direction", 1), name)
            else:
                raise ValueError("each criteria entry must be a dict or callable")

            criteria.append(
                {
                    "name": name,
                    "func": func,
                    "direction": direction,
                    "direction_label": _direction_label(direction),
                }
            )
        return criteria

    if "criteria_funcs" not in params or "optimize_directions" not in params:
        raise ValueError(
            "Missing GA parameter(s): criteria or criteria_funcs + optimize_directions"
        )

    criteria_funcs = list(params.get("criteria_funcs") or [])
    if not criteria_funcs:
        raise ValueError("criteria_funcs must contain at least one criterion")
    if any(not callable(func) for func in criteria_funcs):
        raise ValueError("all criteria_funcs entries must be callable")

    optimize_directions = [_direction_to_int(direction, "legacy criterion") for direction in params["optimize_directions"]]
    if len(optimize_directions) != len(criteria_funcs):
        raise ValueError("optimize_directions must match criteria_funcs length")

    names = list(params.get("criteria_names") or [])
    criteria = []
    for index, (func, direction) in enumerate(zip(criteria_funcs, optimize_directions)):
        fallback_name = f"criterion_{index + 1}"
        name = str(names[index]) if index < len(names) else _callable_name(func, fallback_name)
        criteria.append(
            {
                "name": name,
                "func": func,
                "direction": direction,
                "direction_label": _direction_label(direction),
            }
        )
    return criteria


def _operator_label(value: Any) -> str:
    op = str(value or "<=").strip().lower()
    if op not in _OPERATORS:
        allowed = ", ".join(sorted({"<=", ">=", "<", ">", "==", "!="}))
        raise ValueError(f"constraint operator must be one of: {allowed}")
    canonical = {
        "=<": "<=",
        "lt_eq": "<=",
        "le": "<=",
        "=>": ">=",
        "gt_eq": ">=",
        "ge": ">=",
        "lt": "<",
        "gt": ">",
        "=": "==",
        "eq": "==",
        "ne": "!=",
    }
    return canonical.get(op, op)


def _constraints_from_params(params: Dict[str, Any]) -> List[ConstraintSpec]:
    """Возвращает универсальные спецификации ограничений.

    Новый контракт принимает список ``constraints``. Старый контракт
    ``constraint_func`` + ``max_constraint`` автоматически превращается в одно
    ограничение ``legacy_constraint <= max_constraint``.
    """
    if "constraints" in params and params.get("constraints") is not None:
        constraints_input = list(params.get("constraints") or [])
        constraints: List[ConstraintSpec] = []
        for index, spec in enumerate(constraints_input):
            fallback_name = f"constraint_{index + 1}"
            if callable(spec):
                constraints.append(
                    {
                        "name": _callable_name(spec, fallback_name),
                        "predicate": spec,
                        "func": None,
                        "operator": None,
                        "bound": None,
                        "source": "generic",
                    }
                )
                continue

            if not isinstance(spec, dict):
                raise ValueError("each constraints entry must be a dict or callable")

            func = spec.get("func", spec.get("function"))
            predicate = spec.get("predicate")
            if predicate is not None and not callable(predicate):
                raise ValueError(f"constraints[{index}].predicate must be callable")
            if func is not None and not callable(func):
                raise ValueError(f"constraints[{index}].func must be callable")
            if predicate is None and func is None:
                raise ValueError(
                    f"constraints[{index}] must contain either func or predicate"
                )

            name = str(
                spec.get("name")
                or _callable_name(predicate or func, fallback_name)  # type: ignore[arg-type]
            )
            if predicate is not None and "bound" not in spec:
                constraints.append(
                    {
                        "name": name,
                        "predicate": predicate,
                        "func": func,
                        "operator": None,
                        "bound": None,
                        "source": "generic",
                    }
                )
                continue

            if "bound" not in spec and "max" not in spec and "limit" not in spec:
                # func без bound трактуем как bool-предикат по результату func(subset).
                constraints.append(
                    {
                        "name": name,
                        "predicate": None,
                        "func": func,
                        "operator": None,
                        "bound": None,
                        "source": "generic",
                    }
                )
                continue

            raw_bound = spec.get("bound", spec.get("max", spec.get("limit")))
            bound = _as_finite_float(raw_bound, f"constraints[{index}].bound")
            op = _operator_label(spec.get("operator", spec.get("op", spec.get("comparator", "<="))))
            constraints.append(
                {
                    "name": name,
                    "predicate": predicate,
                    "func": func,
                    "operator": op,
                    "bound": bound,
                    "source": "generic",
                }
            )
        return constraints

    if "constraint_func" not in params or "max_constraint" not in params:
        return []

    constraint_func = params["constraint_func"]
    if not callable(constraint_func):
        raise ValueError("constraint_func must be callable")
    max_constraint = _as_finite_float(params["max_constraint"], "max_constraint")
    return [
        {
            "name": str(params.get("constraint_name") or "legacy_constraint"),
            "predicate": None,
            "func": constraint_func,
            "operator": "<=",
            "bound": max_constraint,
            "source": "legacy",
        }
    ]


def _pairwise_matrix_from_params(params: Dict[str, Any], criteria_count: int) -> np.ndarray | None:
    if "pairwise_matrix" not in params or params.get("pairwise_matrix") is None:
        return None
    pairwise_matrix = np.array(params["pairwise_matrix"], dtype=float)
    expected_shape = (criteria_count, criteria_count)
    if pairwise_matrix.shape != expected_shape:
        raise ValueError("pairwise_matrix must be square and match number of criteria")
    if not np.all(np.isfinite(pairwise_matrix)):
        raise ValueError("pairwise_matrix must contain only finite numbers")
    return pairwise_matrix


def _validate_params(params: Dict[str, Any]) -> Dict[str, Any]:
    if "items" not in params:
        raise ValueError("Missing GA parameter(s): items")

    items = list(params.get("items") or [])
    criteria = _criteria_from_params(params)
    constraints = _constraints_from_params(params)
    pairwise_matrix = _pairwise_matrix_from_params(params, len(criteria))

    if params.get("weights") is not None:
        weights = _normalize_weights(params["weights"], len(criteria))
        weights_source = "direct"
    elif pairwise_matrix is not None:
        weights = _calculate_ahp_weights(pairwise_matrix)
        weights_source = "ahp_pairwise_matrix"
    else:
        weights = np.ones(len(criteria), dtype=float) / len(criteria)
        weights_source = "equal_default"

    pop_size = _as_positive_int(params.get("pop_size", 50), "pop_size")
    generations = _as_positive_int(params.get("generations", 200), "generations")
    mutation_rate = _as_finite_float(params.get("mutation_rate", 0.02), "mutation_rate")
    if not 0.0 <= mutation_rate <= 1.0:
        raise ValueError("mutation_rate must be in the [0, 1] range")

    try:
        elite = int(params.get("elite", 1))
    except (TypeError, ValueError) as exc:
        raise ValueError("elite must be an integer") from exc
    if elite < 0:
        raise ValueError("elite must be >= 0")
    if elite > pop_size:
        raise ValueError("elite must be <= pop_size")

    stagnation_limit = _as_positive_int(
        params.get("stagnation_limit", 50), "stagnation_limit"
    )
    max_random_attempts = _as_positive_int(
        params.get("max_random_attempts", max(1000, pop_size * 50)),
        "max_random_attempts",
    )
    top_solutions_limit = int(params.get("top_solutions_limit", min(10, pop_size)))
    if top_solutions_limit < 0:
        raise ValueError("top_solutions_limit must be >= 0")

    quality_check_enabled = bool(params.get("quality_check_enabled", True))
    quality_check_max_items = int(params.get("quality_check_max_items", 18))
    if quality_check_max_items < 0:
        raise ValueError("quality_check_max_items must be >= 0")

    init_mode = str(
        params.get("init_mode", params.get("initialization_mode", DEFAULT_INIT_MODE))
    )
    if init_mode not in ALLOWED_INIT_MODES:
        allowed = ", ".join(sorted(ALLOWED_INIT_MODES))
        raise ValueError(f"init_mode must be one of: {allowed}")

    allow_empty = bool(params.get("allow_empty", not bool(params.get("exclude_empty", True))))

    return {
        "items": items,
        "criteria": criteria,
        "constraints": constraints,
        "pairwise_matrix": pairwise_matrix,
        "weights": weights,
        "weights_source": weights_source,
        "pop_size": pop_size,
        "generations": generations,
        "mutation_rate": mutation_rate,
        "elite": elite,
        "seed": params.get("seed", None),
        "stagnation_limit": stagnation_limit,
        "max_random_attempts": max_random_attempts,
        "top_solutions_limit": top_solutions_limit,
        "quality_check_enabled": quality_check_enabled,
        "quality_check_max_items": quality_check_max_items,
        "init_mode": init_mode,
        "allow_empty": allow_empty,
    }


def _calculate_ahp_weights(pairwise_matrix: np.ndarray) -> np.ndarray:
    """Вычисляет нормированные веса AHP без запуска переборного механизма."""
    eigenvalues, eigenvectors = np.linalg.eig(pairwise_matrix)
    max_index = int(np.argmax(eigenvalues.real))
    principal = np.abs(eigenvectors[:, max_index].real)
    total = float(principal.sum())
    if total <= 1e-12 or not np.isfinite(total):
        raise ValueError("pairwise_matrix produced an invalid AHP weight vector")
    weights = principal / total
    if not np.all(np.isfinite(weights)):
        raise ValueError("AHP weights must contain only finite numbers")
    return weights


def _scalarize_criterion_value(value: Any) -> float:
    """Приводит значение критерия к float, включая простые fuzzy-объекты."""
    if hasattr(value, "centroid"):
        return float(value.centroid())
    if hasattr(value, "low") and hasattr(value, "high"):
        return float((value.low + value.high) / 2.0)
    return float(value)


def _normalize_raw_scores(
    raw_vec: np.ndarray,
    mins: np.ndarray,
    maxs: np.ndarray,
) -> np.ndarray:
    """
    Нормализует направленные значения критериев в диапазон 0..1.

    Для критериев минимизации исходное значение заранее умножается на -1,
    поэтому после нормализации для всех критериев действует правило
    "чем больше — тем лучше".
    """
    ranges = maxs - mins
    normalized = np.ones_like(raw_vec, dtype=float)
    variable = np.abs(ranges) > 1e-12
    normalized[variable] = (raw_vec[variable] - mins[variable]) / ranges[variable]
    return np.clip(normalized, 0.0, 1.0)


def _normalization_bounds_from_scores(
    raw_scores: Sequence[np.ndarray],
    criteria_count: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Возвращает min/max для нормализации по набору направленных оценок."""
    if not raw_scores:
        zeros = np.zeros(criteria_count, dtype=float)
        return zeros, zeros
    arr = np.array(raw_scores, dtype=float)
    return arr.min(axis=0), arr.max(axis=0)


def _criteria_metadata(criteria: Sequence[CriterionSpec]) -> List[Dict[str, Any]]:
    return [
        {
            "name": criterion["name"],
            "direction": criterion["direction_label"],
        }
        for criterion in criteria
    ]


def _constraint_metadata(constraints: Sequence[ConstraintSpec]) -> List[Dict[str, Any]]:
    return [
        {
            "name": constraint["name"],
            "operator": constraint.get("operator"),
            "bound": constraint.get("bound"),
            "source": constraint.get("source", "generic"),
        }
        for constraint in constraints
    ]


def _empty_result(
    *,
    criteria_count: int,
    termination_reason: str,
    error: str,
    weights: Sequence[float] | None = None,
    init_mode: str | None = None,
    criteria: Sequence[CriterionSpec] | None = None,
    constraints: Sequence[ConstraintSpec] | None = None,
    weights_source: str | None = None,
    allow_empty: bool = False,
) -> Dict[str, Any]:
    criteria = list(criteria or [])
    constraints = list(constraints or [])
    criterion_names = [criterion["name"] for criterion in criteria]
    criterion_directions = [criterion["direction_label"] for criterion in criteria]
    zero_by_criterion = {name: 0.0 for name in criterion_names}

    return {
        "best_mask": tuple(),
        "best_items": [],
        "best_raw_scores": [0.0] * criteria_count,
        "best_normalized_scores": [0.0] * criteria_count,
        "best_raw_scores_by_criterion": zero_by_criterion,
        "best_directed_scores_by_criterion": zero_by_criterion,
        "best_normalized_scores_by_criterion": zero_by_criterion,
        "best_agg": None,
        "generations_ran": 0,
        "termination_reason": termination_reason,
        "history": [],
        "error": error,
        "feasible_initial_count": 0,
        "weights": list(weights) if weights is not None else None,
        "weights_source": weights_source,
        "normalization_mins": None,
        "normalization_maxs": None,
        "fitness_scale": "normalized_0_1",
        "init_mode": init_mode,
        "initialization_source": init_mode,
        "mech_used_for_initialization": False,
        "mech_seed_count": 0,
        "random_seed_count": 0,
        "heuristic_seed_count": 0,
        "criterion_names": criterion_names,
        "criterion_directions": criterion_directions,
        "criteria_metadata": _criteria_metadata(criteria),
        "criteria": [],
        "constraints_metadata": _constraint_metadata(constraints),
        "constraints": [],
        "top_solutions": [],
        "top_solutions_source": "unique_feasible_archive",
        "unique_feasible_solutions_seen": 0,
        "quality_check": {
            "status": "skipped",
            "method": "exhaustive_enumeration",
            "available": False,
            "reason": error,
            "item_count": 0,
        },
        "allow_empty": allow_empty,
    }


def _evaluate_constraints(
    subset: List[Item],
    constraints: Sequence[ConstraintSpec],
) -> List[Dict[str, Any]]:
    details = []
    for constraint in constraints:
        name = constraint["name"]
        predicate = constraint.get("predicate")
        func = constraint.get("func")
        op = constraint.get("operator")
        bound = constraint.get("bound")

        try:
            if predicate is not None and func is None:
                passed = bool(predicate(subset))
                value = None
            elif predicate is not None and bound is None:
                passed = bool(predicate(subset))
                value = None
            elif func is not None and op is not None and bound is not None:
                value = _as_finite_float(func(subset), f"constraint {name} value")
                passed = bool(_OPERATORS[op](value, bound))
                if predicate is not None:
                    passed = passed and bool(predicate(subset))
            elif func is not None:
                raw_value = func(subset)
                passed = bool(raw_value)
                value = raw_value if isinstance(raw_value, (bool, int, float, str)) else None
            else:
                passed = True
                value = None
        except Exception as exc:  # noqa: BLE001 - нужна безопасная диагностика ограничения
            raise _ConstraintEvaluationError(
                f"constraint '{name}' returned an invalid value"
            ) from exc

        details.append(
            {
                "name": name,
                "value": value,
                "operator": op,
                "bound": bound,
                "passed": bool(passed),
                "source": constraint.get("source", "generic"),
            }
        )
    return details


def _mech_compatible_constraint(
    constraints: Sequence[ConstraintSpec],
) -> Tuple[Callable[[List[Item]], Any], float] | None:
    """Возвращает единственное ограничение, совместимое со старым mech, если оно есть."""
    if len(constraints) != 1:
        return None
    constraint = constraints[0]
    if constraint.get("func") is None:
        return None
    if constraint.get("operator") != "<=":
        return None
    if constraint.get("bound") is None:
        return None
    return constraint["func"], float(constraint["bound"])


def run_ga_mvp(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Запускает универсальную MVP-версию генетического алгоритма.

    Новый контракт:
    - ``criteria``: список произвольных критериев с ``name``, ``func`` и
      ``direction``;
    - ``constraints``: список произвольных ограничений с ``name``, ``func``,
      ``operator`` и ``bound`` или predicate-функцией;
    - ``weights`` или ``pairwise_matrix`` для взвешивания критериев.

    Старые поля результата и старые входные параметры сохранены для обратной
    совместимости.
    """
    validated = _validate_params(params)
    items: List[Item] = validated["items"]
    criteria: List[CriterionSpec] = validated["criteria"]
    constraints: List[ConstraintSpec] = validated["constraints"]
    pairwise_matrix: np.ndarray | None = validated["pairwise_matrix"]
    weights: np.ndarray = validated["weights"]
    weights_source: str = validated["weights_source"]

    pop_size: int = validated["pop_size"]
    generations: int = validated["generations"]
    mutation_rate: float = validated["mutation_rate"]
    elite: int = validated["elite"]
    seed = validated["seed"]
    stagnation_limit: int = validated["stagnation_limit"]
    max_random_attempts: int = validated["max_random_attempts"]
    top_solutions_limit: int = validated["top_solutions_limit"]
    quality_check_enabled: bool = validated["quality_check_enabled"]
    quality_check_max_items: int = validated["quality_check_max_items"]
    init_mode: str = validated["init_mode"]
    allow_empty: bool = validated["allow_empty"]

    criteria_count = len(criteria)
    criterion_names = [criterion["name"] for criterion in criteria]
    criterion_directions = [criterion["direction_label"] for criterion in criteria]
    criteria_funcs: List[Callable[[List[Item]], Any]] = [criterion["func"] for criterion in criteria]
    optimize_directions: List[int] = [criterion["direction"] for criterion in criteria]

    logger.info(
        "Запуск GA MVP: items=%s, criteria=%s, constraints=%s, pop_size=%s, generations=%s, mutation_rate=%s, init_mode=%s",
        len(items),
        len(criteria),
        len(constraints),
        pop_size,
        generations,
        mutation_rate,
        init_mode,
    )

    if not items:
        return _empty_result(
            criteria_count=criteria_count,
            termination_reason=TERMINATION_NO_FEASIBLE,
            error="GA не запущен: список item-ов пуст.",
            weights=weights.tolist(),
            init_mode=init_mode,
            criteria=criteria,
            constraints=constraints,
            weights_source=weights_source,
            allow_empty=allow_empty,
        )

    rnd = random.Random(seed)
    logger.debug("GA weights source=%s weights=%s", weights_source, weights.tolist())

    def is_feasible(mask: Sequence[int]) -> bool:
        normalized_mask = mask_to_tuple(mask)
        if not allow_empty and not any(normalized_mask):
            return False
        subset = subset_from_mask(items, normalized_mask)
        try:
            return all(
                detail["passed"] for detail in _evaluate_constraints(subset, constraints)
            )
        except _ConstraintEvaluationError:
            logger.exception("Некорректное значение ограничения")
            return False

    def constraint_details_for_mask(mask: Sequence[int]) -> List[Dict[str, Any]]:
        subset = subset_from_mask(items, mask)
        return _evaluate_constraints(subset, constraints)

    feasible_solution_archive: List[Mask] = []
    feasible_solution_archive_seen: set[Mask] = set()

    def register_feasible_mask(mask: Sequence[int]) -> None:
        """Сохраняет уникальное допустимое решение, которое реально видел GA.

        Архив нужен для честного ``top_solutions``: раньше итоговый top-N
        строился только из лучшего решения и финальной популяции, поэтому
        сильные кандидаты из ранних поколений могли исчезать из отчёта.
        """
        normalized_mask = mask_to_tuple(mask)
        if normalized_mask in feasible_solution_archive_seen:
            return
        if not is_feasible(normalized_mask):
            return
        feasible_solution_archive_seen.add(normalized_mask)
        feasible_solution_archive.append(normalized_mask)

    def register_feasible_masks(masks: Sequence[Sequence[int]]) -> None:
        for mask in masks:
            register_feasible_mask(mask)

    def add_unique_mask(target: List[Mask], mask: Sequence[int]) -> None:
        normalized_mask = mask_to_tuple(mask)
        if normalized_mask not in target and is_feasible(normalized_mask):
            target.append(normalized_mask)

    def raw_scores_for_mask(mask: Sequence[int]) -> np.ndarray:
        subset = subset_from_mask(items, mask)
        raw = []
        for criterion in criteria:
            value = criterion["func"](subset)
            try:
                scalar = _scalarize_criterion_value(value)
            except Exception:
                logger.exception(
                    "Не удалось привести значение критерия %s к float, будет использован 0.0",
                    criterion["name"],
                )
                scalar = 0.0
            raw.append(criterion["direction"] * scalar)
        return np.array(raw, dtype=float)

    def original_scores_for_mask(mask: Sequence[int]) -> np.ndarray:
        subset = subset_from_mask(items, mask)
        raw = []
        for criterion in criteria:
            value = criterion["func"](subset)
            try:
                scalar = _scalarize_criterion_value(value)
            except Exception:
                logger.exception(
                    "Не удалось привести значение критерия %s к float, будет использован 0.0",
                    criterion["name"],
                )
                scalar = 0.0
            raw.append(scalar)
        return np.array(raw, dtype=float)

    def scores_by_criterion(values: Sequence[float]) -> Dict[str, float]:
        return {name: float(value) for name, value in zip(criterion_names, values)}

    def random_mask() -> Mask | None:
        for _attempt in range(max_random_attempts):
            mask = tuple(rnd.choice((0, 1)) for _item in items)
            if is_feasible(mask):
                return mask
        return None

    def add_singleton_masks(target: List[Mask]) -> int:
        before = len(target)
        for index in range(len(items)):
            singleton = tuple(1 if item_index == index else 0 for item_index in range(len(items)))
            add_unique_mask(target, singleton)
        return len(target) - before

    def add_weighted_greedy_masks(target: List[Mask]) -> int:
        before = len(target)
        ranked: List[Tuple[float, int]] = []
        for index in range(len(items)):
            singleton = tuple(1 if item_index == index else 0 for item_index in range(len(items)))
            if not is_feasible(singleton):
                continue
            score = float(np.dot(raw_scores_for_mask(singleton), weights))
            ranked.append((score, index))

        for reverse in (True, False):
            current = [0] * len(items)
            for _score, index in sorted(ranked, reverse=reverse):
                candidate = current.copy()
                candidate[index] = 1
                if is_feasible(candidate):
                    current = candidate
                    add_unique_mask(target, current)
        return len(target) - before

    def add_random_masks(target: List[Mask]) -> int:
        before = len(target)
        attempts = 0
        while len(target) < pop_size and attempts < max_random_attempts:
            mask = random_mask()
            attempts += 1
            if mask is not None:
                add_unique_mask(target, mask)
            elif target:
                break
        return len(target) - before

    init_pop: List[Mask] = []
    mech_out: Dict[str, Any] | None = None
    mech_seed_count = 0
    heuristic_seed_count = 0
    random_seed_count = 0
    mech_used_for_initialization = False

    mech_constraint = _mech_compatible_constraint(constraints)
    if init_mode == INIT_MODE_PARETO_SEEDED and mech_constraint is not None and pairwise_matrix is not None:
        constraint_func, max_constraint = mech_constraint
        mech_params = {
            "items": items,
            "max_constraint": max_constraint,
            "constraint_func": constraint_func,
            "criteria_funcs": criteria_funcs,
            "optimize_directions": optimize_directions,
            "pairwise_matrix": pairwise_matrix,
            "exclude_empty": not allow_empty,
        }
        mech_out = mech_run(mech_params)
        mech_used_for_initialization = True
        for solution in mech_out.get("pareto", []):
            before = len(init_pop)
            add_unique_mask(
                init_pop,
                make_mask_from_mech_itemlist(items, solution.get("items", [])),
            )
            if len(init_pop) > before:
                mech_seed_count += 1
        heuristic_seed_count += add_singleton_masks(init_pop)
        random_seed_count += add_random_masks(init_pop)
    elif init_mode == INIT_MODE_PARETO_SEEDED:
        logger.info(
            "pareto_seeded_init запрошен, но ограничения/веса несовместимы с KnapsackMechanic; используется heuristic/random fallback"
        )
        heuristic_seed_count += add_singleton_masks(init_pop)
        heuristic_seed_count += add_weighted_greedy_masks(init_pop)
        random_seed_count += add_random_masks(init_pop)
    elif init_mode == INIT_MODE_HEURISTIC:
        heuristic_seed_count += add_singleton_masks(init_pop)
        heuristic_seed_count += add_weighted_greedy_masks(init_pop)
        random_seed_count += add_random_masks(init_pop)
    else:
        random_seed_count += add_random_masks(init_pop)
        # Страховка от ложного "no feasible" при малом числе random-попыток:
        # режим остаётся случайным, но если случайный старт не нашёл допустимых масок,
        # проверяем одиночные элементы без полного перебора подмножеств.
        if not init_pop:
            heuristic_seed_count += add_singleton_masks(init_pop)

    if not init_pop:
        return _empty_result(
            criteria_count=criteria_count,
            termination_reason=TERMINATION_NO_FEASIBLE,
            error="GA не запущен: не найдено ни одного допустимого решения.",
            weights=weights.tolist(),
            init_mode=init_mode,
            criteria=criteria,
            constraints=constraints,
            weights_source=weights_source,
            allow_empty=allow_empty,
        )

    register_feasible_masks(init_pop)

    while len(init_pop) < pop_size:
        init_pop.append(rnd.choice(init_pop))

    population: List[Mask] = init_pop[:pop_size]
    register_feasible_masks(population)
    feasible_initial_count = len({mask for mask in population if is_feasible(mask)})
    logger.info(
        "Стартовая популяция GA подготовлена: %s решений, уникальных допустимых=%s, init_mode=%s",
        len(population),
        feasible_initial_count,
        init_mode,
    )

    if (
        mech_out is not None
        and mech_out.get("mins") is not None
        and mech_out.get("maxs") is not None
    ):
        normalization_mins = np.array(mech_out["mins"], dtype=float)
        normalization_maxs = np.array(mech_out["maxs"], dtype=float)
    else:
        normalization_masks = list(dict.fromkeys(init_pop))
        singleton_masks: List[Mask] = []
        add_singleton_masks(singleton_masks)
        normalization_masks = list(dict.fromkeys(normalization_masks + singleton_masks))
        initial_raw_scores = [raw_scores_for_mask(mask) for mask in normalization_masks]
        normalization_mins, normalization_maxs = _normalization_bounds_from_scores(
            initial_raw_scores,
            criteria_count,
        )

    if normalization_mins.shape != (criteria_count,) or normalization_maxs.shape != (
        criteria_count,
    ):
        raise ValueError("normalization bounds must match number of criteria")
    if not np.all(np.isfinite(normalization_mins)) or not np.all(np.isfinite(normalization_maxs)):
        raise ValueError("normalization bounds must contain only finite numbers")

    logger.debug(
        "GA normalization bounds: mins=%s, maxs=%s",
        normalization_mins.tolist(),
        normalization_maxs.tolist(),
    )

    def normalized_scores_from_raw(raw_vec: np.ndarray) -> np.ndarray:
        return _normalize_raw_scores(raw_vec, normalization_mins, normalization_maxs)

    def agg_fitness_from_raw(raw_vec: np.ndarray) -> float:
        normalized_vec = normalized_scores_from_raw(raw_vec)
        return float(np.dot(normalized_vec, weights))

    def one_point_crossover(left: Mask, right: Mask) -> Tuple[Mask, Mask]:
        if len(left) <= 1:
            return left, right
        point = rnd.randrange(1, len(left))
        child_a = left[:point] + right[point:]
        child_b = right[:point] + left[point:]
        return tuple(child_a), tuple(child_b)

    def repair(mask: Sequence[int]) -> Mask:
        mutable_mask = [int(bool(bit)) for bit in mask]
        while allow_empty or any(mutable_mask):
            if is_feasible(mutable_mask):
                return tuple(mutable_mask)
            if not any(mutable_mask):
                break
            ones = [idx for idx, bit in enumerate(mutable_mask) if bit]
            mutable_mask[rnd.choice(ones)] = 0

        for index in rnd.sample(range(len(mutable_mask)), len(mutable_mask)):
            candidate = [0] * len(mutable_mask)
            candidate[index] = 1
            if is_feasible(candidate):
                return tuple(candidate)

        return rnd.choice(init_pop)

    def mutate(mask: Mask) -> Mask:
        mutable_mask = list(mask)
        for index in range(len(mutable_mask)):
            if rnd.random() < mutation_rate:
                mutable_mask[index] = 1 - mutable_mask[index]
        return repair(mutable_mask)

    best: Mask | None = None
    best_score = -math.inf
    stagnation = 0
    gen = 0
    termination_reason = TERMINATION_MAX_GENERATIONS
    history: List[Dict[str, Any]] = []

    for gen in range(1, generations + 1):
        population = [repair(mask) for mask in population]
        register_feasible_masks(population)
        raw_list = [raw_scores_for_mask(mask) for mask in population]
        agg_list = [agg_fitness_from_raw(raw) for raw in raw_list]

        gen_best_idx = int(np.argmax(agg_list))
        gen_best_mask = population[gen_best_idx]
        gen_best_score = agg_list[gen_best_idx]
        gen_best_normalized = normalized_scores_from_raw(raw_list[gen_best_idx])
        feasible_count = sum(1 for mask in population if is_feasible(mask))
        mean_score = float(np.mean(agg_list)) if agg_list else None

        if gen == 1 or gen_best_score > best_score + 1e-12:
            best = gen_best_mask
            best_score = gen_best_score
            stagnation = 0
        else:
            stagnation += 1

        history.append(
            {
                "generation": gen,
                "best_score": float(best_score),
                "generation_best_score": float(gen_best_score),
                "mean_score": mean_score,
                "feasible_count": feasible_count,
                "best_mask": mask_to_tuple(best),
                "generation_best_normalized_scores": gen_best_normalized.tolist(),
                "generation_best_normalized_scores_by_criterion": scores_by_criterion(
                    gen_best_normalized
                ),
            }
        )

        if gen % max(1, generations // 10) == 0 or gen == 1 or gen == generations:
            logger.debug(
                "GA progress gen=%s/%s best_score=%.6f gen_best=%.6f stagnation=%s feasible=%s",
                gen,
                generations,
                best_score,
                gen_best_score,
                stagnation,
                feasible_count,
            )

        if stagnation >= stagnation_limit:
            termination_reason = TERMINATION_STAGNATION
            logger.info("GA ранне остановлен по stagnation_limit=%s", stagnation_limit)
            break

        sorted_idx = sorted(range(len(population)), key=lambda idx: agg_list[idx], reverse=True)
        next_pop = [population[idx] for idx in sorted_idx[:elite]]

        def tournament_select(
            current_population: List[Mask] = population,
            current_agg_list: List[float] = agg_list,
        ) -> Mask:
            tournament_size = min(3, len(current_population))
            candidates = rnd.sample(range(len(current_population)), tournament_size)
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

        register_feasible_masks(next_pop)
        population = next_pop

    def solution_payload(mask: Sequence[int], *, rank: int | None = None) -> Dict[str, Any]:
        solution_mask = mask_to_tuple(mask)
        subset = subset_from_mask(items, solution_mask)
        directed_raw = raw_scores_for_mask(solution_mask)
        original_raw = original_scores_for_mask(solution_mask)
        normalized = normalized_scores_from_raw(directed_raw)
        aggregate_score = agg_fitness_from_raw(directed_raw)
        solution_criteria = []
        for index, criterion in enumerate(criteria):
            solution_criteria.append(
                {
                    "name": criterion["name"],
                    "direction": criterion["direction_label"],
                    "raw_value": float(original_raw[index]),
                    "directed_value": float(directed_raw[index]),
                    "normalized_value": float(normalized[index]),
                    "weight": float(weights[index]),
                }
            )
        payload = {
            "mask": solution_mask,
            "items": [item.properties for item in subset],
            "raw_scores": directed_raw.tolist(),
            "normalized_scores": normalized.tolist(),
            "raw_scores_by_criterion": scores_by_criterion(original_raw),
            "directed_scores_by_criterion": scores_by_criterion(directed_raw),
            "normalized_scores_by_criterion": scores_by_criterion(normalized),
            "agg": float(aggregate_score),
            "criteria": solution_criteria,
            "constraints": constraint_details_for_mask(solution_mask),
        }
        if rank is not None:
            payload["rank"] = int(rank)
        return payload

    if best is not None:
        register_feasible_mask(best)
    register_feasible_masks(population)

    candidate_masks = list(feasible_solution_archive)
    scored_candidates: List[Tuple[float, Mask]] = []
    for candidate_mask in candidate_masks:
        normalized_mask = mask_to_tuple(candidate_mask)
        scored_candidates.append((agg_fitness_from_raw(raw_scores_for_mask(normalized_mask)), normalized_mask))
    scored_candidates.sort(key=lambda item: item[0], reverse=True)
    top_solutions = [
        solution_payload(mask, rank=rank)
        for rank, (_score, mask) in enumerate(scored_candidates[:top_solutions_limit], start=1)
    ]

    def exact_quality_check() -> Dict[str, Any]:
        """Проверяет GA на малом наборе полным перебором.

        Это диагностический режим: он не влияет на выбор решения, а только
        сравнивает найденный GA результат с точным эталоном, если число item-ов
        достаточно маленькое.
        """
        item_count = len(items)
        base_report: Dict[str, Any] = {
            "method": "exhaustive_enumeration",
            "enabled": quality_check_enabled,
            "available": False,
            "item_count": item_count,
            "max_items": quality_check_max_items,
        }
        if not quality_check_enabled:
            return {
                **base_report,
                "status": "skipped",
                "reason": "disabled",
            }
        if item_count > quality_check_max_items:
            return {
                **base_report,
                "status": "skipped",
                "reason": "too_many_items",
                "message": (
                    "Полный перебор пропущен: число item-ов превышает "
                    "quality_check_max_items."
                ),
            }

        total_masks = 1 << item_count
        exact_candidates: List[Tuple[float, Mask]] = []
        for mask_number in range(total_masks):
            mask = tuple((mask_number >> index) & 1 for index in range(item_count))
            if not is_feasible(mask):
                continue
            exact_candidates.append((agg_fitness_from_raw(raw_scores_for_mask(mask)), mask))

        exact_candidates.sort(key=lambda item: item[0], reverse=True)
        if not exact_candidates:
            return {
                **base_report,
                "status": "warning",
                "available": True,
                "reason": "no_feasible_solution",
                "total_masks_checked": total_masks,
                "feasible_count": 0,
            }

        exact_best_score, exact_best_mask = exact_candidates[0]
        ga_best_mask = mask_to_tuple(best)
        ga_best_score = float(best_score) if best is not None and math.isfinite(best_score) else None
        exact_rank_by_mask = {candidate_mask: rank for rank, (_score, candidate_mask) in enumerate(exact_candidates, start=1)}
        ga_best_exact_rank = exact_rank_by_mask.get(ga_best_mask)
        ga_best_mask_matches_exact_best = bool(ga_best_mask == exact_best_mask)
        ga_best_matches_exact_best = bool(
            ga_best_score is not None
            and math.isclose(float(ga_best_score), float(exact_best_score), rel_tol=0.0, abs_tol=1e-12)
        )

        comparison_limit = min(
            top_solutions_limit if top_solutions_limit > 0 else 10,
            len(exact_candidates),
        )
        exact_top_masks = [mask for _score, mask in exact_candidates[:comparison_limit]]
        ga_top_masks = [mask_to_tuple(solution.get("mask")) for solution in top_solutions[:comparison_limit]]
        overlap_count = len(set(exact_top_masks).intersection(ga_top_masks))
        overlap_ratio = overlap_count / comparison_limit if comparison_limit else 0.0
        ga_top_matches_exact_top = overlap_count == comparison_limit

        warnings: List[str] = []
        if not ga_best_matches_exact_best:
            warnings.append(
                "GA best не совпал с точным оптимумом на малом наборе; "
                "проверьте параметры ГА или ограничения."
            )
        if not ga_top_matches_exact_top:
            warnings.append(
                "GA top_solutions не полностью совпадает с exact top-N; "
                "пул кандидатов может быть неполным."
            )

        return {
            **base_report,
            "status": "ok" if not warnings else "warning",
            "available": True,
            "total_masks_checked": total_masks,
            "feasible_count": len(exact_candidates),
            "ga_best_matches_exact_best": ga_best_matches_exact_best,
            "ga_best_mask_matches_exact_best": ga_best_mask_matches_exact_best,
            "ga_best_exact_rank": ga_best_exact_rank,
            "ga_best_score": ga_best_score,
            "exact_best_score": float(exact_best_score),
            "score_delta": (
                float(ga_best_score - exact_best_score)
                if ga_best_score is not None
                else None
            ),
            "ga_best_mask": ga_best_mask,
            "exact_best_mask": exact_best_mask,
            "exact_best_solution": solution_payload(exact_best_mask, rank=1),
            "top_solutions_limit": comparison_limit,
            "ga_top_masks": ga_top_masks,
            "exact_top_masks": exact_top_masks,
            "top_overlap_count": overlap_count,
            "top_overlap_ratio": overlap_ratio,
            "ga_top_matches_exact_top": ga_top_matches_exact_top,
            "warnings": warnings,
        }

    quality_check = exact_quality_check()

    best_solution = solution_payload(best) if best is not None else None
    best_subset = subset_from_mask(items, best) if best is not None else []
    best_raw = (
        np.array(best_solution["raw_scores"], dtype=float)
        if best_solution is not None
        else np.zeros(criteria_count)
    )
    best_original = (
        np.array([best_solution["raw_scores_by_criterion"][name] for name in criterion_names], dtype=float)
        if best_solution is not None
        else np.zeros(criteria_count)
    )
    best_normalized = (
        np.array(best_solution["normalized_scores"], dtype=float)
        if best_solution is not None
        else normalized_scores_from_raw(best_raw)
    )
    best_agg = best_solution["agg"] if best_solution is not None else None
    best_constraints = best_solution["constraints"] if best_solution is not None else []

    criteria_details = best_solution["criteria"] if best_solution is not None else []

    result = {
        "best_mask": mask_to_tuple(best),
        "best_items": [item.properties for item in best_subset],
        # Старое поле сохраняет прежнюю семантику: направленные значения,
        # где min-критерии умножены на -1.
        "best_raw_scores": best_raw.tolist(),
        "best_normalized_scores": best_normalized.tolist(),
        "best_raw_scores_by_criterion": scores_by_criterion(best_original),
        "best_directed_scores_by_criterion": scores_by_criterion(best_raw),
        "best_normalized_scores_by_criterion": scores_by_criterion(best_normalized),
        "best_agg": float(best_agg) if best_agg is not None else None,
        "generations_ran": gen,
        "termination_reason": termination_reason,
        "history": history,
        "error": None,
        "feasible_initial_count": feasible_initial_count,
        "weights": weights.tolist(),
        "weights_source": weights_source,
        "normalization_mins": normalization_mins.tolist(),
        "normalization_maxs": normalization_maxs.tolist(),
        "fitness_scale": "normalized_0_1",
        "init_mode": init_mode,
        "initialization_source": init_mode,
        "mech_used_for_initialization": mech_used_for_initialization,
        "mech_seed_count": mech_seed_count,
        "random_seed_count": random_seed_count,
        "heuristic_seed_count": heuristic_seed_count,
        "criterion_names": criterion_names,
        "criterion_directions": criterion_directions,
        "criteria_metadata": _criteria_metadata(criteria),
        "criteria": criteria_details,
        "constraints_metadata": _constraint_metadata(constraints),
        "constraints": best_constraints,
        "top_solutions": top_solutions,
        "top_solutions_source": "unique_feasible_archive",
        "unique_feasible_solutions_seen": len(feasible_solution_archive),
        "quality_check": quality_check,
        "allow_empty": allow_empty,
    }
    logger.info(
        "GA завершён. Лучший агрегированный score=%s, причина остановки=%s",
        result["best_agg"],
        result["termination_reason"],
    )
    return result
