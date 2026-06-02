from __future__ import annotations

import logging
import math
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from domain import AnalysisResult
from .aggregation import aggregate_configuration, filter_hard_constraints
from .math_utils import (
    ahp_priority_and_consistency,
    build_pairwise_matrix_from_scores,
    geometric_mean_matrix,
)

logger = logging.getLogger(__name__)

def _normalize_criterion_directions(
    soft_criteria: Sequence[str],
    criterion_directions: Mapping[str, str] | None,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for criterion in soft_criteria:
        raw = str((criterion_directions or {}).get(criterion, "max")).strip().lower()
        result[criterion] = "min" if raw in {"min", "minimize", "minimise", "-", "-1"} else "max"
    return result


def _directed_values_for_ahp(values: Sequence[float], direction: str) -> list[float]:
    """Return positive values where larger always means better for AHP ratios.

    The legacy AHP builder constructs pairwise matrices as ``s_i / s_j`` and
    therefore cannot receive negative numbers.  For minimization criteria we
    convert raw values to a simple 1..9 utility scale: the cheapest/least-power
    alternative gets 9, the worst gets 1, equal values get neutral 1.
    """

    numeric_values = [float(value) for value in values]
    if not numeric_values:
        return []
    if direction != "min":
        return [value if value > 0 else 1e-9 for value in numeric_values]
    minimum = min(numeric_values)
    maximum = max(numeric_values)
    if abs(maximum - minimum) < 1e-12:
        return [1.0 for _value in numeric_values]
    return [1.0 + 8.0 * ((maximum - value) / (maximum - minimum)) for value in numeric_values]


def run_ahp_pipeline(
    configurations: Sequence[Dict[str, Any]],
    soft_criteria: Sequence[str],
    experts_criteria_matrices: Sequence[np.ndarray],
    constraints: Dict[str, Any],
    saaty_cap: bool = True,
    top_pct: float = 0.10,
    criterion_directions: Mapping[str, str] | None = None,
) -> Dict[str, Any]:
    """
    configurations: список исходных конфигураций (см. load_configurations_from_json)
    soft_criteria: список имён критериев, которые используются в AHP (числовые поля в агрегатах)
    experts_criteria_matrices: список матриц (square) от экспертов, каждой матрицей — попарные сравнения важности soft_criteria
        размер матрицы должен быть len(soft_criteria)
    constraints: словарь с жёсткими ограничениями (см. filter_hard_constraints)
    saaty_cap: капать ли матрицы альтернатив в шкалу Саати
    criterion_directions: направление критериев из общего профиля; для ``min``
        значения альтернатив переводятся в положительную utility-шкалу, чтобы
        меньшая стоимость/мощность не выглядела хуже только из-за большей
        численной величины. Если не передано, сохраняется старое поведение.

    Возвращает словарь с полным отчётом, пригодный для вставки в документ.
    """
    logger.info(
        "Запуск AHP-пайплайна: конфигураций=%s, soft_criteria=%s",
        len(configurations),
        list(soft_criteria),
    )

    aggregates = [aggregate_configuration(c) for c in configurations]
    passed, removed = filter_hard_constraints(aggregates, constraints)
    logger.info(
        "AHP-фильтрация завершена: прошло=%s, отклонено=%s",
        len(passed),
        len(removed),
    )

    report = {
        "total_input": len(aggregates),
        "passed_count": len(passed),
        "removed_count": len(removed),
        "removed_details": removed,
        "passed": passed,
    }

    if len(passed) == 0:
        logger.warning("AHP-пайплайн завершён без допустимых конфигураций")
        report["note"] = "No feasible configurations after hard constraints."
        return report

    if len(passed) < 5:
        logger.warning("После фильтрации осталось меньше 5 альтернатив: %s", len(passed))
        report["warning"] = (
            "Fewer than 5 feasible alternatives remain after filtering. TЗ требует >=5."
        )

    if len(experts_criteria_matrices) < 2:
        raise ValueError(
            "Требуется минимум 2 матрицы экспертов для сравнения критериев (soft criteria)"
        )
    for matrix in experts_criteria_matrices:
        matrix = np.array(matrix, dtype=float)
        if matrix.shape[0] != matrix.shape[1] or matrix.shape[0] != len(soft_criteria):
            raise ValueError(
                "Каждая матрица эксперта должна быть квадратной и размером == числу soft_criteria"
            )

    crit_agg = geometric_mean_matrix(experts_criteria_matrices)
    crit_weights, crit_lam, crit_ci, crit_cr = ahp_priority_and_consistency(crit_agg)
    logger.debug("AHP-веса критериев: %s", crit_weights.tolist())

    directions = _normalize_criterion_directions(soft_criteria, criterion_directions)
    report["criteria"] = {
        "names": list(soft_criteria),
        "directions": directions,
        "aggregated_matrix": crit_agg.tolist(),
        "weights": crit_weights.tolist(),
        "lambda_max": crit_lam,
        "CI": crit_ci,
        "CR": crit_cr,
    }

    n_alt = len(passed)
    alt_ids = [a["id"] for a in passed]
    alt_scores_matrix = np.zeros((len(soft_criteria), n_alt), dtype=float)
    directed_scores_matrix = np.zeros((len(soft_criteria), n_alt), dtype=float)
    alt_pairwise_matrices = []
    alt_weights_per_criterion = []
    alt_consistency_per_criterion = []

    for ci, crit in enumerate(soft_criteria):
        values = []
        for aggregate in passed:
            value = aggregate.get(crit, None)
            values.append(0.0 if value is None else float(value))
        directed_values = _directed_values_for_ahp(values, directions.get(crit, "max"))
        alt_scores_matrix[ci, :] = np.array(values, dtype=float)
        directed_scores_matrix[ci, :] = np.array(directed_values, dtype=float)
        pairwise = build_pairwise_matrix_from_scores(directed_values, saaty_cap=saaty_cap)
        alt_pairwise_matrices.append(pairwise)
        weights, lam, ci_value, cr_value = ahp_priority_and_consistency(pairwise)
        alt_weights_per_criterion.append(weights.tolist())
        alt_consistency_per_criterion.append({"lambda": lam, "CI": ci_value, "CR": cr_value})
        logger.debug("AHP-альтернативы по критерию %s: weights=%s", crit, weights.tolist())

    report["alternatives"] = {
        "ids": alt_ids,
        "scores": alt_scores_matrix.tolist(),
        "directed_scores": directed_scores_matrix.tolist(),
        "pairwise_matrices": [pairwise.tolist() for pairwise in alt_pairwise_matrices],
        "weights_per_criterion": alt_weights_per_criterion,
        "consistency_per_criterion": alt_consistency_per_criterion,
    }

    alt_w_matrix = np.array(alt_weights_per_criterion, dtype=float)
    if alt_w_matrix.shape[0] != len(soft_criteria) or alt_w_matrix.shape[1] != n_alt:
        alt_w_matrix = alt_w_matrix.reshape(len(soft_criteria), n_alt)

    final_scores = (np.array(crit_weights, dtype=float) @ alt_w_matrix).flatten()
    if final_scores.sum() > 0:
        final_scores_norm = final_scores / float(final_scores.sum())
    else:
        final_scores_norm = np.ones_like(final_scores) / float(len(final_scores))

    ranked = sorted(
        [(aid, float(score)) for aid, score in zip(alt_ids, final_scores_norm)], key=lambda x: -x[1]
    )
    logger.info("AHP-ранжирование завершено. Победитель: %s", ranked[0][0])

    result = AnalysisResult(
        case_name="AHP configuration selection",
        ranking=[{"id": aid, "name": aid, "score": score} for aid, score in ranked],
        weights={aid: float(score) for aid, score in ranked},
        winner_id=ranked[0][0],
        winner_name=ranked[0][0],
        metadata={},
    )

    report["final"] = {
        "raw_final_scores": final_scores.tolist(),
        "normalized_final_scores": final_scores_norm.tolist(),
        "ranking": ranked,
        "ranking_models": result.ranking,
        "winner_id": result.winner_id,
        "winner_name": result.winner_name,
    }

    if len(passed) <= 10:
        top_selected = [ranked_item[0] for ranked_item in ranked]
    else:
        top_n = max(1, math.ceil(len(passed) * top_pct))
        top_selected = [ranked_item[0] for ranked_item in ranked[:top_n]]

    report["selection"] = {
        "top_pct": top_pct,
        "selected_ids": top_selected,
        "selected_count": len(top_selected),
    }

    logger.info("AHP-пайплайн завершён. Выбрано альтернатив: %s", len(top_selected))
    return report
