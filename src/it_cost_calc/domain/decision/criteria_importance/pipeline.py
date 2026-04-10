from __future__ import annotations

import copy
import logging
from fractions import Fraction
from math import gcd, log
from typing import Any, Dict, List, Tuple

from it_cost_calc.domain import Alternative, AnalysisResult, RelationOperator
from .case_data import _criteria, _alternatives, _relations, validate_case

logger = logging.getLogger(__name__)


class DSU:
    def __init__(self, items: List[str]):
        self.parent = {item: item for item in items}

    def find(self, x: str) -> str:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _lcm(a: int, b: int) -> int:
    return abs(a * b) // gcd(a, b) if a and b else max(abs(a), abs(b), 1)


def dominance_relation(vec_a: List[float], vec_b: List[float]) -> str:
    a_ge_b = all(a >= b for a, b in zip(vec_a, vec_b))
    b_ge_a = all(b >= a for a, b in zip(vec_a, vec_b))
    a_gt_b = any(a > b for a, b in zip(vec_a, vec_b))
    b_gt_a = any(b > a for a, b in zip(vec_a, vec_b))

    if a_ge_b and a_gt_b:
        return "dominates"
    if b_ge_a and b_gt_a:
        return "dominated"
    return "incomparable"


def _build_importance_groups(case_data: Dict[str, Any]) -> Dict[str, Any]:
    criteria_ids = [criterion.id for criterion in _criteria(case_data)]
    dsu = DSU(criteria_ids)

    for rel in _relations(case_data):
        if rel.operator == RelationOperator.EQUAL:
            dsu.union(rel.left, rel.right)

    groups: Dict[str, List[str]] = {}
    for criterion_id in criteria_ids:
        root = dsu.find(criterion_id)
        groups.setdefault(root, []).append(criterion_id)

    edge_constraints: Dict[Tuple[str, str], float] = {}
    for rel in _relations(case_data):
        if rel.operator != RelationOperator.GREATER:
            continue
        left = dsu.find(rel.left)
        right = dsu.find(rel.right)
        if left == right:
            continue
        factor = float(rel.factor)
        key = (left, right)
        edge_constraints[key] = max(edge_constraints.get(key, 1.0), factor)

    group_ids = list(groups.keys())
    log_weights = {gid: 0.0 for gid in group_ids}

    for _ in range(len(group_ids) * max(len(edge_constraints), 1)):
        changed = False
        for (left, right), factor in edge_constraints.items():
            proposed = log_weights[right] + log(factor)
            if proposed > log_weights[left] + 1e-12:
                log_weights[left] = proposed
                changed = True
        if not changed:
            break

    # Проверка на противоречивый цикл
    for (left, right), factor in edge_constraints.items():
        if log_weights[left] + 1e-12 < log_weights[right] + log(factor):
            raise ValueError("Отношения важности критериев противоречивы: найден цикл ограничений")

    raw_weights = {gid: pow(2.718281828459045, value) for gid, value in log_weights.items()}
    min_weight = min(raw_weights.values()) if raw_weights else 1.0
    rel_weights = {gid: raw_weights[gid] / min_weight for gid in raw_weights}

    fractions = {gid: Fraction(rel_weights[gid]).limit_denominator(12) for gid in rel_weights}
    common_den = 1
    for frac in fractions.values():
        common_den = _lcm(common_den, frac.denominator)

    multiplicities = {}
    for gid, frac in fractions.items():
        multiplicities[gid] = max(1, int(frac * common_den))

    criterion_to_group = {criterion_id: dsu.find(criterion_id) for criterion_id in criteria_ids}
    criterion_multiplicity = {
        criterion_id: multiplicities[criterion_to_group[criterion_id]]
        for criterion_id in criteria_ids
    }

    criterion_order = sorted(criteria_ids, key=lambda cid: (-criterion_multiplicity[cid], cid))

    return {
        "groups": groups,
        "edges": [
            {"left_group": left, "right_group": right, "factor": factor}
            for (left, right), factor in edge_constraints.items()
        ],
        "group_multiplicity": multiplicities,
        "criterion_multiplicity": criterion_multiplicity,
        "criterion_order": criterion_order,
    }


def _base_vectors(case_data: Dict[str, Any], criterion_order: List[str]) -> Dict[str, List[float]]:
    base = {}
    for alt in _alternatives(case_data):
        alt_id = alt.id
        base[alt_id] = [
            float(case_data["scores"][alt_id][criterion_id]) for criterion_id in criterion_order
        ]
    return base


def _expanded_vectors(
    case_data: Dict[str, Any],
    criterion_order: List[str],
    criterion_multiplicity: Dict[str, int],
) -> Dict[str, List[float]]:
    expanded = {}
    for alt in _alternatives(case_data):
        alt_id = alt.id
        vector: List[float] = []
        for criterion_id in criterion_order:
            value = float(case_data["scores"][alt_id][criterion_id])
            vector.extend([value] * int(criterion_multiplicity[criterion_id]))
        expanded[alt_id] = vector
    return expanded


def _dominance_matrix(
    alternatives: List[Alternative], vectors: Dict[str, List[float]]
) -> Dict[str, Dict[str, str]]:
    matrix: Dict[str, Dict[str, str]] = {}
    for a in alternatives:
        aid = a.id
        matrix[aid] = {}
        for b in alternatives:
            bid = b.id
            if aid == bid:
                matrix[aid][bid] = "self"
            else:
                matrix[aid][bid] = dominance_relation(vectors[aid], vectors[bid])
    return matrix


def _weighted_sum(
    case_data: Dict[str, Any], criterion_multiplicity: Dict[str, int]
) -> Dict[str, float]:
    result = {}
    for alt in _alternatives(case_data):
        alt_id = alt.id
        total = 0.0
        for criterion_id, mult in criterion_multiplicity.items():
            total += float(case_data["scores"][alt_id][criterion_id]) * mult
        result[alt_id] = total
    return result


def _nondominated_set(
    alternatives: List[Alternative], dominance: Dict[str, Dict[str, str]]
) -> List[str]:
    alt_ids = [alternative.id for alternative in alternatives]
    result = []
    for aid in alt_ids:
        dominated = any(dominance[aid][other] == "dominated" for other in alt_ids if other != aid)
        if not dominated:
            result.append(aid)
    return result


def _explain(
    case_data: Dict[str, Any],
    analysis_model: Dict[str, Any],
    weighted_sum: Dict[str, float],
    nondominated: List[str],
    winner_id: str,
) -> str:
    alt_name = {alternative.id: alternative.name for alternative in _alternatives(case_data)}
    criteria_name = {criterion.id: criterion.name for criterion in _criteria(case_data)}
    top_criteria = sorted(
        analysis_model["criterion_multiplicity"].items(), key=lambda item: (-item[1], item[0])
    )[:5]
    top_criteria_text = ", ".join(
        f"{criterion_id} ({criteria_name[criterion_id]}, кратность {mult})"
        for criterion_id, mult in top_criteria
    )
    nondominated_text = ", ".join(alt_name[aid] for aid in nondominated)

    lines = [
        "После применения отношений важности критериев построена N-кратная модель оценки.",
        f"Наиболее важные критерии в модели: {top_criteria_text}.",
        f"Множество недоминируемых альтернатив: {nondominated_text}.",
        f"Финальный выбор: {alt_name[winner_id]} (суммарная взвешенная оценка {weighted_sum[winner_id]:.2f}).",
    ]

    winner_scores = case_data["scores"][winner_id]
    winners_best = sorted(
        winner_scores.items(),
        key=lambda item: (
            -(float(item[1]) * analysis_model["criterion_multiplicity"][item[0]]),
            item[0],
        ),
    )[:3]
    lines.append(
        "Наибольший вклад в итог дали: "
        + ", ".join(f"{criterion_id}={float(value):g}" for criterion_id, value in winners_best)
        + "."
    )
    return "\n".join(lines)


def run_importance_pipeline(case_data: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("Запуск анализа важности критериев")
    validate_case(case_data)
    logger.debug(
        "Кейс валидации пройден: критериев=%s, альтернатив=%s, отношений=%s",
        len(case_data.get("criteria", [])),
        len(case_data.get("alternatives", [])),
        len(case_data.get("relations", [])),
    )
    case_copy = copy.deepcopy(case_data)
    alternatives = _alternatives(case_copy)

    raw_order = [criterion.id for criterion in _criteria(case_copy)]
    raw_vectors = _base_vectors(case_copy, raw_order)
    raw_dominance = _dominance_matrix(alternatives, raw_vectors)
    raw_nondominated = _nondominated_set(alternatives, raw_dominance)

    analysis_model = _build_importance_groups(case_copy)
    logger.debug("Построены группы важности: %s", analysis_model.get("groups", {}))
    expanded_vectors = _expanded_vectors(
        case_copy,
        analysis_model["criterion_order"],
        analysis_model["criterion_multiplicity"],
    )
    final_dominance = _dominance_matrix(alternatives, expanded_vectors)
    final_nondominated = _nondominated_set(alternatives, final_dominance)
    weighted_sum = _weighted_sum(case_copy, analysis_model["criterion_multiplicity"])

    ranked = sorted(
        alternatives,
        key=lambda alt: (
            alt.id not in final_nondominated,
            -weighted_sum[alt.id],
            alt.name,
        ),
    )
    winner_id = ranked[0].id

    result = AnalysisResult(
        case_name=case_copy.get("name", "Без названия"),
        ranking=[
            {
                "id": alt.id,
                "name": alt.name,
                "weighted_sum": weighted_sum[alt.id],
                "nondominated": alt.id in final_nondominated,
            }
            for alt in ranked
        ],
        weights=weighted_sum,
        winner_id=winner_id,
        winner_name=next(alt.name for alt in alternatives if alt.id == winner_id),
        explanation=_explain(
            case_copy, analysis_model, weighted_sum, final_nondominated, winner_id
        ),
        metadata={},
    )

    report = {
        "case_name": case_copy.get("name", "Без названия"),
        "raw_vectors": raw_vectors,
        "raw_dominance": raw_dominance,
        "raw_nondominated": raw_nondominated,
        "analysis_model": analysis_model,
        "expanded_vectors": expanded_vectors,
        "final_dominance": final_dominance,
        "final_nondominated": final_nondominated,
        "weighted_sum": weighted_sum,
        "ranking": result.ranking,
        "winner_id": result.winner_id,
        "winner_name": result.winner_name,
        "explanation": result.explanation,
    }
    logger.info("Анализ важности критериев завершён. Победитель: %s", result.winner_id)
    return report
