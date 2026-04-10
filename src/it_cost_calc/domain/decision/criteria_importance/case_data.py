from __future__ import annotations

import copy
from typing import Any, Dict, List

from it_cost_calc.domain import (
    Alternative,
    Criterion,
    Relation,
    RelationOperator,
    ensure_alternative,
    ensure_criterion,
    ensure_relation,
)

DEFAULT_BUDGETING_CASE: Dict[str, Any] = {
    "name": "Выбор системы бюджетирования",
    "criteria": [
        {"id": "1", "name": "Аппаратная + Программная платформа"},
        {"id": "2", "name": "Масштабируемость"},
        {"id": "3", "name": "Гибкость настроек"},
        {"id": "4.1", "name": "Возможность совместной работы"},
        {"id": "4.2", "name": "Формирование планов и бюджетов"},
        {"id": "4.3", "name": "Аналитические возможности и контроль исполнения бюджета"},
        {"id": "4.4", "name": "Отчетность и анализ"},
        {"id": "5", "name": "Связь с внешними источниками информации (импорт/экспорт)"},
        {"id": "6", "name": "Интерфейс"},
        {"id": "7", "name": "Стоимость владения"},
        {
            "id": "8",
            "name": "Процесс внедрения и сопровождения + потенциал поставщика решения + широта использования",
        },
    ],
    "alternatives": [
        {"id": "system1", "name": "Система 1"},
        {"id": "system2", "name": "Система 2"},
        {"id": "system3", "name": "Система 3"},
    ],
    "scores": {
        "system1": {
            "1": 4.0,
            "2": 5.0,
            "3": 4.0,
            "4.1": 4.0,
            "4.2": 4.0,
            "4.3": 5.0,
            "4.4": 3.5,
            "5": 5.0,
            "6": 4.0,
            "7": 5.0,
            "8": 4.0,
        },
        "system2": {
            "1": 5.0,
            "2": 5.0,
            "3": 5.0,
            "4.1": 5.0,
            "4.2": 4.5,
            "4.3": 5.0,
            "4.4": 3.5,
            "5": 5.0,
            "6": 5.0,
            "7": 4.0,
            "8": 5.0,
        },
        "system3": {
            "1": 5.0,
            "2": 5.0,
            "3": 4.0,
            "4.1": 3.5,
            "4.2": 5.0,
            "4.3": 5.0,
            "4.4": 5.0,
            "5": 5.0,
            "6": 5.0,
            "7": 4.0,
            "8": 5.0,
        },
    },
    "relations": [
        {"left": "4.1", "op": ">", "factor": 2.0, "right": "4.4"},
        {"left": "4.1", "op": "=", "factor": 1.0, "right": "4.2"},
        {"left": "4.2", "op": "=", "factor": 1.0, "right": "4.3"},
        {"left": "4.2", "op": "=", "factor": 1.0, "right": "3"},
        {"left": "3", "op": ">", "factor": 4.0 / 3.0, "right": "6"},
        {"left": "1", "op": ">", "factor": 3.0 / 2.0, "right": "8"},
        {"left": "3", "op": "=", "factor": 1.0, "right": "5"},
        {"left": "4.4", "op": ">", "factor": 4.0 / 3.0, "right": "6"},
        {"left": "6", "op": "=", "factor": 1.0, "right": "2"},
        {"left": "1", "op": "=", "factor": 1.0, "right": "2"},
        {"left": "8", "op": ">", "factor": 2.0, "right": "7"},
    ],
}


def load_default_budgeting_case() -> Dict[str, Any]:
    return copy.deepcopy(DEFAULT_BUDGETING_CASE)


def _criteria(case_data: Dict[str, Any]) -> List[Criterion]:
    return [ensure_criterion(item) for item in case_data.get("criteria", [])]


def _alternatives(case_data: Dict[str, Any]) -> List[Alternative]:
    return [ensure_alternative(item) for item in case_data.get("alternatives", [])]


def _relations(case_data: Dict[str, Any]) -> List[Relation]:
    return [ensure_relation(item) for item in case_data.get("relations", [])]


def validate_case(case_data: Dict[str, Any]) -> None:
    criteria = _criteria(case_data)
    alternatives = _alternatives(case_data)
    relations = _relations(case_data)
    criteria_ids = {criterion.id for criterion in criteria}
    alt_ids = {alternative.id for alternative in alternatives}
    if not criteria_ids:
        raise ValueError("Нет критериев для анализа")
    if len(alternatives) < 2:
        raise ValueError("Для анализа нужны минимум две альтернативы")

    scores = case_data.get("scores", {})
    for alt_id in alt_ids:
        if alt_id not in scores:
            raise ValueError(f"Нет оценок для альтернативы {alt_id}")
        for criterion_id in criteria_ids:
            if criterion_id not in scores[alt_id]:
                raise ValueError(f"Нет оценки для критерия {criterion_id} у альтернативы {alt_id}")
            float(scores[alt_id][criterion_id])

    for rel in relations:
        left = rel.left
        right = rel.right
        if left not in criteria_ids or right not in criteria_ids:
            raise ValueError(f"Связь {left} ? {right} ссылается на неизвестный критерий")
        op = rel.operator
        if op not in (RelationOperator.EQUAL, RelationOperator.GREATER):
            raise ValueError(f"Неподдерживаемая операция: {op.value}")
        factor = float(rel.factor)
        if factor <= 0:
            raise ValueError("Коэффициент важности должен быть положительным")
