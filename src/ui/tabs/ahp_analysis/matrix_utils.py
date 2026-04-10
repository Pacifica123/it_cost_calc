from __future__ import annotations

from fractions import Fraction
from typing import Iterable, Sequence

import numpy as np

from .constants import DEFAULT_EXPERT_MATRICES, DEFAULT_SOFT_CRITERIA


ALLOWED_PAIRWISE_HINTS = ("1/9", "1/7", "1/5", "1/3", "1", "3", "5", "7", "9")


def criterion_indices(criteria_ids: Iterable[str]) -> list[int]:
    indices = []
    for criterion_id in criteria_ids:
        if criterion_id not in DEFAULT_SOFT_CRITERIA:
            raise ValueError(f"Неизвестный критерий AHP: {criterion_id}")
        indices.append(DEFAULT_SOFT_CRITERIA.index(criterion_id))
    return indices


def extract_default_expert_matrices(criteria_ids: Sequence[str]) -> list[np.ndarray]:
    if not criteria_ids:
        raise ValueError("Нужно выбрать хотя бы один критерий")
    indices = criterion_indices(criteria_ids)
    matrices: list[np.ndarray] = []
    for matrix in DEFAULT_EXPERT_MATRICES:
        subset = np.array(matrix, dtype=float)[np.ix_(indices, indices)]
        matrices.append(subset)
    return matrices


def default_pairwise_value(left_id: str, right_id: str) -> float:
    matrices = extract_default_expert_matrices(DEFAULT_SOFT_CRITERIA)
    li = DEFAULT_SOFT_CRITERIA.index(left_id)
    ri = DEFAULT_SOFT_CRITERIA.index(right_id)
    values = [float(matrix[li, ri]) for matrix in matrices]
    product = 1.0
    for value in values:
        product *= value
    return product ** (1.0 / len(values))


def parse_pairwise_value(raw_value: str | float | int) -> float:
    if isinstance(raw_value, (int, float)):
        value = float(raw_value)
    else:
        text = str(raw_value).strip().replace(",", ".")
        if not text:
            raise ValueError("Значение попарного сравнения не может быть пустым")
        if "/" in text:
            value = float(Fraction(text))
        else:
            value = float(text)
    if value <= 0:
        raise ValueError("Значение попарного сравнения должно быть положительным")
    return value


def format_pairwise_value(value: float) -> str:
    fraction = Fraction(float(value)).limit_denominator(9)
    if abs(float(fraction) - float(value)) < 1e-6 and fraction.denominator != 1:
        return f"{fraction.numerator}/{fraction.denominator}"
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def build_pairwise_matrix(criteria_ids: Sequence[str], value_provider) -> np.ndarray:
    size = len(criteria_ids)
    if size == 0:
        raise ValueError("Нужно выбрать хотя бы один критерий")
    matrix = np.ones((size, size), dtype=float)
    for row_index in range(size):
        for col_index in range(row_index + 1, size):
            raw = value_provider(criteria_ids[row_index], criteria_ids[col_index])
            value = parse_pairwise_value(raw)
            matrix[row_index, col_index] = value
            matrix[col_index, row_index] = 1.0 / value
    return matrix
