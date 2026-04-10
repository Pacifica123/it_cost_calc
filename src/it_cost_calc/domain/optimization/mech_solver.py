from __future__ import annotations

import datetime
import logging
import itertools
import json
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np

from .mech_models import FuzzyTerm, Item
from .mech_utils import to_json_serializable

logger = logging.getLogger(__name__)


class KnapsackMechanic:
    """
    Класс, реализующий перебор рюкзачных подмножеств и вычисление метрик.

    Параметры конструктора:
        items: список объектов Item
        max_constraint: числовой предел (например, максимальный вес)
        constraint_func: функция подсчёта ограничивающей метрики по подмножеству (subset -> float)
        criteria_funcs: список функций критериев (subset -> float)
        optimize_directions: список 1 или -1, 1 если критерий нужно максимизировать, -1 если минимизировать
        pairwise_matrix: матрица попарных сравнений для AHP
        exclude_empty: если True, пустое множество будет исключено из рассмотрения
        dominance_eps: порог при сравнении доминирования (учёт плавающей погрешности)
    """

    def __init__(
        self,
        items: Sequence[Item],
        max_constraint: float,
        constraint_func: Callable[[Sequence[Item]], float],
        criteria_funcs: Sequence[Callable[[Sequence[Item]], float]],
        optimize_directions: Sequence[int],
        pairwise_matrix: np.ndarray,
        exclude_empty: bool = True,
        dominance_eps: float = 1e-9,
    ):
        self.items = list(items)
        self.max_constraint = float(max_constraint)
        self.constraint_func = constraint_func
        self.criteria_funcs = list(criteria_funcs)
        self.optimize_directions = list(optimize_directions)
        self.pairwise_matrix = np.array(pairwise_matrix, dtype=float)
        self.exclude_empty = bool(exclude_empty)
        self.dominance_eps = float(dominance_eps)

        # Validate sizes
        if len(self.criteria_funcs) != len(self.optimize_directions):
            raise ValueError("criteria_funcs and optimize_directions must have same length")

        if self.pairwise_matrix.shape[0] != self.pairwise_matrix.shape[
            1
        ] or self.pairwise_matrix.shape[0] != len(self.criteria_funcs):
            raise ValueError("pairwise_matrix must be square and match number of criteria")

    # --------------------------
    # AHP / Весы
    # --------------------------
    def calculate_ahp_weights(self) -> np.ndarray:
        """
        Вычисляет веса критериев методом собственных векторов (AHP).
        Возвращает нормированный (sum==1) вектор весов.
        """
        eigenvalues, eigenvectors = np.linalg.eig(self.pairwise_matrix)
        # Используем максимальный по модулю собственный корень
        max_index = int(np.argmax(eigenvalues.real))
        principal = eigenvectors[:, max_index].real
        # Если вектор содержит отрицательные значения — берем по модулю (стандартная практика)
        principal = np.abs(principal)
        if principal.sum() == 0:
            raise RuntimeError("Invalid AHP eigenvector (sum is zero)")
        weights = principal / principal.sum()
        return weights

    # --------------------------
    # Перебор допустимых подмножеств и формирование векторов критериев
    # --------------------------
    def enumerate_feasible(self) -> List[Tuple[Tuple[Item, ...], Tuple[float, ...]]]:
        """
        Перечисляет все подмножества, удовлетворяющие constraint <= max_constraint,
        и возвращает список (subset_tuple, raw_scores_tuple).
        raw_scores уже приведены к направлению 'чем больше - тем лучше' путем умножения на optimize_directions.
        """

        def defuzzify(value):
            if isinstance(value, FuzzyTerm):
                return value.centroid()
            return float(value)

        feasible = []
        for r in range(len(self.items) + 1):
            for comb in itertools.combinations(self.items, r):
                if self.exclude_empty and len(comb) == 0:
                    continue
                cval = self.constraint_func(comb)
                if cval <= self.max_constraint + 1e-12:
                    # Считаем критерии и приводим к "больше - лучше"
                    raw = tuple(
                        direction * defuzzify(func(comb))
                        for func, direction in zip(self.criteria_funcs, self.optimize_directions)
                    )
                    feasible.append((comb, raw))
        return feasible

    # --------------------------
    # Удаление дубликатов по векторам критериев
    # --------------------------
    @staticmethod
    def deduplicate_by_scores(
        feasible: List[Tuple[Tuple[Item, ...], Tuple[float, ...]]],
    ) -> List[Tuple[Tuple[Item, ...], Tuple[float, ...]]]:
        """
        Оставляет только первое вхождение для каждого уникального вектора критериев (raw_scores tuple).
        """
        seen = {}
        deduped = []
        for subset, scores in feasible:
            key = tuple(np.round(scores, 12))  # округление для устойчивости плавающей арифметики
            if key not in seen:
                seen[key] = True
                deduped.append((subset, scores))
        return deduped

    # --------------------------
    # Доминирование (Pareto)
    # --------------------------
    def dominates(self, v1: Sequence[float], v2: Sequence[float]) -> bool:
        """
        Проверка, доминирует ли v1 над v2. Здесь больше - лучше.
        Добавлен eps для нечувствительности к очень маленьким числовым погрешностям.
        """
        ge = all((x + self.dominance_eps) >= y for x, y in zip(v1, v2))
        gt = any((x - self.dominance_eps) > y for x, y in zip(v1, v2))
        return ge and gt

    def fuzzy_dominates(self, v1, v2):
        # каждая компонента — степень уверенности, что v1[i] ≥ v2[i]
        degrees = [min(1.0, max(0.0, (x - y + 1) / 2)) for x, y in zip(v1, v2)]
        return np.mean(degrees) > 0.7  # доминирует, если уверенность > 0.7

    def pareto_filter(
        self, scored: List[Tuple[Tuple[Item, ...], Tuple[float, ...]]]
    ) -> List[Tuple[Tuple[Item, ...], Tuple[float, ...]]]:
        """
        Отбирает набор непобедимых (Pareto) решений из списка (subset, raw_scores).
        Работает на сырых (приведённых к 'больше лучше') значениях.
        """
        pareto = []
        pareto_scores = []
        for subset, scores in scored:
            dominated = False
            to_remove = []
            for i, existing in enumerate(pareto_scores):
                if self.dominates(existing, scores):
                    dominated = True
                    break
                if self.dominates(scores, existing):
                    to_remove.append(i)
            if not dominated:
                for i in reversed(to_remove):
                    pareto.pop(i)
                    pareto_scores.pop(i)
                pareto.append((subset, scores))
                pareto_scores.append(scores)
        return pareto

    # --------------------------
    # Нормализация и метрики
    # --------------------------
    @staticmethod
    def normalize_list(
        scores_list: Sequence[Tuple[float, ...]],
    ) -> Tuple[List[Tuple[float, ...]], np.ndarray, np.ndarray]:
        """
        Нормализует список векторов по формуле (x - min) / (max - min).
        Возвращает (normalized_list, mins, maxs).
        Если для компоненты max==min, задаёт диапазон 1 (без изменения значения).
        """
        arr = np.array(scores_list, dtype=float)
        mins = arr.min(axis=0)
        maxs = arr.max(axis=0)
        ranges = maxs - mins
        ranges[ranges == 0] = 1.0
        normalized = (arr - mins) / ranges
        norm_tuples = [tuple(row) for row in normalized]
        return norm_tuples, mins, maxs

    def weighted_agg(self, norm_scores: Sequence[float], weights: Sequence[float]) -> float:
        """Взвешенная сумма (нормализованные значения и веса AHP)."""
        return float(np.dot(np.array(norm_scores, dtype=float), np.array(weights, dtype=float)))

    def weighted_distance_to_ideal(
        self, norm_scores: Sequence[float], weights: Sequence[float]
    ) -> float:
        """
        Взвешенное евклидово расстояние до идеального вектора (идеал = 1 для каждой нормализованной компоненты).
        Использует те же веса, чтобы обеспечить согласованность с агрегированной оценкой.
        """
        # Идеал после нормализации — единичный вектор (поскольку нормализация сделана так, что максимум -> 1)
        diff = np.array([1.0] * len(norm_scores), dtype=float) - np.array(norm_scores, dtype=float)
        # Применим веса: distance = sqrt( sum (w_i * diff_i^2) )
        w = np.array(weights, dtype=float)
        return float(np.sqrt(np.sum(w * (diff**2))))

    # --------------------------
    # Основной воркфлоу
    # --------------------------
    def solve(self) -> Dict[str, Any]:
        """
        Выполняет полный процесс:
         - Перечисление допустимых подмножеств
         - Удаление дубликатов
         - Отбор парето-решений
         - Нормализация (по всем парето-решениям)
         - Вычисление агрегированной оценки и расстояния до идеала
        Возвращает словарь с результатами и метаданными.
        """
        log = []
        logger.info(
            "Запуск перебора KnapsackMechanic: items=%s, max_constraint=%s",
            len(self.items),
            self.max_constraint,
        )
        timestamp = datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z")
        log.append(f"Run at {timestamp}")
        log.append(f"Items count: {len(self.items)}, max_constraint: {self.max_constraint}")

        # AHP веса
        weights = self.calculate_ahp_weights()
        log.append(f"AHP weights: {weights.tolist()}")

        feasible = self.enumerate_feasible()
        log.append(f"Feasible count (before dedup): {len(feasible)}")
        feasible = self.deduplicate_by_scores(feasible)
        log.append(f"Feasible count (after dedup): {len(feasible)}")

        if len(feasible) == 0:
            self._write_log(log)
            return {
                "timestamp": timestamp,
                "weights": weights.tolist(),
                "pareto": [],
                "ideal": None,
                "mins": None,
                "maxs": None,
                "notes": "No feasible solutions found.",
            }

        # Pareto selection on raw (bigger is better) scores
        pareto_raw = self.pareto_filter(feasible)
        log.append(f"Pareto count (raw): {len(pareto_raw)}")

        # Extract raw score vectors for normalization (use pareto set)
        pareto_scores = [scores for (_subset, scores) in pareto_raw]
        norm_scores, mins, maxs = self.normalize_list(pareto_scores)
        log.append(f"Normalization mins: {mins.tolist()}, maxs: {maxs.tolist()}")

        # ideal in normalized space is vector of all 1.0
        ideal = tuple([1.0] * len(self.criteria_funcs))

        results = []
        for (subset, raw_scores), nvec in zip(pareto_raw, norm_scores):
            agg = self.weighted_agg(nvec, weights)
            dist = self.weighted_distance_to_ideal(nvec, weights)
            results.append(
                {
                    "items": [it.properties for it in subset],
                    "raw_scores": tuple(raw_scores),
                    "normalized_scores": tuple(nvec),
                    "agg_score": float(agg),
                    "dist_to_ideal": float(dist),
                }
            )
            log.append(
                json.dumps(
                    {
                        "items": [to_json_serializable(it.properties) for it in subset],
                        "raw_scores": tuple(raw_scores),
                        "normalized_scores": tuple(nvec),
                        "agg_score": float(agg),
                        "dist_to_ideal": float(dist),
                    },
                    ensure_ascii=False,
                )
            )

        # Запишем лог
        self._write_log(log)

        return {
            "timestamp": timestamp,
            "weights": weights.tolist(),
            "pareto": results,
            "ideal": ideal,
            "mins": mins.tolist(),
            "maxs": maxs.tolist(),
        }

    @staticmethod
    def _write_log(lines: List[str]):
        for line in lines:
            logger.debug("[mech] %s", line)
