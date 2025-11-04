"""
ahp_config_selector.py

Полностью с нуля: модуль для выполнения итогового проекта по AHP (метод анализа иерархий)
в предметной области "Подбор оптимальной конфигурации оборудования".

Ключевые возможности (соответствуют ТЗ):
- Принимает на вход JSON / Python-структуру с N >= 5 конфигурациями (альтернативами).
- Поддерживает разделение критериев на жёсткие (constraints) и мягкие (для AHP).
- Обрабатывает качественные (нечеткие) критерии заданные в виде интервальных значений
  (fuzzy: low, high) и дефаззифицирует их центроидом.
- Аггрегирует характеристики конфигурации (сумма стоимости, суммарное энергопотребление,
  средняя надёжность и т.п.)
- Фильтрует альтернативы по жёстким ограничениям (budget, max_energy, matching people)
- Программно генерирует матрицы попарных сравнений альтернатив по мягким критериям
  на основе числовых характеристик (отношения значений). Можно ограничить шкалу Саати 1..9
- Поддерживает ввод матриц попарных сравнений важности мягких критериев от >=2 экспертов
  и их агрегацию геометрическим средним
- Вычисляет приоритеты (главный собственный вектор) и проверки согласованности CI/CR
  для матрицы критериев и для матриц альтернатив по каждому критерию
- Формирует итоговую агрегацию AHP и выдаёт ранжирование альтернатив и top-10% по правилу
  (если отфильтрованных мало (<=10) — выводятся все)
- Возвращает структурированный словарь с полным отчётом (матрицы, веса, CI/CR, итоговые баллы).

Пример использования внизу файла.

Примечание: модуль написан для воспроизводимости и простоты — не требует внешних сервисов.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple, Optional
import json
import math
import numpy as np
import copy

# ------------------------------
# Вспомогательные структуры
# ------------------------------

@dataclass
class FuzzyInterval:
    low: float
    high: float

    def centroid(self) -> float:
        return 0.5 * (self.low + self.high)


# ------------------------------
# AHP / математика
# ------------------------------

RI_TABLE = {
    1: 0.0,
    2: 0.0,
    3: 0.58,
    4: 0.90,
    5: 1.12,
    6: 1.24,
    7: 1.32,
    8: 1.41,
    9: 1.45,
    10: 1.49,
    11: 1.51,
    12: 1.48,
}


def geometric_mean_matrix(matrices: Sequence[np.ndarray]) -> np.ndarray:
    """Элемент-по-элементное геометрическое среднее списка положительных матриц."""
    if len(matrices) == 0:
        raise ValueError("No matrices to aggregate")
    arr = np.stack([np.array(m, dtype=float) for m in matrices], axis=2)
    # Предотвращаем нули
    arr[arr <= 0] = 1e-9
    gm = np.prod(arr, axis=2) ** (1.0 / arr.shape[2])
    return gm


def ahp_priority_and_consistency(matrix: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
    """
    Возвращает (weights_vector (sum=1), lambda_max, CI, CR).
    matrix должен быть положительной и квадратичной.
    """
    A = np.array(matrix, dtype=float)
    if A.shape[0] != A.shape[1]:
        raise ValueError("AHP matrix must be square")
    n = A.shape[0]
    # собственный вектор
    eigenvals, eigenvecs = np.linalg.eig(A)
    # выбран максимальный по действительной части
    idx = int(np.argmax(eigenvals.real))
    lam = float(eigenvals[idx].real)
    vec = eigenvecs[:, idx].real
    # положительный вектор
    vec = np.abs(vec)
    if vec.sum() == 0:
        raise RuntimeError("Zero eigenvector")
    w = vec / vec.sum()
    CI = (lam - n) / (n - 1) if n > 1 else 0.0
    RI = RI_TABLE.get(n, 1.49)  # fallback
    CR = CI / RI if RI > 0 else 0.0
    return w, lam, CI, CR


# ------------------------------
# Обработка входных конфигураций
# ------------------------------

def load_configurations_from_json(text: str) -> List[Dict[str, Any]]:
    data = json.loads(text)
    if isinstance(data, dict) and 'configurations' in data:
        cfgs = data['configurations']
    elif isinstance(data, list):
        cfgs = data
    else:
        raise ValueError('JSON должен содержать либо список конфигураций, либо объект с ключом "configurations"')
    return cfgs


def aggregate_configuration(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    На вход — конфигурация в формате:
    {
      "id": "A",
      "devices": [ {"role":"client","vendor":"AMD", "cpu_score":2.0, "energy":50, "cost":300, "reliability": {"low":0.6,"high":0.8} }, ... ],
      "meta": {"people":3}
    }

    Возвращает агрегированные характеристики конфигурации:
    - total_cost
    - total_energy
    - avg_reliability (дефаззификация центроидом)
    - total_performance (сумма cpu_score + ram_score... если нет — 0)
    - counts per role (clients, servers, routers)
    - people (из meta или 0)
    """
    devs = cfg.get('devices', [])
    total_cost = 0.0
    total_energy = 0.0
    perf_sum = 0.0
    rel_vals = []
    counts = {}
    for d in devs:
        role = d.get('role', 'unknown')
        counts[role] = counts.get(role, 0) + 1
        total_cost += float(d.get('cost', 0.0))
        total_energy += float(d.get('energy', 0.0))
        # производительность — суммируем поля, если они есть
        # допустим поля cpu_score и ram_score и custom perf
        perf_sum += float(d.get('cpu_score', 0.0)) + float(d.get('ram_score', 0.0)) + float(d.get('perf', 0.0))
        r = d.get('reliability', None)
        if isinstance(r, dict) and 'low' in r and 'high' in r:
            rel_vals.append(FuzzyInterval(float(r['low']), float(r['high'])).centroid())
        else:
            if r is None:
                pass
            else:
                try:
                    rel_vals.append(float(r))
                except Exception:
                    pass
    avg_reliability = float(np.mean(rel_vals)) if len(rel_vals) > 0 else 0.0
    res = {
        'id': cfg.get('id', None),
        'total_cost': total_cost,
        'total_energy': total_energy,
        'total_performance': perf_sum,
        'avg_reliability': avg_reliability,
        'counts': counts,
        'people': int(cfg.get('meta', {}).get('people', cfg.get('people', 0)))
    }
    return res


# ------------------------------
# Фильтрация по жёстким критериям
# ------------------------------

def filter_hard_constraints(aggregates: Sequence[Dict[str, Any]], constraints: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    constraints может содержать:
      - max_budget
      - max_energy
      - people_match_tolerance: допустимое число лишних клиентских ПК в процентах (например 0.2 = 20%)

    Возвращает (passed, removed)
    """
    passed = []
    removed = []
    for a in aggregates:
        fail_reasons = []
        if 'max_budget' in constraints and a['total_cost'] > float(constraints['max_budget']) + 1e-9:
            fail_reasons.append('budget')
        if 'max_energy' in constraints and a['total_energy'] > float(constraints['max_energy']) + 1e-9:
            fail_reasons.append('energy')
        # people vs client pcs
        people = int(a.get('people', 0))
        clients = int(a.get('counts', {}).get('client', 0))
        tol = float(constraints.get('people_match_tolerance', 0.2))
        # требуем clients >= people; и клиенты сверх людей не должны превышать tol*people
        if people > 0:
            if clients < people:
                fail_reasons.append('not_enough_clients')
            else:
                excess = clients - people
                if excess > math.ceil(tol * people):
                    fail_reasons.append('too_many_clients')
        # если люди == 0 — допускаем любые клиенты
        if len(fail_reasons) == 0:
            passed.append(a)
        else:
            removed.append({'aggregate': a, 'reasons': fail_reasons})
    return passed, removed


# ------------------------------
# Построение матриц попарных сравнений альтернатив по критериям
# ------------------------------

def build_pairwise_matrix_from_scores(scores: Sequence[float], saaty_cap: bool = True) -> np.ndarray:
    """
    Построить матрицу A размером n x n, где A[i,j] = s_i / s_j.
    Опционально капируем отношение в шкалу Саати [1/9, 9] (saаty_cap=True).
    """
    s = np.array(scores, dtype=float)
    n = s.shape[0]
    A = np.ones((n, n), dtype=float)
    # защищаем от нулей
    s_prot = np.copy(s)
    s_prot[s_prot == 0] = 1e-9
    for i in range(n):
        for j in range(n):
            A[i, j] = s_prot[i] / s_prot[j]
            if saaty_cap:
                # приводим к диапазону 1/9 .. 9
                if A[i,j] > 9:
                    A[i,j] = 9.0
                if A[i,j] < 1.0/9.0:
                    A[i,j] = 1.0/9.0
    return A


# ------------------------------
# Основной рабочий процесс
# ------------------------------

def run_ahp_pipeline(
    configurations: Sequence[Dict[str, Any]],
    soft_criteria: Sequence[str],
    experts_criteria_matrices: Sequence[np.ndarray],
    constraints: Dict[str, Any],
    saaty_cap: bool = True,
    top_pct: float = 0.10,
) -> Dict[str, Any]:
    """
    configurations: список исходных конфигураций (см. load_configurations_from_json)
    soft_criteria: список имён критериев, которые используются в AHP (числовые поля в агрегатах)
    experts_criteria_matrices: список матриц (square) от экспертов, каждой матрицей — попарные сравнения важности soft_criteria
        размер матрицы должен быть len(soft_criteria)
    constraints: словарь с жёсткими ограничениями (см. filter_hard_constraints)
    saaty_cap: капать ли матрицы альтернатив в шкалу Саати

    Возвращает словарь с полным отчётом, пригодный для вставки в документ.
    """
    # 1) Агрегируем конфигурации
    aggregates = [aggregate_configuration(c) for c in configurations]

    # 2) Фильтрация по жёстким критериям
    passed, removed = filter_hard_constraints(aggregates, constraints)

    report = {
        'total_input': len(aggregates),
        'passed_count': len(passed),
        'removed_count': len(removed),
        'removed_details': removed,
        'passed': passed,
    }

    if len(passed) == 0:
        report['note'] = 'No feasible configurations after hard constraints.'
        return report

    # 3) Минимальные требования ТЗ: не менее 5 альтернатив
    if len(passed) < 5:
        # предупредим, но продолжаем — можно кластеризовать/генерировать дополнительные, но в рамках ТЗ
        report['warning'] = 'Fewer than 5 feasible alternatives remain after filtering. TЗ требует >=5.'

    # 4) AHP: критерии — soft_criteria
    # Требуется >=2 эксперта
    if len(experts_criteria_matrices) < 2:
        raise ValueError('Требуется минимум 2 матрицы экспертов для сравнения критериев (soft criteria)')
    # Проверяем форму
    for M in experts_criteria_matrices:
        M = np.array(M, dtype=float)
        if M.shape[0] != M.shape[1] or M.shape[0] != len(soft_criteria):
            raise ValueError('Каждая матрица эксперта должна быть квадратной и размером == числу soft_criteria')

    # Аггрегация мнений экспертов по критериям
    crit_agg = geometric_mean_matrix(experts_criteria_matrices)
    crit_weights, crit_lam, crit_CI, crit_CR = ahp_priority_and_consistency(crit_agg)

    report['criteria'] = {
        'names': list(soft_criteria),
        'aggregated_matrix': crit_agg.tolist(),
        'weights': crit_weights.tolist(),
        'lambda_max': crit_lam,
        'CI': crit_CI,
        'CR': crit_CR,
    }

    # 5) Для каждой soft критерия строим матрицу альтернатив
    n_alt = len(passed)
    alt_ids = [a['id'] for a in passed]
    alt_scores_matrix = np.zeros((len(soft_criteria), n_alt), dtype=float)
    alt_pairwise_matrices = []
    alt_weights_per_criterion = []
    alt_consistency_per_criterion = []

    for ci, crit in enumerate(soft_criteria):
        # для каждого альтернативного агрегата извлекаем значение
        vals = []
        for a in passed:
            v = a.get(crit, None)
            if v is None:
                # если поле отсутствует — ставим 0
                vals.append(0.0)
            else:
                vals.append(float(v))
        alt_scores_matrix[ci, :] = np.array(vals, dtype=float)
        P = build_pairwise_matrix_from_scores(vals, saaty_cap=saaty_cap)
        alt_pairwise_matrices.append(P)
        w, lam, CI, CR = ahp_priority_and_consistency(P)
        alt_weights_per_criterion.append(w.tolist())
        alt_consistency_per_criterion.append({'lambda': lam, 'CI': CI, 'CR': CR})

    report['alternatives'] = {
        'ids': alt_ids,
        'scores': alt_scores_matrix.tolist(),
        'pairwise_matrices': [P.tolist() for P in alt_pairwise_matrices],
        'weights_per_criterion': alt_weights_per_criterion,
        'consistency_per_criterion': alt_consistency_per_criterion,
    }

    # 6) Итоговая агрегация: итоговый вес альтернативы = sum_j crit_weight_j * alt_weight_{j,i}
    W_crit = np.array(crit_weights, dtype=float).reshape(1, -1)  # shape (1, k)
    alt_w_matrix = np.array(alt_weights_per_criterion, dtype=float)  # shape (k, n_alt)
    # ensure shapes: alt_w_matrix[j,i]
    if alt_w_matrix.shape[0] != len(soft_criteria) or alt_w_matrix.shape[1] != n_alt:
        alt_w_matrix = alt_w_matrix.reshape(len(soft_criteria), n_alt)

    final_scores = (np.array(crit_weights, dtype=float) @ alt_w_matrix).flatten()

    # нормируем
    if final_scores.sum() > 0:
        final_scores_norm = final_scores / float(final_scores.sum())
    else:
        final_scores_norm = np.ones_like(final_scores) / float(len(final_scores))

    ranked = sorted([(aid, float(score)) for aid, score in zip(alt_ids, final_scores_norm)], key=lambda x: -x[1])

    report['final'] = {
        'raw_final_scores': final_scores.tolist(),
        'normalized_final_scores': final_scores_norm.tolist(),
        'ranking': ranked,
    }

    # 7) Top selection: топ 10% или все если <=10
    if len(passed) <= 10:
        top_selected = [r[0] for r in ranked]
    else:
        top_n = max(1, math.ceil(len(passed) * top_pct))
        top_selected = [r[0] for r in ranked[:top_n]]

    report['selection'] = {
        'top_pct': top_pct,
        'selected_ids': top_selected,
        'selected_count': len(top_selected),
    }

    return report


# ------------------------------
# Пример использования
# ------------------------------
if __name__ == '__main__':
    # Пример входных данных — 6 конфигураций
    sample_json = {
        'configurations': [
            {
                'id': 'A',
                'devices': [
                    {'role': 'client', 'vendor': 'Intel', 'cpu_score': 2.0, 'ram_score': 1.0, 'energy': 50, 'cost': 300, 'reliability': {'low': 0.7, 'high': 0.9}},
                    {'role': 'client', 'vendor': 'AMD', 'cpu_score': 2.2, 'ram_score': 1.0, 'energy': 55, 'cost': 310, 'reliability': {'low': 0.75, 'high': 0.9}},
                    {'role': 'server', 'vendor': 'Intel', 'cpu_score': 5.0, 'ram_score': 4.0, 'energy': 200, 'cost': 2000, 'reliability': {'low': 0.85, 'high': 0.95}},
                ],
                'meta': {'people': 3}
            },
            {
                'id': 'B',
                'devices': [
                    {'role': 'client', 'vendor': 'AMD', 'cpu_score': 2.5, 'ram_score': 1.2, 'energy': 60, 'cost': 320, 'reliability': {'low': 0.8, 'high': 0.95}},
                    {'role': 'client', 'vendor': 'AMD', 'cpu_score': 2.5, 'ram_score': 1.2, 'energy': 60, 'cost': 320, 'reliability': {'low': 0.8, 'high': 0.95}},
                    {'role': 'server', 'vendor': 'Intel', 'cpu_score': 4.5, 'ram_score': 3.5, 'energy': 180, 'cost': 1800, 'reliability': {'low': 0.8, 'high': 0.92}},
                ],
                'meta': {'people': 2}
            },
            {
                'id': 'C',
                'devices': [
                    {'role': 'client', 'vendor': 'Intel', 'cpu_score': 1.5, 'ram_score': 0.8, 'energy': 40, 'cost': 250, 'reliability': {'low': 0.6, 'high': 0.75}},
                    {'role': 'client', 'vendor': 'Intel', 'cpu_score': 1.5, 'ram_score': 0.8, 'energy': 40, 'cost': 250, 'reliability': {'low': 0.6, 'high': 0.75}},
                    {'role': 'client', 'vendor': 'Intel', 'cpu_score': 1.5, 'ram_score': 0.8, 'energy': 40, 'cost': 250, 'reliability': {'low': 0.6, 'high': 0.75}},
                    {'role': 'server', 'vendor': 'AMD', 'cpu_score': 4.0, 'ram_score': 3.0, 'energy': 160, 'cost': 1600, 'reliability': {'low': 0.75, 'high': 0.9}},
                ],
                'meta': {'people': 3}
            },
            {
                'id': 'D',
                'devices': [
                    {'role': 'client', 'vendor': 'Generic', 'cpu_score': 1.0, 'ram_score': 0.5, 'energy': 30, 'cost': 150, 'reliability': {'low': 0.4, 'high': 0.6}},
                    {'role': 'server', 'vendor': 'Generic', 'cpu_score': 3.0, 'ram_score': 2.0, 'energy': 120, 'cost': 1000, 'reliability': {'low': 0.5, 'high': 0.7}},
                ],
                'meta': {'people': 1}
            },
            {
                'id': 'E',
                'devices': [
                    {'role': 'client', 'vendor': 'Intel', 'cpu_score': 2.0, 'ram_score': 1.0, 'energy': 50, 'cost': 300, 'reliability': {'low': 0.7, 'high': 0.85}},
                    {'role': 'client', 'vendor': 'Intel', 'cpu_score': 2.0, 'ram_score': 1.0, 'energy': 50, 'cost': 300, 'reliability': {'low': 0.7, 'high': 0.85}},
                    {'role': 'client', 'vendor': 'Intel', 'cpu_score': 2.0, 'ram_score': 1.0, 'energy': 50, 'cost': 300, 'reliability': {'low': 0.7, 'high': 0.85}},
                    {'role': 'client', 'vendor': 'Intel', 'cpu_score': 2.0, 'ram_score': 1.0, 'energy': 50, 'cost': 300, 'reliability': {'low': 0.7, 'high': 0.85}},
                    {'role': 'server', 'vendor': 'Intel', 'cpu_score': 5.5, 'ram_score': 5.0, 'energy': 250, 'cost': 2500, 'reliability': {'low': 0.9, 'high': 0.98}},
                ],
                'meta': {'people': 4}
            },
            {
                'id': 'F',
                'devices': [
                    {'role': 'client', 'vendor': 'AMD', 'cpu_score': 2.5, 'ram_score': 1.2, 'energy': 60, 'cost': 320, 'reliability': {'low': 0.7, 'high': 0.9}},
                    {'role': 'router', 'vendor': 'Cisco', 'cpu_score': 0.0, 'ram_score': 0.0, 'energy': 50, 'cost': 400, 'reliability': {'low': 0.85, 'high': 0.95}},
                    {'role': 'server', 'vendor': 'AMD', 'cpu_score': 4.8, 'ram_score': 3.8, 'energy': 190, 'cost': 1900, 'reliability': {'low': 0.82, 'high': 0.93}},
                ],
                'meta': {'people': 2}
            }
        ]
    }

    # Назначаем soft критерии (нечеткие и числовые поля в агрегатах)
    soft_criteria = ['avg_reliability', 'total_performance', 'total_cost', 'total_energy', 'lifespan']

    # Создаём две матрицы экспертов для важности критериев (пример)
    # В реальности эксперты заполняют poparno матрицы. Здесь — искусственные примеры.
    M1 = np.array([
        [1, 3, 5, 4, 3],
        [1/3, 1, 3, 2, 2],
        [1/5, 1/3, 1, 1/2, 1/3],
        [1/4, 1/2, 2, 1, 1/2],
        [1/3, 1/2, 3, 2, 1],
    ], dtype=float)
    M2 = np.array([
        [1, 2, 4, 3, 2],
        [1/2, 1, 2, 2, 1],
        [1/4, 1/2, 1, 1/2, 1/3],
        [1/3, 1/2, 2, 1, 1/2],
        [1/2, 1, 3, 2, 1],
    ], dtype=float)

    # constraints
    constraints = {
        'max_budget': 6000.0,
        'max_energy': 700.0,
        'people_match_tolerance': 0.25
    }

    # run
    configs = sample_json['configurations']

    # Расширим агрегированные поля — добавить lifespan для примера
    for c in configs:
        # lifespan в годах — примерно связана с vendor: generic -> 3, intel/amd -> 5-8
        lif = 5
        for d in c['devices']:
            v = d.get('vendor', '').lower()
            if 'generic' in v:
                lif = min(lif, 3)
            if 'intel' in v:
                lif = max(lif, 6)
        c.setdefault('meta', {})
        c['meta']['notes'] = 'sample'

    # аггрегируем и добавим lifespan в агрегаты
    aggs = [aggregate_configuration(c) for c in configs]
    for a, c in zip(aggs, configs):
        # примитивно: lifespan = median vendor-based; но для демо добавим 5
        a['lifespan'] = 5.0

    res = run_ahp_pipeline(configurations=configs, soft_criteria=soft_criteria, experts_criteria_matrices=[M1, M2], constraints=constraints)

    # Сохраним отчет
    with open('ahp_report_sample.json', 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print('Run finished. Отчёт сохранён в ahp_report_sample.json')
