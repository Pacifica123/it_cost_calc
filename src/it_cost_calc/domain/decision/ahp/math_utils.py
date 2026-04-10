from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from .models import RI_TABLE


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


def build_pairwise_matrix_from_scores(
    scores: Sequence[float], saaty_cap: bool = True
) -> np.ndarray:
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
                if A[i, j] > 9:
                    A[i, j] = 9.0
                if A[i, j] < 1.0 / 9.0:
                    A[i, j] = 1.0 / 9.0
    return A
