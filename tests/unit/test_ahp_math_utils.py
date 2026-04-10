import numpy as np
import pytest

from it_cost_calc.domain.decision.ahp.math_utils import (
    ahp_priority_and_consistency,
    build_pairwise_matrix_from_scores,
    geometric_mean_matrix,
)


def test_geometric_mean_matrix_aggregates_expert_matrices_elementwise():
    matrix_a = np.array([[1, 4], [1 / 4, 1]], dtype=float)
    matrix_b = np.array([[1, 9], [1 / 9, 1]], dtype=float)

    aggregated = geometric_mean_matrix([matrix_a, matrix_b])

    assert aggregated[0, 1] == pytest.approx(6.0)
    assert aggregated[1, 0] == pytest.approx(1 / 6)


def test_ahp_priority_and_consistency_returns_expected_vector_for_consistent_matrix():
    matrix = np.array(
        [
            [1, 2, 4],
            [1 / 2, 1, 2],
            [1 / 4, 1 / 2, 1],
        ],
        dtype=float,
    )

    weights, lambda_max, ci, cr = ahp_priority_and_consistency(matrix)

    assert list(weights) == pytest.approx([4 / 7, 2 / 7, 1 / 7], rel=1e-6)
    assert lambda_max == pytest.approx(3.0)
    assert ci == pytest.approx(0.0)
    assert cr == pytest.approx(0.0)


def test_build_pairwise_matrix_from_scores_caps_values_to_saaty_scale():
    matrix = build_pairwise_matrix_from_scores([1000.0, 100.0, 1.0], saaty_cap=True)

    assert matrix[0, 1] == pytest.approx(9.0)
    assert matrix[0, 2] == pytest.approx(9.0)
    assert matrix[2, 0] == pytest.approx(1 / 9)
