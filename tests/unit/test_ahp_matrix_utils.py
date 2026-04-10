import numpy as np

from ui.tabs.ahp_analysis.matrix_utils import (
    build_pairwise_matrix,
    extract_default_expert_matrices,
    format_pairwise_value,
    parse_pairwise_value,
)


def test_extract_default_expert_matrices_respects_selected_criteria_order_and_size():
    matrices = extract_default_expert_matrices(["avg_reliability", "total_cost", "lifespan"])

    assert len(matrices) == 2
    assert matrices[0].shape == (3, 3)
    assert float(matrices[0][0, 0]) == 1.0


def test_build_pairwise_matrix_supports_fraction_strings_and_reciprocals():
    matrix = build_pairwise_matrix(
        ["avg_reliability", "total_performance", "lifespan"],
        lambda left, right: {
            ("avg_reliability", "total_performance"): "3",
            ("avg_reliability", "lifespan"): "5",
            ("total_performance", "lifespan"): "1/3",
        }[(left, right)],
    )

    assert np.isclose(matrix[0, 1], 3.0)
    assert np.isclose(matrix[1, 0], 1 / 3)
    assert np.isclose(matrix[1, 2], 1 / 3)
    assert np.isclose(matrix[2, 1], 3.0)
    assert parse_pairwise_value("1/5") == 0.2
    assert format_pairwise_value(1 / 3) == "1/3"
