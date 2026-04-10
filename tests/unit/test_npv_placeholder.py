import pytest

from it_cost_calc.domain.finance.npv import calculate_npv


def test_calculate_npv_matches_expected_discounted_cash_flow_value():
    result = calculate_npv(1000, 0.1, [500, 600, 700])

    assert result == pytest.approx(476.33358377159993)
