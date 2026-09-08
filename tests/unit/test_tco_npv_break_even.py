import pytest

from application.services.tco_model_service import TCOModelService
from domain.finance.npv import calculate_npv


def test_break_even_annual_effect_inverts_constant_flow_npv():
    service = TCOModelService()
    threshold = service.break_even_annual_effect(
        investment=1000.0,
        annual_opex=200.0,
        discount_rate=0.10,
        horizon_years=5,
    )
    annual_effect = threshold["annual_effect"]
    cash_flows = [annual_effect - 200.0] * 5

    assert calculate_npv(1000.0, 0.10, cash_flows) == pytest.approx(0.0, abs=1e-9)


def test_npv_inputs_expose_break_even_effect_without_inventing_actual_effect():
    service = TCOModelService()
    basis = service.build_npv_inputs(
        {"initial_investment": 1000.0, "annual_opex": 200.0},
        horizon_years=5,
        annual_effect=0.0,
        discount_rate=0.10,
    )

    assert basis["annual_effect"] == 0.0
    assert basis["break_even_annual_effect"] > basis["annual_opex"]
    assert any("безубыточного" in warning for warning in basis["warnings"])
