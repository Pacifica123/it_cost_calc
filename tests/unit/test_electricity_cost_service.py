from it_cost_calc.application.services.electricity_cost_service import ElectricityCostService
from it_cost_calc.domain import ElectricityProfile


def test_electricity_cost_service_applies_round_the_clock_override_for_servers():
    service = ElectricityCostService()

    result = service.calculate(
        equipment_rows=[
            ElectricityProfile(name="srv", quantity=1, max_power=100.0),
            {"name": "pc", "quantity": 2, "max_power": 50.0},
        ],
        hours_per_day=8,
        working_days=20,
        cost_per_kwh=2,
        round_the_clock_names={"srv"},
    )

    assert round(result["total_cost"], 2) == 176.0
    assert result["items"][0]["hours_per_day"] == 24.0
    assert result["items"][0]["working_days"] == 30.0
