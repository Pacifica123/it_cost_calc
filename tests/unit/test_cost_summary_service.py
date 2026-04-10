from application.services.cost_summary_service import CostSummaryService


class DummyCrud:
    def __init__(self):
        self.entities = {
            "server": [{"name": "srv", "quantity": 2, "price": 100.0}],
            "client": [{"name": "pc", "quantity": 3, "price": 50.0}],
            "subscription_licenses": [{"name": "lic", "monthly_cost": 10.0}],
            "migration": [{"name": "setup", "one_time_cost": 25.0}],
        }


def test_cost_summary_service_calculates_expected_totals():
    service = CostSummaryService()
    crud = DummyCrud()

    service.update_capital_costs(crud)
    service.update_operational_costs(crud)
    service.update_electricity_costs(7.5)

    totals = service.calculate_totals()

    assert totals["total_capital"] == 350.0
    assert totals["total_operational_one_time"] == 25.0
    assert totals["total_operational_monthly"] == 10.0
    assert totals["electricity_costs"] == 7.5
