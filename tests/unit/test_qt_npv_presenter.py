from pathlib import Path

from shared.constants import ANALYSIS_SCOPE_TECHNICAL
from ui_qt.presenters import NpvInput, NpvPresenter, QtAppPresenter
from ui_qt.presenters.npv_presenter import format_cash_flows, parse_cash_flows


def _app(tmp_path: Path) -> QtAppPresenter:
    return QtAppPresenter(
        repo_root=Path(__file__).resolve().parents[2],
        runtime_entities_path=tmp_path / "runtime_entities.json",
    )


def _select_technical_candidate(app: QtAppPresenter) -> None:
    app.scoped_candidate_pool_service.replace(
        ANALYSIS_SCOPE_TECHNICAL,
        [
            {
                "id": "GA-1",
                "name": "Выбранная ТО-конфигурация",
                "scope": ANALYSIS_SCOPE_TECHNICAL,
                "components": [
                    {
                        "name": "srv",
                        "category": "server",
                        "source_category": "server",
                        "scope": ANALYSIS_SCOPE_TECHNICAL,
                        "component_type": "server",
                        "quantity": 1,
                        "price": 1000,
                        "max_power": 100,
                    },
                    {
                        "name": "rent",
                        "category": "server_rental",
                        "source_category": "server_rental",
                        "scope": ANALYSIS_SCOPE_TECHNICAL,
                        "component_type": "server",
                        "monthly_cost": 100,
                    },
                ],
                "totals": {"capital_cost": 1000, "client_seats": 1, "total_power_watts": 100},
                "metrics": {"ga_score": 1.0},
                "source": "ga",
                "metadata": {"rank": 1},
            }
        ],
        source_label="test GA",
        source_method="ga",
    )
    app.run_ahp_for_candidate_pool(ANALYSIS_SCOPE_TECHNICAL)
    app.run_pareto_for_candidate_pool(ANALYSIS_SCOPE_TECHNICAL)
    app.run_hybrid_assessment(ANALYSIS_SCOPE_TECHNICAL)


def test_npv_presenter_calculates_manual_flow(tmp_path: Path):
    presenter = NpvPresenter(_app(tmp_path))

    report = presenter.calculate(
        NpvInput(
            investment=1000,
            discount_rate=0.1,
            cash_flows=(400, 400, 400),
            horizon_years=3,
        )
    )
    summary = presenter.summary(report)

    assert round(summary.npv, 2) == -5.26
    assert summary.years == 3
    assert presenter.table_rows(report)[0]["year"] == 0
    assert presenter.table_rows(report)[-1]["accumulated_npv"] == report["rows"][-1]["accumulated_npv"]
    chart_points = presenter.chart_points(report)
    assert [point.year for point in chart_points] == [0, 1, 2, 3]
    assert chart_points[-1].accumulated_npv == report["rows"][-1]["accumulated_npv"]
    assert summary.rate == 0.1


def test_npv_presenter_prepares_empty_basis_until_hybrid_selection(tmp_path: Path):
    app = _app(tmp_path)
    app.replace_entities(
        {
            "server": [{"name": "unused", "quantity": 50, "price": 1000, "max_power": 100}],
            "server_rental": [{"name": "unused-rent", "monthly_cost": 100}],
        }
    )
    presenter = NpvPresenter(app)

    values = presenter.prepare_from_costs(
        horizon_years=4,
        annual_effect=5000,
        discount_rate=0.12,
    )

    assert values.investment == 0.0
    assert values.cash_flows == (5000.0, 5000.0, 5000.0, 5000.0)
    assert presenter.basis_rows()[0]["value"] == 0.0


def test_npv_presenter_prepares_basis_from_selected_hybrid_costs(tmp_path: Path):
    app = _app(tmp_path)
    _select_technical_candidate(app)
    app.calculate_electricity_costs(hours_per_day=8, working_days=20, cost_per_kwh=5)
    presenter = NpvPresenter(app)

    values = presenter.prepare_from_costs(
        horizon_years=4,
        annual_effect=5000,
        discount_rate=0.12,
    )

    assert values.investment == 1000.0
    assert values.horizon_years == 4
    assert len(values.cash_flows) == 4
    assert values.cash_flows[0] < 5000.0
    assert presenter.basis_rows()[0]["name"] == "CAPEX"
    assert presenter.basis_rows()[0]["value"] == 1000.0


def test_cash_flow_parse_and_format_helpers():
    assert parse_cash_flows("10; 20,30") == (10.0, 20.0, 30.0)
    assert parse_cash_flows("", fallback_years=2, fallback_value=7) == (7.0, 7.0)
    assert format_cash_flows([1, 2.5]) == "1.00, 2.50"
