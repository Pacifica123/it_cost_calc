from pathlib import Path

from shared.constants import ANALYSIS_SCOPE_TECHNICAL
from ui_qt.presenters import EnergyPresenter, EnergyProfileInput, QtAppPresenter


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
                        "price": 10,
                        "max_power": 400,
                    },
                    {
                        "name": "pc",
                        "category": "client",
                        "source_category": "client",
                        "scope": ANALYSIS_SCOPE_TECHNICAL,
                        "component_type": "workstation",
                        "quantity": 2,
                        "price": 5,
                        "max_power": 80,
                        "client_seats": 2,
                    },
                ],
                "totals": {"capital_cost": 20, "client_seats": 2, "total_power_watts": 560},
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


def test_energy_presenter_ignores_source_catalog_until_hybrid_selection(tmp_path: Path):
    app = _app(tmp_path)
    app.replace_entities(
        {
            "server": [{"name": "srv", "quantity": 1, "price": 10, "max_power": 400}],
            "client": [{"name": "pc", "quantity": 2, "price": 5, "max_power": 80}],
            "licenses": [{"name": "os", "quantity": 1, "price": 1}],
        }
    )
    presenter = EnergyPresenter(app)

    assert presenter.table_rows() == []
    assert presenter.summary().total_power == 0.0


def test_energy_presenter_reads_only_selected_hybrid_equipment(tmp_path: Path):
    app = _app(tmp_path)
    app.replace_entities(
        {
            "server": [{"name": "unused-srv", "quantity": 50, "price": 10, "max_power": 1000}],
            "client": [{"name": "unused-pc", "quantity": 50, "price": 5, "max_power": 300}],
        }
    )
    _select_technical_candidate(app)
    presenter = EnergyPresenter(app)

    rows = presenter.table_rows()

    assert [row["name"] for row in rows] == ["srv", "pc"]
    assert rows[0]["mode"] == "24/7"
    assert presenter.summary().total_power == 560.0


def test_energy_presenter_calculates_selected_hybrid_result(tmp_path: Path):
    app = _app(tmp_path)
    _select_technical_candidate(app)
    presenter = EnergyPresenter(app)

    report = presenter.calculate(EnergyProfileInput(hours_per_day=10, working_days=20, cost_per_kwh=5))
    summary = presenter.summary()

    assert round(report["total_cost"], 2) == 1600.0
    assert round(summary.total_kwh, 2) == 320.0
    assert app.get_electricity_cost() == report["total_cost"]
    assert app.get_electricity_profile()["hours_per_day"] == 10.0


def test_energy_presenter_empty_state_is_safe(tmp_path: Path):
    presenter = EnergyPresenter(_app(tmp_path))

    report = presenter.calculate({"hours_per_day": 8, "working_days": 22, "cost_per_kwh": 1})

    assert report == {"total_cost": 0.0, "items": []}
    assert presenter.summary().item_count == 0
    assert presenter.summary().total_cost == 0.0
