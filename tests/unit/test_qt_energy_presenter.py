from pathlib import Path

from ui_qt.presenters import EnergyPresenter, EnergyProfileInput, QtAppPresenter


def _app(tmp_path: Path) -> QtAppPresenter:
    return QtAppPresenter(
        repo_root=Path(__file__).resolve().parents[2],
        runtime_entities_path=tmp_path / "runtime_entities.json",
    )


def test_energy_presenter_reads_only_power_equipment(tmp_path: Path):
    app = _app(tmp_path)
    app.replace_entities(
        {
            "server": [{"name": "srv", "quantity": 1, "price": 10, "max_power": 400}],
            "client": [{"name": "pc", "quantity": 2, "price": 5, "max_power": 80}],
            "licenses": [{"name": "os", "quantity": 1, "price": 1}],
        }
    )
    presenter = EnergyPresenter(app)

    rows = presenter.table_rows()

    assert [row["name"] for row in rows] == ["srv", "pc"]
    assert rows[0]["mode"] == "24/7"
    assert presenter.summary().total_power == 560.0


def test_energy_presenter_calculates_and_publishes_result(tmp_path: Path):
    app = _app(tmp_path)
    app.replace_entities(
        {
            "server": [{"name": "srv", "quantity": 1, "price": 10, "max_power": 1000}],
            "client": [{"name": "pc", "quantity": 2, "price": 5, "max_power": 100}],
        }
    )
    presenter = EnergyPresenter(app)

    report = presenter.calculate(EnergyProfileInput(hours_per_day=10, working_days=20, cost_per_kwh=5))
    summary = presenter.summary()

    assert round(report["total_cost"], 2) == 3800.0
    assert summary.total_kwh == 760.0
    assert app.get_electricity_cost() == report["total_cost"]
    assert app.get_electricity_profile()["hours_per_day"] == 10.0


def test_energy_presenter_empty_state_is_safe(tmp_path: Path):
    presenter = EnergyPresenter(_app(tmp_path))

    report = presenter.calculate({"hours_per_day": 8, "working_days": 22, "cost_per_kwh": 1})

    assert report == {"total_cost": 0.0, "items": []}
    assert presenter.summary().item_count == 0
    assert presenter.summary().total_cost == 0.0
