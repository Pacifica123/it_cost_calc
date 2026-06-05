from pathlib import Path

from ui_qt.presenters import GaInput, GaPresenter, QtAppPresenter


def _app(tmp_path: Path) -> QtAppPresenter:
    return QtAppPresenter(
        repo_root=Path(__file__).resolve().parents[2],
        runtime_entities_path=tmp_path / "runtime_entities.json",
    )


def test_ga_presenter_reports_empty_state(tmp_path: Path):
    app = _app(tmp_path)
    presenter = GaPresenter(app, scope="technical")

    summary = presenter.summary()

    assert summary.status == "Нет данных"
    assert summary.row_count == 0
    assert presenter.candidate_rows() == []


def test_ga_presenter_runs_technical_scope_and_updates_pool(tmp_path: Path):
    app = _app(tmp_path)
    app.replace_entities(
        {
            "server": [
                {
                    "name": "Server A",
                    "quantity": 1,
                    "price": 100,
                    "ram_gb": 16,
                    "cpu_cores": 4,
                    "storage_gb": 512,
                    "energy": 80,
                }
            ],
            "client": [
                {
                    "name": "Client A",
                    "quantity": 3,
                    "price": 50,
                    "ram_gb": 8,
                    "cpu_cores": 2,
                    "storage_gb": 256,
                    "energy": 30,
                    "client_seats": 3,
                }
            ],
            "network": [
                {
                    "name": "Router A",
                    "quantity": 1,
                    "price": 20,
                    "lan_ports": 8,
                    "lan_speed_mbps": 1000,
                    "ipv6_support": True,
                }
            ],
        }
    )
    presenter = GaPresenter(app, scope="technical")

    result = presenter.run(
        GaInput(
            population=10,
            generations=8,
            mutation_rate=0.02,
            seed=3,
            budget=1000,
            min_units=1,
        )
    )

    assert result["status"] == "ok"
    assert presenter.summary().status == "Готово"
    assert presenter.summary().pool_count >= 1
    assert presenter.candidate_rows()
    assert app.get_candidate_pool_snapshot("technical").source_method == "ga"


def test_ga_presenter_uses_software_scope_only(tmp_path: Path):
    app = _app(tmp_path)
    app.replace_entities(
        {
            "server": [{"name": "Server A", "quantity": 1, "price": 100}],
            "licenses": [
                {
                    "name": "License A",
                    "quantity": 5,
                    "price": 10,
                    "software_license_quantity": 5,
                    "functionality_score": 4,
                    "support_score": 4,
                    "reliability_score": 4,
                    "lifespan": 3,
                }
            ],
        }
    )
    presenter = GaPresenter(app, scope="software")

    result = presenter.run(
        GaInput(
            population=8,
            generations=5,
            mutation_rate=0.02,
            seed=5,
            budget=500,
            min_units=1,
        )
    )

    assert result["status"] == "ok"
    assert result["parameters"]["categories"] == ["licenses"]
    assert presenter.summary().label == "ПО"


def test_ga_screen_source_keeps_primary_action_outside_parameter_grid():
    source = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "ui_qt"
        / "screens"
        / "decision"
        / "ga_screen.py"
    ).read_text(encoding="utf-8")

    assert "layout = QVBoxLayout(frame)" in source
    assert "actions.addWidget(self.run_button, 0)" in source
    assert "layout.addWidget(self.run_button, 3, 3)" not in source
    assert "NoButtons" in source
