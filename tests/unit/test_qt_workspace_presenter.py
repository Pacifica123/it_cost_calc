from pathlib import Path

from ui_qt.presenters import QtAppPresenter, WorkspacePresenter


def _app(tmp_path: Path) -> QtAppPresenter:
    return QtAppPresenter(
        repo_root=Path(__file__).resolve().parents[2],
        runtime_entities_path=tmp_path / "runtime_entities.json",
    )


def test_workspace_presenter_filters_technical_categories(tmp_path: Path):
    app = _app(tmp_path)
    app.replace_entities(
        {
            "server": [{"name": "srv", "quantity": 2, "price": 100}],
            "client": [{"name": "pc", "quantity": 1, "price": 50}],
            "licenses": [{"name": "os", "quantity": 3, "price": 10}],
            "server_administration": [{"name": "admin", "monthly_cost": 20}],
            "subscription_licenses": [{"name": "sub", "monthly_cost": 5}],
        }
    )
    presenter = WorkspacePresenter(app, scope="technical")

    summary = presenter.summary()
    capex_rows = presenter.capex_table_rows()
    opex_rows = presenter.opex_table_rows()

    assert summary.label == "ТО"
    assert summary.capex_total == 250
    assert summary.opex_monthly == 20
    assert summary.row_count == 3
    assert {row["category"] for row in capex_rows} == {"Серверы", "Клиенты"}
    assert opex_rows[0]["category"] == "Администрирование"


def test_workspace_presenter_filters_software_categories(tmp_path: Path):
    app = _app(tmp_path)
    app.replace_entities(
        {
            "server": [{"name": "srv", "quantity": 1, "price": 100}],
            "licenses": [{"name": "os", "quantity": 3, "price": 10}],
            "migration": [{"name": "mig", "one_time_cost": 40}],
            "subscription_licenses": [{"name": "sub", "monthly_cost": 5}],
        }
    )
    presenter = WorkspacePresenter(app, scope="software")

    summary = presenter.summary()

    assert summary.label == "ПО"
    assert summary.capex_total == 30
    assert summary.opex_monthly == 5
    assert summary.opex_once == 40
    assert summary.row_count == 3
    assert presenter.capex_table_rows()[0]["category"] == "Лицензии"


def test_workspace_presenter_category_rows_are_compact(tmp_path: Path):
    app = _app(tmp_path)
    app.replace_entities({"licenses": [{"name": "os", "quantity": 1, "price": 10}]})
    presenter = WorkspacePresenter(app, scope="software")

    rows = presenter.category_rows()

    assert rows[0] == {
        "group": "CAPEX",
        "category": "Лицензии",
        "items": 1,
        "total": 10,
    }
    assert presenter.has_data()
