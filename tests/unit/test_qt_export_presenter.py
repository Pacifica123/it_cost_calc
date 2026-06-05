from pathlib import Path

from ui_qt.presenters import ExportPresenter, QtAppPresenter


def _app(tmp_path: Path) -> QtAppPresenter:
    return QtAppPresenter(
        repo_root=tmp_path,
        runtime_entities_path=tmp_path / "data" / "generated" / "runtime_entities.json",
    )


def test_export_presenter_builds_decision_report_files(tmp_path: Path):
    app = _app(tmp_path)
    app.replace_entities(
        {
            "server": [{"name": "srv", "quantity": 1, "price": 1000, "max_power": 100}],
            "server_rental": [{"name": "rent", "monthly_cost": 100}],
        }
    )
    presenter = ExportPresenter(app)

    paths = presenter.export("full_decision_report")

    assert {"json", "markdown", "csv", "components_csv"} <= set(paths)
    assert all(path.exists() for path in paths.values())
    assert presenter.summary().last_files == 4
    assert "Сводка экспорта" in presenter.dashboard_text()


def test_export_presenter_exports_partial_markdown_and_csv(tmp_path: Path):
    app = _app(tmp_path)
    app.replace_entities({"server": [{"name": "srv", "quantity": 2, "price": 10}]})
    presenter = ExportPresenter(app)

    cost_paths = presenter.export("cost_summary")
    csv_paths = presenter.export("raw_costs_csv")

    assert cost_paths["markdown"].read_text(encoding="utf-8").startswith("# Сводка затрат")
    assert csv_paths["csv"].exists()
    assert presenter.path_rows(csv_paths)[0]["kind"] == "csv"


def test_export_presenter_rejects_unknown_mode(tmp_path: Path):
    presenter = ExportPresenter(_app(tmp_path))

    try:
        presenter.export("bad")
    except ValueError as exc:
        assert "Unknown export mode" in str(exc)
    else:  # pragma: no cover - defensive branch
        raise AssertionError("unknown export mode accepted")
