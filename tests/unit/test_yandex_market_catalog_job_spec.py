from pathlib import Path

from ui_qt.presenters import CatalogStagingPresenter, QtAppPresenter


def test_yandex_market_job_spec_runs_unbuffered_cli_in_separate_process(
    tmp_path: Path,
) -> None:
    app = QtAppPresenter(
        repo_root=tmp_path,
        runtime_entities_path=tmp_path / "runtime.json",
    )
    presenter = CatalogStagingPresenter(app, staging_path=tmp_path / "staging.json")
    job = presenter.build_yandex_market_job(
        categories=["routers", "servers"],
        per_category_limit=7,
        time_limit_seconds=180,
        visible_browser=True,
        browser_engine="firefox",
        region="Москва",
    )

    assert job.working_directory == tmp_path
    assert job.output_path.name == "equipment_catalog.json"
    assert job.snapshot_path.name == "snapshot"
    assert job.arguments[:4] == (
        "-u",
        str(tmp_path / "scripts" / "update_equipment_catalog.py"),
        "--mode",
        "yandex-market-live",
    )
    assert "routers,servers" in job.arguments
    assert str(
        tmp_path / "data/generated/catalog/yandex_market_browser_profiles/firefox"
    ) in job.arguments
    assert "--headless" not in job.arguments
    assert "Москва" in job.arguments
    assert presenter.yandex_market_browser_url("routers") == (
        "https://market.yandex.ru/category/routery"
    )

    capture_path = tmp_path / "browser.har"
    capture_path.write_text("{}", encoding="utf-8")
    capture_job = presenter.build_yandex_market_capture_job(
        capture_path,
        region="Кемерово",
    )
    assert "yandex-market-har" in capture_job.arguments
    assert str(capture_path) in capture_job.arguments
    assert "Кемерово" in capture_job.arguments
    assert capture_job.output_path.name == "equipment_catalog.json"
