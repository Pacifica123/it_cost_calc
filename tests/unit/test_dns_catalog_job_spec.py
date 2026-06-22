from pathlib import Path

from ui_qt.presenters import CatalogStagingPresenter, QtAppPresenter


def test_dns_job_spec_runs_unbuffered_cli_in_separate_process(tmp_path: Path) -> None:
    app = QtAppPresenter(
        repo_root=tmp_path,
        runtime_entities_path=tmp_path / "runtime.json",
    )
    presenter = CatalogStagingPresenter(app, staging_path=tmp_path / "staging.json")
    job = presenter.build_dns_job(
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
    assert job.arguments[:3] == (
        "-u",
        str(tmp_path / "scripts" / "update_equipment_catalog.py"),
        "--mode",
    )
    assert "routers,servers" in job.arguments
    assert job.arguments[job.arguments.index("--browser-engine") + 1] == "firefox"
    assert str(tmp_path / "data/generated/catalog/dns_browser_profiles/firefox") in job.arguments
    assert "--headless" not in job.arguments
    assert "Москва" in job.arguments
    assert presenter.dns_browser_url("routers") == (
        "https://www.dns-shop.ru/catalog/17a8aa1c16404e77/wi-fi-routery/"
    )
