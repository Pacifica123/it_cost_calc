from pathlib import Path


def test_dns_catalog_gui_uses_http_collector_and_hides_playwright_flow() -> None:
    dialog_source = Path("src/ui_qt/dialogs/http_catalog_import_dialog.py").read_text(
        encoding="utf-8"
    )
    screen_source = Path("src/ui_qt/screens/catalog_staging.py").read_text(encoding="utf-8")
    presenter_source = Path("src/ui_qt/presenters/catalog_staging_presenter.py").read_text(
        encoding="utf-8"
    )
    cli_source = Path("tools/catalog_parser/cli.py").read_text(encoding="utf-8")

    assert "QProcess" in dialog_source
    assert "readyReadStandardOutput" in dialog_source
    assert "terminate()" in dialog_source
    assert "build_dns_http_job" in dialog_source
    assert "build_dns_job" not in dialog_source
    assert "browser_engine" not in dialog_source
    assert "visible_browser" not in dialog_source
    assert "Собрать из DNS (HTTP)" in screen_source
    assert "HttpCatalogImportDialog" in screen_source
    assert "DnsCatalogImportDialog" not in screen_source
    assert '"dns-http-live"' in presenter_source
    assert '"dns-live"' in cli_source
    assert '"dns-http-live"' in cli_source
    assert "Импорт HAR / HTML" in dialog_source
    assert "Открыть диагностику" in dialog_source
