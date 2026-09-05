from pathlib import Path


def test_yandex_market_catalog_gui_uses_http_collector_and_hides_playwright_flow() -> None:
    dialog_source = Path("src/ui_qt/dialogs/http_catalog_import_dialog.py").read_text(
        encoding="utf-8"
    )
    screen_source = Path("src/ui_qt/screens/catalog_staging.py").read_text(encoding="utf-8")
    presenter_source = Path("src/ui_qt/presenters/catalog_staging_presenter.py").read_text(
        encoding="utf-8"
    )
    cli_source = Path("tools/catalog_parser/cli.py").read_text(encoding="utf-8")

    assert "QProcess" in dialog_source
    assert "terminate()" in dialog_source
    assert "build_yandex_market_http_job" in dialog_source
    assert "build_yandex_market_job" not in dialog_source
    assert "Собрать из Яндекс Маркета (HTTP)" in screen_source
    assert 'source="yandex_market"' in screen_source
    assert "YandexMarketCatalogImportDialog" not in screen_source
    assert '"yandex-market-http-live"' in presenter_source
    assert '"yandex-market-live"' in cli_source
    assert '"yandex-market-http-live"' in cli_source
