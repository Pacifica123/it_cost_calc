from pathlib import Path


def test_yandex_market_collectors_are_cli_only_after_feed_first_gui_switch() -> None:
    screen_source = Path("src/ui_qt/screens/catalog_staging.py").read_text(encoding="utf-8")
    presenter_source = Path("src/ui_qt/presenters/catalog_staging_presenter.py").read_text(
        encoding="utf-8"
    )
    cli_source = Path("tools/catalog_parser/cli.py").read_text(encoding="utf-8")

    assert "CatalogFeedSourceDialog" in screen_source
    assert "Собрать из Яндекс Маркета" not in screen_source
    assert "HttpCatalogImportDialog" not in screen_source
    assert '"yandex-market-http-live"' in presenter_source
    assert '"yandex-market-live"' in cli_source
    assert '"yandex-market-http-live"' in cli_source
    assert '"yandex-market-har"' in cli_source
    assert '"yandex-market-html"' in cli_source
