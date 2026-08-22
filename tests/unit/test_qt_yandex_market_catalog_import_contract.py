from pathlib import Path


def test_yandex_market_catalog_gui_uses_shared_cancellable_process() -> None:
    dialog_source = Path("src/ui_qt/dialogs/dns_catalog_import_dialog.py").read_text(
        encoding="utf-8"
    )
    market_dialog_source = Path(
        "src/ui_qt/dialogs/yandex_market_catalog_import_dialog.py"
    ).read_text(encoding="utf-8")
    screen_source = Path("src/ui_qt/screens/catalog_staging.py").read_text(encoding="utf-8")
    presenter_source = Path("src/ui_qt/presenters/catalog_staging_presenter.py").read_text(
        encoding="utf-8"
    )

    assert "QProcess" in dialog_source
    assert "terminate()" in dialog_source
    assert 'source="yandex_market"' in market_dialog_source
    assert "Собрать из Яндекс Маркета" in screen_source
    assert "YandexMarketCatalogImportDialog" in screen_source
    assert "build_yandex_market_job" in presenter_source
    assert "build_yandex_market_capture_job" in presenter_source
    assert "yandex-market-live" in presenter_source
    assert "yandex-market-har" in presenter_source
