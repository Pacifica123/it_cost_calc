from pathlib import Path


def test_catalog_screen_exposes_compact_p456_actions() -> None:
    root = Path(__file__).resolve().parents[2]
    screen = (root / "src/ui_qt/screens/catalog_staging.py").read_text(encoding="utf-8")
    assert 'QPushButton("Доп. данные"' in screen
    assert 'addAction("Импорт КП")' in screen
    assert 'addAction("Бенчмарк ЕИС")' in screen
    assert 'addAction("Захват браузера")' in screen
    assert "CatalogCommercialQuoteDialog" in screen
    assert "CatalogProcurementBenchmarkDialog" in screen
    assert "CatalogBrowserCaptureDialog" in screen
