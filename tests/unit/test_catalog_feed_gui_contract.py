from pathlib import Path


def test_catalog_screen_uses_generic_feed_ui_instead_of_store_collectors():
    source = Path("src/ui_qt/screens/catalog_staging.py").read_text(encoding="utf-8")

    assert "Источник данных" in source
    assert "Импорт файла" in source
    assert "Автообогащение" in source
    assert "Icecat (аккаунт)" in source
    assert "table_page" in source
    assert "Собрать из DNS" not in source
    assert "Собрать из Яндекс Маркета" not in source
