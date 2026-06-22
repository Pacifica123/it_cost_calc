from pathlib import Path


def test_dns_catalog_gui_uses_separate_cancellable_process() -> None:
    dialog_source = Path("src/ui_qt/dialogs/dns_catalog_import_dialog.py").read_text(
        encoding="utf-8"
    )
    screen_source = Path("src/ui_qt/screens/catalog_staging.py").read_text(encoding="utf-8")

    assert "QProcess" in dialog_source
    assert "readyReadStandardOutput" in dialog_source
    assert "terminate()" in dialog_source
    assert "_close_after_stop" in dialog_source
    assert "Загрузить в staging" in dialog_source
    assert "Собрать из DNS" in screen_source
    assert "dialog.catalog_path" in screen_source
    assert "DNS запретил доступ. Подробности" in dialog_source
    assert "QComboBox" in dialog_source
    assert "установится автоматически" in dialog_source
    assert 'process_environment.insert("PYTHONUTF8", "1")' in dialog_source
    assert "getincrementaldecoder" in dialog_source
