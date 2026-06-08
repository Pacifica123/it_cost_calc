from __future__ import annotations

from pathlib import Path


def test_main_window_keeps_lazy_screen_contract():
    source = Path("src/ui_qt/main_window.py").read_text(encoding="utf-8")

    assert "_screen_factories" in source
    assert "_screen_indexes" in source
    assert "startup_ms" in source
    assert "_apply_responsive_rules" in source
    assert "QShortcut" in source
