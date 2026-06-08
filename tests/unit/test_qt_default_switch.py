from __future__ import annotations

import ast
from pathlib import Path


def _imports_module(source: str, module_name: str) -> bool:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == module_name for alias in node.names):
                return True
        if isinstance(node, ast.ImportFrom) and node.module == module_name:
            return True
    return False


def test_default_app_module_does_not_import_tkinter():
    source = Path("src/app.py").read_text(encoding="utf-8")

    assert not _imports_module(source, "tkinter")
    assert "ui_qt.app" in source
    assert "from ui_legacy" not in source


def test_bootstrap_default_main_runs_qt_and_legacy_is_lazy():
    source = Path("src/bootstrap.py").read_text(encoding="utf-8")

    assert "def main(" in source
    assert "return main_qt(argv)" in source
    assert "def main_legacy_tk" in source
    assert "from ui_legacy.app import CalculatorApp" in source
    assert not source.lstrip().startswith("import tkinter")


def test_run_app_supports_qt_default_and_legacy_flag():
    source = Path("scripts/run_app.py").read_text(encoding="utf-8")

    assert "--legacy-tk" in source
    assert "--smoke-check" in source
    assert "return app_main()" in source
    assert "return main_legacy_tk()" in source


def test_legacy_app_is_archived_separately():
    source = Path("src/ui_legacy/app.py").read_text(encoding="utf-8")

    assert "class CalculatorApp" in source
    assert "import tkinter as tk" in source
