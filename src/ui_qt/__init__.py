"""Parallel PySide6/Qt UI shell.

The package is intentionally independent from the current Tkinter UI.  Early
migration patches keep it importable even when PySide6 is not installed yet;
only actual Qt startup imports the PySide6 runtime.
"""

from __future__ import annotations

__all__ = ["QtDependencyError", "create_qt_application", "run_qt_app", "smoke_check"]

from ui_qt.app import QtDependencyError, create_qt_application, run_qt_app, smoke_check
