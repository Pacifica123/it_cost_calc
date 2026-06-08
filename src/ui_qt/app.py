from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence


class QtDependencyError(RuntimeError):
    """Raised when the optional Qt runtime is not available."""


def ensure_src_on_path(repo_root: Path | None = None) -> None:
    """Keep direct script execution compatible with the src-layout project."""

    root = repo_root or Path(__file__).resolve().parents[2]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _missing_qt_message() -> str:
    return "PySide6 is not installed. Install project dependencies first."


def _load_qapplication():
    try:
        from PySide6.QtWidgets import QApplication
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            raise QtDependencyError(_missing_qt_message()) from exc
        raise
    return QApplication


def create_qt_application(argv: Sequence[str] | None = None):
    """Create or return the active QApplication instance."""

    ensure_src_on_path()
    QApplication = _load_qapplication()
    app = QApplication.instance()
    if app is None:
        app = QApplication(list(argv or sys.argv))
    app.setApplicationName("IT Cost Calc")
    app.setOrganizationName("Diploma project workspace")
    return app


def create_main_window():
    """Create the minimal Qt main window for the migration shell."""

    ensure_src_on_path()
    from ui_qt.main_window import QtMainWindow

    return QtMainWindow()


def run_qt_app(argv: Sequence[str] | None = None, *, show: bool = True) -> int:
    """Run the Qt UI shell."""

    app = create_qt_application(argv)
    window = create_main_window()
    if show:
        window.show()
    return int(app.exec())


def smoke_check(argv: Sequence[str] | None = None) -> int:
    """Create QApplication and the main window without showing it."""

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = create_qt_application(argv or ["qt-smoke-check"])
    window = create_main_window()
    window.resize(640, 420)
    window.deleteLater()
    app.processEvents()
    return 0
