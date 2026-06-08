from __future__ import annotations

from collections.abc import Sequence


def create_app():
    """Create the default Qt main window.

    The historical Tkinter implementation was moved to ``ui_legacy.app`` and
    is no longer imported by the normal runtime path.
    """

    from ui_qt.app import create_main_window

    return create_main_window()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the default Qt UI."""

    from bootstrap import main as bootstrap_main

    return bootstrap_main(argv)


__all__ = ["create_app", "main"]
