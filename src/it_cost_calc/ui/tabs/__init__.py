"""Compatibility wrapper for ``ui.tabs`` with lazy exports."""

from __future__ import annotations

from types import ModuleType

import ui.tabs as _tabs

__all__ = list(_tabs.__all__)


def __getattr__(name: str) -> ModuleType:
    return getattr(_tabs, name)


def __dir__() -> list[str]:
    return sorted([*globals(), *__all__])
