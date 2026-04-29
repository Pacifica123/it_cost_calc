"""Compatibility namespace for the historical ``it_cost_calc`` package.

The project now keeps a single canonical implementation in the top-level
``application``, ``domain``, ``infrastructure``, ``shared`` and ``ui`` packages.
This module exposes the same objects through ``it_cost_calc.*`` imports without
copying the source tree.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

__all__ = [
    "application",
    "domain",
    "infrastructure",
    "shared",
    "ui",
]


def _alias_public_package(name: str) -> ModuleType:
    module = importlib.import_module(name)
    legacy_name = f"{__name__}.{name}"
    sys.modules[legacy_name] = module
    globals()[name] = module
    return module


for _package_name in __all__:
    _alias_public_package(_package_name)

