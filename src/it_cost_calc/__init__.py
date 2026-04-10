"""Совместимость со старым пространством имён `it_cost_calc`."""

from __future__ import annotations

import importlib
import pkgutil
import sys
from types import ModuleType


def _alias_module(legacy_name: str, real_name: str) -> ModuleType:
    module = importlib.import_module(real_name)
    sys.modules[legacy_name] = module
    return module


def _alias_package_tree(legacy_name: str, real_name: str) -> None:
    package = _alias_module(legacy_name, real_name)
    package_path = getattr(package, "__path__", None)
    if not package_path:
        return

    for module_info in pkgutil.walk_packages(package_path, prefix=f"{real_name}."):
        real_child_name = module_info.name
        legacy_child_name = f"{legacy_name}{real_child_name[len(real_name):]}"
        _alias_module(legacy_child_name, real_child_name)


for legacy_name, real_name in {
    "it_cost_calc.app": "app",
    "it_cost_calc.bootstrap": "bootstrap",
    "it_cost_calc.application": "application",
    "it_cost_calc.domain": "domain",
    "it_cost_calc.infrastructure": "infrastructure",
    "it_cost_calc.shared": "shared",
    "it_cost_calc.ui": "ui",
}.items():
    if real_name in {"application", "domain", "infrastructure", "shared", "ui"}:
        _alias_package_tree(legacy_name, real_name)
    else:
        _alias_module(legacy_name, real_name)
