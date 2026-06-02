"""Публичные точки входа для вкладок интерфейса.

Модуль намеренно не импортирует все вкладки при загрузке пакета. Часть вкладок
имеет тяжёлые необязательные зависимости интерфейса, например matplotlib для NPV.
Ленивые экспорты сохраняют совместимость со старыми импортами и не ломают
пакетные unit-тесты в окружениях без GUI-зависимостей.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

_MODULE_EXPORTS = {
    "capex_tab": "ui.tabs.capex_tab",
    "configuration_selection_tab": "ui.tabs.configuration_selection_tab",
    "criteria_importance_tab": "ui.tabs.criteria_importance_tab",
    "energy_tab": "ui.tabs.energy_tab",
    "export_tab": "ui.tabs.export_tab",
    "genetic_optimization_tab": "ui.tabs.genetic_optimization_tab",
    "infrastructure_tab": "ui.tabs.infrastructure_tab",
    "solution_component_editor_tab": "ui.tabs.solution_component_editor_tab",
    "solution_area_workspace_tab": "ui.tabs.solution_area_workspace_tab",
    "npv_tab": "ui.tabs.npv_tab",
    "opex_tab": "ui.tabs.opex_tab",
}

_COMPAT_EXPORTS = {
    "ahp_analysis_tab": "configuration_selection_tab",
    "ahp_tab": "configuration_selection_tab",
    "capital_costs_tab": "capex_tab",
    "capital_costs": "capex_tab",
    "electricity_costs_tab": "energy_tab",
    "electricity_costs": "energy_tab",
    "it_infrastructure": "infrastructure_tab",
    "npv_analysis": "npv_tab",
    "operational_costs_tab": "opex_tab",
    "operational_costs": "opex_tab",
}

__all__ = [*_MODULE_EXPORTS, *_COMPAT_EXPORTS]


def _load_export(name: str) -> ModuleType:
    module_path = _MODULE_EXPORTS[name]
    module = import_module(module_path)
    globals()[name] = module
    return module


def __getattr__(name: str) -> ModuleType:
    if name in _MODULE_EXPORTS:
        return _load_export(name)

    if name in _COMPAT_EXPORTS:
        module = _load_export(_COMPAT_EXPORTS[name])
        globals()[name] = module
        return module

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*globals(), *__all__])
