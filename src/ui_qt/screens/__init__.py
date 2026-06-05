from __future__ import annotations

_SCREEN_MODULES = {
    "ComponentEditorScreen": "ui_qt.screens.component_editor",
    "EnergyScreen": "ui_qt.screens.energy",
    "ExportScreen": "ui_qt.screens.export",
    "NpvScreen": "ui_qt.screens.npv",
    "PlaceholderScreen": "ui_qt.screens.placeholder_screen",
    "SoftwareWorkspaceScreen": "ui_qt.screens.software_workspace",
    "TechnicalWorkspaceScreen": "ui_qt.screens.technical_workspace",
}

__all__ = [*_SCREEN_MODULES]


def __getattr__(name: str):
    if name not in _SCREEN_MODULES:
        raise AttributeError(name)
    from importlib import import_module

    module = import_module(_SCREEN_MODULES[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
