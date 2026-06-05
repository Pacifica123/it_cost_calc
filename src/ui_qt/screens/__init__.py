from __future__ import annotations

_SCREEN_MODULES = {
    "ComponentEditorScreen": "ui_qt.screens.component_editor",
    "PlaceholderScreen": "ui_qt.screens.placeholder_screen",
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
