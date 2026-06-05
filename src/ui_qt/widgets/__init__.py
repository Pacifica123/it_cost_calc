from __future__ import annotations

_WIDGET_MODULES = {
    "ActionBar": "ui_qt.widgets.action_bar",
    "CollapsibleSection": "ui_qt.widgets.collapsible_section",
    "CompactLabel": "ui_qt.widgets.compact_label",
    "EmptyState": "ui_qt.widgets.empty_state",
    "InfoHint": "ui_qt.widgets.info_hint",
    "SettingsPanel": "ui_qt.widgets.settings_panel",
    "StatusStrip": "ui_qt.widgets.status_strip",
}

__all__ = [*_WIDGET_MODULES]


def __getattr__(name: str):
    if name not in _WIDGET_MODULES:
        raise AttributeError(name)
    from importlib import import_module

    module = import_module(_WIDGET_MODULES[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
