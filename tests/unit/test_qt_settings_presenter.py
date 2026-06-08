from __future__ import annotations

import json

from ui_qt.presenters.settings_presenter import QtSettingsPresenter, QtUiSettings, allowed_scales


def test_qt_settings_defaults_are_safe_and_short():
    settings = QtUiSettings()

    assert settings.theme == "light"
    assert settings.ui_scale == 1.0
    assert settings.start_route == "software"
    assert settings.show_advanced is False
    assert 1.0 in allowed_scales()


def test_qt_settings_normalize_invalid_payload():
    settings = QtUiSettings.from_mapping(
        {
            "theme": "neon",
            "ui_scale": 7,
            "start_route": "missing",
            "show_advanced": 1,
        }
    )

    assert settings.theme == "light"
    assert settings.ui_scale == 1.25
    assert settings.start_route == "software"
    assert settings.show_advanced is True


def test_qt_settings_presenter_persists_json(tmp_path):
    path = tmp_path / "settings.json"
    presenter = QtSettingsPresenter(path=path)

    saved = presenter.update(theme="dark", ui_scale=1.1, start_route="export")

    assert saved.theme == "dark"
    assert json.loads(path.read_text(encoding="utf-8"))["start_route"] == "export"
    assert QtSettingsPresenter(path=path).settings == saved
