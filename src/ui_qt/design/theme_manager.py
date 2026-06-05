from __future__ import annotations

from pathlib import Path
from typing import Any

from ui_qt.design.tokens import DEFAULT_THEME, THEMES, ThemeName, normalize_theme_name

_STYLE_FILES: dict[ThemeName, str] = {
    "light": "styles_light.qss",
    "dark": "styles_dark.qss",
}


class ThemeManager:
    """Apply a soft flat Qt theme without coupling callers to QSS paths."""

    def __init__(self, theme_name: str | None = None) -> None:
        self._theme_name = normalize_theme_name(theme_name)

    @property
    def theme_name(self) -> ThemeName:
        return self._theme_name

    @property
    def tokens(self):
        return THEMES[self._theme_name]

    def set_theme(self, theme_name: str | None) -> None:
        self._theme_name = normalize_theme_name(theme_name)

    def stylesheet(self, theme_name: str | None = None) -> str:
        name = normalize_theme_name(theme_name or self._theme_name)
        path = Path(__file__).with_name(_STYLE_FILES[name])
        return path.read_text(encoding="utf-8")

    def apply_to_application(self, app: Any, theme_name: str | None = None) -> ThemeName:
        name = normalize_theme_name(theme_name or self._theme_name or DEFAULT_THEME)
        self._theme_name = name
        app.setStyleSheet(self.stylesheet(name))
        app.setProperty("itCostTheme", name)
        return name
