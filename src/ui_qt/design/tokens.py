from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ThemeName = Literal["light", "dark"]


@dataclass(frozen=True)
class ColorTokens:
    background: str
    surface: str
    surface_alt: str
    border: str
    text: str
    text_muted: str
    accent: str
    accent_hover: str
    accent_text: str
    danger: str


@dataclass(frozen=True)
class FontTokens:
    family: str = "Segoe UI, Arial"
    base_pt: int = 10
    title_pt: int = 16
    table_pt: int = 9
    status_pt: int = 9


@dataclass(frozen=True)
class LayoutTokens:
    radius_sm: int = 6
    radius_md: int = 10
    radius_lg: int = 14
    gap_xs: int = 4
    gap_sm: int = 8
    gap_md: int = 12
    gap_lg: int = 16
    page_margin: int = 18
    button_height: int = 32
    input_height: int = 30


@dataclass(frozen=True)
class ThemeTokens:
    name: ThemeName
    colors: ColorTokens
    fonts: FontTokens = FontTokens()
    layout: LayoutTokens = LayoutTokens()


LIGHT_THEME = ThemeTokens(
    name="light",
    colors=ColorTokens(
        background="#f6f8fb",
        surface="#ffffff",
        surface_alt="#eef3f8",
        border="#d8e0ea",
        text="#17212f",
        text_muted="#607086",
        accent="#3b78e7",
        accent_hover="#2f66c8",
        accent_text="#ffffff",
        danger="#c2413b",
    ),
)

DARK_THEME = ThemeTokens(
    name="dark",
    colors=ColorTokens(
        background="#1d222b",
        surface="#252b35",
        surface_alt="#303744",
        border="#3e4654",
        text="#edf2f7",
        text_muted="#b6c0cf",
        accent="#7aa7ff",
        accent_hover="#94b8ff",
        accent_text="#111827",
        danger="#ff8a80",
    ),
)

THEMES: dict[ThemeName, ThemeTokens] = {
    "light": LIGHT_THEME,
    "dark": DARK_THEME,
}

DEFAULT_THEME: ThemeName = "light"


def normalize_theme_name(value: str | None) -> ThemeName:
    if value == "dark":
        return "dark"
    return DEFAULT_THEME
