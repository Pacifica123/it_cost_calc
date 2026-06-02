"""Application-wide ttk styling for the desktop UI."""

from __future__ import annotations

from tkinter import ttk


BACKGROUND = "#f3f5f7"
SURFACE = "#ffffff"
PANEL = "#fbfcfd"
PANEL_HEADER = "#eef2f6"
PANEL_BORDER = "#d6dde5"
TEXT = "#1f2933"
MUTED_TEXT = "#5f6b7a"
ACCENT = "#2f6fdd"


def configure_app_style(root) -> None:
    """Apply a calm neutral visual hierarchy to Tk/ttk widgets.

    The previous default theme made every area pure white and rendered the
    collapsible-panel button as a large blue square on Windows.  The custom
    styles below keep controls standard, but separate the application
    background, tab surface and collapsible panel headers visually.
    """

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:  # noqa: BLE001 - fall back to the platform theme if unavailable
        pass

    root.configure(background=BACKGROUND)

    style.configure(".", font=("TkDefaultFont", 9), foreground=TEXT)
    style.configure("TFrame", background=BACKGROUND)
    style.configure("App.TFrame", background=BACKGROUND)
    style.configure("Surface.TFrame", background=SURFACE)
    style.configure("Panel.TFrame", background=PANEL, borderwidth=1, relief="solid")
    style.configure("PanelHeader.TFrame", background=PANEL_HEADER)
    style.configure("TLabel", background=BACKGROUND, foreground=TEXT)
    style.configure("Surface.TLabel", background=SURFACE, foreground=TEXT)
    style.configure("Panel.TLabel", background=PANEL, foreground=TEXT)
    style.configure("PanelHeader.TLabel", background=PANEL_HEADER, foreground=TEXT)
    style.configure("PanelTitle.TLabel", background=PANEL_HEADER, foreground=TEXT, font=("TkDefaultFont", 9, "bold"))
    style.configure("PanelHint.TLabel", background=PANEL_HEADER, foreground=MUTED_TEXT, font=("TkDefaultFont", 8))
    style.configure("Disclosure.TLabel", background=PANEL_HEADER, foreground=TEXT, font=("TkDefaultFont", 9, "bold"))
    style.configure("TLabelframe", background=BACKGROUND, bordercolor=PANEL_BORDER)
    style.configure("TLabelframe.Label", background=BACKGROUND, foreground=TEXT)
    style.configure("TNotebook", background=BACKGROUND, borderwidth=0)
    style.configure("TNotebook.Tab", padding=(8, 4))
    style.configure("TButton", padding=(8, 4))
    style.configure("Treeview", rowheight=22)
    style.configure("Treeview.Heading", font=("TkDefaultFont", 9, "bold"))


__all__ = [
    "BACKGROUND",
    "SURFACE",
    "PANEL",
    "PANEL_HEADER",
    "PANEL_BORDER",
    "TEXT",
    "MUTED_TEXT",
    "ACCENT",
    "configure_app_style",
]
