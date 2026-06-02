"""Application-wide ttk styling for the desktop UI."""

from __future__ import annotations

from tkinter import ttk


BACKGROUND = "#d8d2c7"
WORKSPACE = "#e4dfd6"
SURFACE = "#eeeae2"
PANEL = "#faf9f6"
PANEL_BODY = "#f5f3ee"
PANEL_HEADER = "#ede6db"
PANEL_BORDER = "#bdb6aa"
SECTION = "#f7f5f0"
TEXT = "#1f2933"
MUTED_TEXT = "#5b6675"
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
    style.configure("Workspace.TFrame", background=WORKSPACE)
    style.configure("WorkspaceHeader.TFrame", background=WORKSPACE)
    style.configure("Surface.TFrame", background=SURFACE)
    style.configure("Section.TFrame", background=SECTION)
    style.configure("Panel.TFrame", background=PANEL, borderwidth=1, relief="solid")
    style.configure("PanelBody.TFrame", background=PANEL_BODY)
    style.configure("PanelHeader.TFrame", background=PANEL_HEADER)
    style.configure("TLabel", background=BACKGROUND, foreground=TEXT)
    style.configure("App.TLabel", background=BACKGROUND, foreground=TEXT)
    style.configure("Workspace.TLabel", background=WORKSPACE, foreground=TEXT)
    style.configure("WorkspaceHeader.TLabel", background=WORKSPACE, foreground=TEXT)
    style.configure("Surface.TLabel", background=SURFACE, foreground=TEXT)
    style.configure("Panel.TLabel", background=PANEL, foreground=TEXT)
    style.configure("PanelBody.TLabel", background=PANEL_BODY, foreground=TEXT)
    style.configure("PanelHeader.TLabel", background=PANEL_HEADER, foreground=TEXT)
    style.configure("PanelTitle.TLabel", background=PANEL_HEADER, foreground=TEXT, font=("TkDefaultFont", 9, "bold"))
    style.configure("PanelHint.TLabel", background=PANEL_HEADER, foreground=MUTED_TEXT, font=("TkDefaultFont", 8))
    style.configure("Disclosure.TLabel", background=PANEL_HEADER, foreground=TEXT, font=("TkDefaultFont", 9, "bold"))
    style.configure("TLabelframe", background=SECTION, bordercolor=PANEL_BORDER)
    style.configure("TLabelframe.Label", background=SECTION, foreground=TEXT)
    style.configure("Panel.TLabelframe", background=PANEL_BODY, bordercolor=PANEL_BORDER)
    style.configure("Panel.TLabelframe.Label", background=PANEL_BODY, foreground=TEXT)
    style.configure("TPanedwindow", background=BACKGROUND)
    style.configure("Workspace.TPanedwindow", background=WORKSPACE)
    style.configure("TNotebook", background=BACKGROUND, borderwidth=0)
    style.configure("TNotebook.Tab", padding=(8, 4))
    style.configure("TButton", padding=(8, 4))
    style.configure("Treeview", rowheight=22, background="#fbfbf8", fieldbackground="#fbfbf8")
    style.configure("Treeview.Heading", background="#eeeae2", font=("TkDefaultFont", 9, "bold"))


__all__ = [
    "BACKGROUND",
    "WORKSPACE",
    "SURFACE",
    "PANEL",
    "PANEL_BODY",
    "PANEL_HEADER",
    "PANEL_BORDER",
    "SECTION",
    "TEXT",
    "MUTED_TEXT",
    "ACCENT",
    "configure_app_style",
]
