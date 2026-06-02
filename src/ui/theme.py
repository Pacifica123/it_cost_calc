"""Application-wide styling helpers for the desktop UI."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


# The palette is intentionally stronger than the platform default.  On Windows
# some native ttk widgets can briefly repaint themselves with white backgrounds;
# these values are applied both through ttk styles and, for Tk containers, as
# direct widget options.
BACKGROUND = "#d2cabd"          # whole application shell
WORKSPACE = "#ddd7cc"           # tab/workspace surface
SURFACE = "#e7e1d6"             # secondary surface around panels
PANEL = "#eee8de"               # panel container
PANEL_BODY = "#f3eee5"          # panel content area
PANEL_HEADER = "#e4d9c9"        # collapsible header
PANEL_BORDER = "#b1a796"
SECTION = "#f0ebe2"
TABLE_BG = "#fbfaf6"
TEXT = "#1f2933"
MUTED_TEXT = "#5b6675"
ACCENT = "#2f6fdd"


_PANEL_TTK_STYLES = {
    "TFrame": "PanelBody.TFrame",
    "TLabel": "PanelBody.TLabel",
    "TLabelframe": "Panel.TLabelframe",
    "TCheckbutton": "PanelBody.TCheckbutton",
    "TRadiobutton": "PanelBody.TRadiobutton",
    "TNotebook": "PanelBody.TNotebook",
}


def configure_app_style(root) -> None:
    """Apply the neutral visual hierarchy to Tk/ttk widgets."""

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
    style.configure(
        "PanelTitle.TLabel",
        background=PANEL_HEADER,
        foreground=TEXT,
        font=("TkDefaultFont", 9, "bold"),
    )
    style.configure(
        "PanelHint.TLabel",
        background=PANEL_HEADER,
        foreground=MUTED_TEXT,
        font=("TkDefaultFont", 8),
    )
    style.configure(
        "Disclosure.TLabel",
        background=PANEL_HEADER,
        foreground=TEXT,
        font=("TkDefaultFont", 9, "bold"),
    )

    style.configure("TLabelframe", background=SECTION, bordercolor=PANEL_BORDER)
    style.configure("TLabelframe.Label", background=SECTION, foreground=TEXT)
    style.configure("Panel.TLabelframe", background=PANEL_BODY, bordercolor=PANEL_BORDER)
    style.configure("Panel.TLabelframe.Label", background=PANEL_BODY, foreground=TEXT)
    style.configure("PanelBody.TCheckbutton", background=PANEL_BODY, foreground=TEXT)
    style.map("PanelBody.TCheckbutton", background=[("active", PANEL_BODY)])
    style.configure("PanelBody.TRadiobutton", background=PANEL_BODY, foreground=TEXT)
    style.map("PanelBody.TRadiobutton", background=[("active", PANEL_BODY)])

    style.configure("TPanedwindow", background=BACKGROUND)
    style.configure("Workspace.TPanedwindow", background=WORKSPACE)
    style.configure("TNotebook", background=BACKGROUND, borderwidth=0)
    style.configure("PanelBody.TNotebook", background=PANEL_BODY, borderwidth=0)
    style.configure("TNotebook.Tab", padding=(8, 4))
    style.configure("TButton", padding=(8, 4))
    style.configure("Treeview", rowheight=22, background=TABLE_BG, fieldbackground=TABLE_BG)
    style.configure("Treeview.Heading", background=SECTION, font=("TkDefaultFont", 9, "bold"))


def force_panel_backgrounds(widget: tk.Misc) -> None:
    """Force panel colours on an already created subtree.

    The UI mixes classic Tk widgets, ttk widgets and composite widgets.  A ttk
    theme can be applied before children are created, but on Windows native
    redraws may still leave newly created subwidgets with the default white
    surface.  This helper is called after composite panels are assembled and
    assigns direct Tk backgrounds plus explicit ttk styles where this is safe.
    """

    widget_class = widget.winfo_class()
    if isinstance(widget, (tk.Frame, tk.LabelFrame)):
        _safe_configure(widget, background=PANEL_BODY)
    elif isinstance(widget, tk.Canvas):
        _safe_configure(widget, background=PANEL_BODY, highlightthickness=0)
    elif isinstance(widget, tk.Label):
        _safe_configure(widget, background=PANEL_BODY, foreground=TEXT)

    ttk_style = _PANEL_TTK_STYLES.get(widget_class)
    if ttk_style:
        _safe_configure(widget, style=ttk_style)

    for child in widget.winfo_children():
        force_panel_backgrounds(child)


def _safe_configure(widget: tk.Misc, **kwargs) -> None:
    try:
        widget.configure(**kwargs)
    except tk.TclError:
        return


__all__ = [
    "BACKGROUND",
    "WORKSPACE",
    "SURFACE",
    "PANEL",
    "PANEL_BODY",
    "PANEL_HEADER",
    "PANEL_BORDER",
    "SECTION",
    "TABLE_BG",
    "TEXT",
    "MUTED_TEXT",
    "ACCENT",
    "configure_app_style",
    "force_panel_backgrounds",
]
