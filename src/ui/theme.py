"""Application-wide styling helpers for the desktop UI."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


# Flat, calmer palette for the desktop UI.  The goal is a more modern and
# minimalistic hierarchy: a cool light-gray shell, white work surfaces, subtle
# panel contrast and restrained blue accent actions.
BACKGROUND = "#edf1f5"          # whole application shell
WORKSPACE = "#f7f9fc"           # tab/workspace surface
SURFACE = "#eef3f8"             # secondary surface around panels
PANEL = "#ffffff"               # panel container
PANEL_BODY = "#f8fafc"          # panel content area
PANEL_HEADER = "#f1f5f9"        # collapsible header
PANEL_BORDER = "#d7dee7"
SECTION = "#eef2f6"
TABLE_BG = "#ffffff"
TEXT = "#1f2937"
MUTED_TEXT = "#6b7280"
ACCENT = "#3b82f6"
ACCENT_HOVER = "#2563eb"
ACCENT_MUTED = "#dbeafe"
SUBTLE_BUTTON = "#e5ebf2"
SUBTLE_BUTTON_HOVER = "#d7e0ea"


_PANEL_TTK_STYLES = {
    "TFrame": "PanelBody.TFrame",
    "TLabel": "PanelBody.TLabel",
    "TLabelframe": "Panel.TLabelframe",
    "TCheckbutton": "PanelBody.TCheckbutton",
    "TRadiobutton": "PanelBody.TRadiobutton",
    "TNotebook": "PanelBody.TNotebook",
}


def configure_app_style(root) -> None:
    """Apply the flat modern visual hierarchy to Tk/ttk widgets."""

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:  # noqa: BLE001 - fall back to the platform theme if unavailable
        pass

    root.configure(background=BACKGROUND)

    style.configure(".", font=("TkDefaultFont", 9), foreground=TEXT)
    style.map(".", foreground=[("disabled", "#9aa4b2")])

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
        foreground=MUTED_TEXT,
        font=("TkDefaultFont", 8, "bold"),
    )

    style.configure("TLabelframe", background=SECTION, bordercolor=PANEL_BORDER, relief="solid")
    style.configure("TLabelframe.Label", background=SECTION, foreground=TEXT)
    style.configure("Panel.TLabelframe", background=PANEL_BODY, bordercolor=PANEL_BORDER, relief="solid")
    style.configure("Panel.TLabelframe.Label", background=PANEL_BODY, foreground=TEXT)
    style.configure("PanelBody.TCheckbutton", background=PANEL_BODY, foreground=TEXT)
    style.map("PanelBody.TCheckbutton", background=[("active", PANEL_BODY)])
    style.configure("PanelBody.TRadiobutton", background=PANEL_BODY, foreground=TEXT)
    style.map("PanelBody.TRadiobutton", background=[("active", PANEL_BODY)])

    style.configure("TPanedwindow", background=BACKGROUND, sashwidth=6)
    style.configure("Workspace.TPanedwindow", background=WORKSPACE, sashwidth=6)

    style.configure("TNotebook", background=WORKSPACE, borderwidth=0, tabmargins=(0, 0, 0, 0))
    style.configure("PanelBody.TNotebook", background=PANEL_BODY, borderwidth=0)
    style.configure(
        "TNotebook.Tab",
        padding=(12, 7),
        background=SURFACE,
        foreground=MUTED_TEXT,
        borderwidth=0,
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", PANEL), ("active", WORKSPACE)],
        foreground=[("selected", TEXT), ("active", TEXT)],
        expand=[("selected", (0, 0, 0, 0))],
    )

    style.configure(
        "TButton",
        padding=(10, 6),
        background=ACCENT,
        foreground="#ffffff",
        borderwidth=0,
        relief="flat",
        focusthickness=0,
    )
    style.map(
        "TButton",
        background=[("active", ACCENT_HOVER), ("pressed", ACCENT_HOVER), ("disabled", "#cbd5e1")],
        foreground=[("disabled", "#f8fafc")],
    )
    style.configure(
        "Subtle.TButton",
        padding=(10, 6),
        background=SUBTLE_BUTTON,
        foreground=TEXT,
        borderwidth=0,
        relief="flat",
        focusthickness=0,
    )
    style.map(
        "Subtle.TButton",
        background=[("active", SUBTLE_BUTTON_HOVER), ("pressed", SUBTLE_BUTTON_HOVER), ("disabled", SURFACE)],
        foreground=[("disabled", "#9aa4b2")],
    )
    style.configure(
        "Toolbar.TButton",
        padding=(12, 8),
        background=ACCENT,
        foreground="#ffffff",
        borderwidth=0,
        relief="flat",
    )
    style.map(
        "Toolbar.TButton",
        background=[("active", ACCENT_HOVER), ("pressed", ACCENT_HOVER)],
    )

    style.configure(
        "TEntry",
        fieldbackground="#ffffff",
        background="#ffffff",
        foreground=TEXT,
        bordercolor=PANEL_BORDER,
        lightcolor=PANEL_BORDER,
        darkcolor=PANEL_BORDER,
        insertcolor=TEXT,
        padding=5,
    )
    style.configure(
        "TCombobox",
        fieldbackground="#ffffff",
        background="#ffffff",
        foreground=TEXT,
        bordercolor=PANEL_BORDER,
        lightcolor=PANEL_BORDER,
        darkcolor=PANEL_BORDER,
        arrowsize=14,
        padding=4,
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", "#ffffff")],
        selectbackground=[("readonly", "#ffffff")],
        selectforeground=[("readonly", TEXT)],
    )
    style.configure(
        "Treeview",
        rowheight=24,
        background=TABLE_BG,
        fieldbackground=TABLE_BG,
        foreground=TEXT,
        bordercolor=PANEL_BORDER,
        lightcolor=PANEL_BORDER,
        darkcolor=PANEL_BORDER,
    )
    style.map("Treeview", background=[("selected", ACCENT_MUTED)], foreground=[("selected", TEXT)])
    style.configure(
        "Treeview.Heading",
        background=SECTION,
        foreground=TEXT,
        font=("TkDefaultFont", 9, "bold"),
        relief="flat",
        borderwidth=0,
    )


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
    except (tk.TclError, AttributeError):
        # ttkbootstrap can raise AttributeError instead of TclError when a
        # custom ttk style is applied before its Style singleton exists.  The
        # colour pass is best-effort; direct Tk backgrounds above still keep the
        # panel hierarchy visible.
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
    "ACCENT_HOVER",
    "ACCENT_MUTED",
    "SUBTLE_BUTTON",
    "SUBTLE_BUTTON_HOVER",
    "configure_app_style",
    "force_panel_backgrounds",
]
