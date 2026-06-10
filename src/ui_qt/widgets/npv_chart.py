from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ui_qt.design.tokens import THEMES, normalize_theme_name
from ui_qt.widgets.empty_state import EmptyState

from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget

try:  # pragma: no cover - optional GUI dependency branch
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator
except Exception:  # pragma: no cover - graceful GUI fallback
    FigureCanvasQTAgg = None  # type: ignore[assignment]
    Figure = None  # type: ignore[assignment]
    MaxNLocator = None  # type: ignore[assignment]


class NpvChart(QWidget):
    """Minimal Qt/matplotlib chart for accumulated NPV dynamics."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._figure = None
        self._axis = None
        self._canvas = None

        if Figure is None or FigureCanvasQTAgg is None:
            layout.addWidget(
                EmptyState(
                    "Нет графика",
                    self,
                    status="matplotlib недоступен",
                )
            )
            return

        self._figure = Figure(figsize=(6.0, 2.7), dpi=100)
        self._axis = self._figure.add_subplot(111)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.setMinimumHeight(220)
        layout.addWidget(self._canvas)
        self.update_points([], discount_rate=0.0)

    def update_points(self, points: Sequence[Any], *, discount_rate: float) -> None:
        if self._axis is None or self._canvas is None or self._figure is None:
            return

        tokens = THEMES[self._theme_name()].colors
        years = [int(getattr(point, "year", 0)) for point in points]
        values = [float(getattr(point, "accumulated_npv", 0.0)) for point in points]

        self._axis.clear()
        self._figure.patch.set_facecolor(tokens.surface)
        self._axis.set_facecolor(tokens.surface)

        if not values:
            self._axis.text(
                0.5,
                0.5,
                "Нет расчёта",
                ha="center",
                va="center",
                color=tokens.text_muted,
                transform=self._axis.transAxes,
            )
            self._axis.set_xticks([])
            self._axis.set_yticks([])
        else:
            self._axis.plot(
                years,
                values,
                marker="o",
                linestyle="-",
                color=tokens.accent,
                linewidth=2.2,
                markersize=5,
            )
            self._axis.fill_between(years, values, 0, color=tokens.accent, alpha=0.08)
            self._axis.axhline(
                0,
                color=tokens.danger,
                linestyle=(0, (4, 4)),
                linewidth=1.1,
                alpha=0.85,
            )
            self._axis.set_title(
                f"NPV по годам · r={discount_rate:.2%}",
                color=tokens.text,
                fontsize=10,
                pad=8,
            )
            self._axis.set_xlabel("Год", color=tokens.text_muted, fontsize=9)
            self._axis.set_ylabel("Накопленный NPV", color=tokens.text_muted, fontsize=9)
            if MaxNLocator is not None:
                self._axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            self._axis.grid(True, linestyle=":", linewidth=0.8, color=tokens.border, alpha=0.8)

        for spine in ("top", "right"):
            self._axis.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            self._axis.spines[spine].set_color(tokens.border)
        self._axis.tick_params(colors=tokens.text_muted, labelsize=8)
        self._figure.tight_layout(pad=1.0)
        self._canvas.draw_idle()

    def _theme_name(self) -> str:
        app = QApplication.instance()
        value = app.property("itCostTheme") if app is not None else None
        return normalize_theme_name(str(value) if value is not None else None)
