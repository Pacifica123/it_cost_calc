from __future__ import annotations

from collections.abc import Sequence

from ui_qt.design.tokens import THEMES, normalize_theme_name
from ui_qt.widgets.empty_state import EmptyState

from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget

try:  # pragma: no cover - optional GUI dependency branch
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
except Exception:  # pragma: no cover - graceful GUI fallback
    FigureCanvasQTAgg = None  # type: ignore[assignment]
    Figure = None  # type: ignore[assignment]


class AnalysisChart(QWidget):
    """Small matplotlib chart used by the risk, TCO-by-years and financing screens.

    One widget, several plot kinds: tornado bars, a histogram, stacked yearly bars
    and simple lines. Everything takes plain numbers, so the screens stay testable.
    """

    def __init__(self, parent: QWidget | None = None, *, height: int = 220) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._figure = None
        self._axis = None
        self._canvas = None
        self.last_kind = "empty"
        if Figure is None or FigureCanvasQTAgg is None:
            layout.addWidget(EmptyState("Нет графика", self, status="matplotlib недоступен"))
            return
        self._figure = Figure(figsize=(6.0, 2.7), dpi=100)
        self._axis = self._figure.add_subplot(111)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.setMinimumHeight(height)
        layout.addWidget(self._canvas)
        self.show_empty()

    # --- public plots ---------------------------------------------------------------------------

    def show_empty(self, text: str = "Нет расчёта") -> None:
        if not self._begin():
            return
        tokens = self._tokens()
        self._axis.text(0.5, 0.5, text, ha="center", va="center", color=tokens.text_muted, transform=self._axis.transAxes)
        self._axis.set_xticks([])
        self._axis.set_yticks([])
        self._finish("empty")

    def show_tornado(self, labels: Sequence[str], lows: Sequence[float], highs: Sequence[float], base: float, title: str) -> None:
        if not labels:
            self.show_empty()
            return
        if not self._begin():
            return
        tokens = self._tokens()
        positions = list(range(len(labels)))[::-1]
        for position, low, high in zip(positions, lows, highs):
            self._axis.barh(position, base - low, left=low, color=tokens.danger, alpha=0.75, height=0.6)
            self._axis.barh(position, high - base, left=base, color=tokens.accent, alpha=0.85, height=0.6)
        self._axis.axvline(base, color=tokens.text_muted, linewidth=1.0)
        self._axis.axvline(0, color=tokens.danger, linestyle=(0, (4, 4)), linewidth=1.0, alpha=0.8)
        self._axis.set_yticks(positions)
        self._axis.set_yticklabels(list(labels), fontsize=8)
        self._axis.set_title(title, color=tokens.text, fontsize=10, pad=8)
        self._axis.set_xlabel("NPV, ₽", color=tokens.text_muted, fontsize=9)
        self._finish("tornado")

    def show_histogram(self, lows: Sequence[float], highs: Sequence[float], counts: Sequence[int], title: str) -> None:
        if not counts:
            self.show_empty()
            return
        if not self._begin():
            return
        tokens = self._tokens()
        for low, high, count in zip(lows, highs, counts):
            color = tokens.danger if high <= 0 else tokens.accent
            self._axis.bar(low, count, width=(high - low) or 1.0, align="edge", color=color, alpha=0.8, edgecolor=tokens.surface)
        self._axis.axvline(0, color=tokens.danger, linestyle=(0, (4, 4)), linewidth=1.0)
        self._axis.set_title(title, color=tokens.text, fontsize=10, pad=8)
        self._axis.set_xlabel("NPV, ₽", color=tokens.text_muted, fontsize=9)
        self._axis.set_ylabel("Прогонов", color=tokens.text_muted, fontsize=9)
        self._finish("histogram")

    def show_stacked_bars(self, categories: Sequence[str], series: Sequence[tuple[str, Sequence[float]]], title: str) -> None:
        if not categories or not series:
            self.show_empty()
            return
        if not self._begin():
            return
        tokens = self._tokens()
        palette = (tokens.accent, tokens.danger, tokens.text_muted, tokens.border)
        bottoms = [0.0] * len(categories)
        positions = list(range(len(categories)))
        for index, (name, values) in enumerate(series):
            values = [float(value) for value in values]
            self._axis.bar(positions, values, bottom=bottoms, label=name, color=palette[index % len(palette)], alpha=0.85 - 0.1 * (index // len(palette)))
            bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
        self._axis.set_xticks(positions)
        self._axis.set_xticklabels(list(categories), fontsize=8)
        self._axis.set_title(title, color=tokens.text, fontsize=10, pad=8)
        self._axis.legend(fontsize=7, frameon=False, labelcolor=tokens.text_muted)
        self._finish("stacked")

    def show_lines(self, xs: Sequence[float], series: Sequence[tuple[str, Sequence[float]]], title: str, *, x_label: str = "") -> None:
        if not xs or not series:
            self.show_empty()
            return
        if not self._begin():
            return
        tokens = self._tokens()
        palette = (tokens.accent, tokens.danger, tokens.text_muted)
        for index, (name, values) in enumerate(series):
            self._axis.plot(list(xs), [float(value) for value in values], marker="o", markersize=4, linewidth=2.0, label=name, color=palette[index % len(palette)])
        self._axis.axhline(0, color=tokens.border, linewidth=0.8)
        self._axis.set_title(title, color=tokens.text, fontsize=10, pad=8)
        if x_label:
            self._axis.set_xlabel(x_label, color=tokens.text_muted, fontsize=9)
        if len(series) > 1:
            self._axis.legend(fontsize=7, frameon=False, labelcolor=tokens.text_muted)
        self._finish("lines")

    # --- helpers --------------------------------------------------------------------------------

    def _begin(self) -> bool:
        if self._axis is None or self._figure is None:
            return False
        tokens = self._tokens()
        self._axis.clear()
        self._figure.patch.set_facecolor(tokens.surface)
        self._axis.set_facecolor(tokens.surface)
        return True

    def _finish(self, kind: str) -> None:
        tokens = self._tokens()
        for spine in ("top", "right"):
            self._axis.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            self._axis.spines[spine].set_color(tokens.border)
        self._axis.tick_params(colors=tokens.text_muted, labelsize=8)
        if kind != "empty":
            self._axis.grid(True, linestyle=":", linewidth=0.8, color=tokens.border, alpha=0.8)
        self._figure.tight_layout(pad=1.0)
        self._canvas.draw_idle()
        self.last_kind = kind

    def _tokens(self):
        app = QApplication.instance()
        value = app.property("itCostTheme") if app is not None else None
        return THEMES[normalize_theme_name(str(value) if value is not None else None)].colors
