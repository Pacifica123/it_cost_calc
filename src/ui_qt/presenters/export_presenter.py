from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ui_qt.presenters.app_presenter import QtAppPresenter


@dataclass(frozen=True)
class ExportMode:
    """Short export option shown by the Qt export screen."""

    mode_id: str
    label: str
    details: str


@dataclass(frozen=True)
class ExportSummary:
    """Compact dashboard summary for export screen cards."""

    total_rows: int
    npv_ready: bool
    report_dir: Path
    last_files: int = 0


class ExportPresenter:
    """Qt-facing export boundary over existing application services."""

    _MODES: tuple[ExportMode, ...] = (
        ExportMode(
            "full_decision_report",
            "Полный отчёт",
            "JSON, Markdown, CSV кандидатов и CSV компонентов.",
        ),
        ExportMode(
            "cost_summary",
            "Затраты",
            "Markdown-снимок CAPEX, OPEX, энергии и TCO.",
        ),
        ExportMode(
            "raw_costs_csv",
            "CSV затрат",
            "Табличный CSV с текущими затратами.",
        ),
        ExportMode(
            "npv",
            "NPV",
            "Markdown-снимок последнего NPV-расчёта.",
        ),
        ExportMode(
            "technical_analysis",
            "ТО-анализ",
            "Маркер частичного экспорта до переноса ТО.",
        ),
        ExportMode(
            "software_analysis",
            "ПО-анализ",
            "Маркер частичного экспорта до переноса ПО.",
        ),
    )

    def __init__(self, app_presenter: QtAppPresenter | None = None) -> None:
        self.app_presenter = app_presenter or QtAppPresenter()
        self._last_paths: dict[str, Path] = {}

    def modes(self) -> tuple[ExportMode, ...]:
        return self._MODES

    def mode_by_id(self, mode_id: str) -> ExportMode:
        for mode in self._MODES:
            if mode.mode_id == mode_id:
                return mode
        raise ValueError(f"Unknown export mode: {mode_id!r}")

    def dashboard_text(self) -> str:
        return self.app_presenter.build_export_dashboard_summary()

    def summary(self) -> ExportSummary:
        return ExportSummary(
            total_rows=self.app_presenter.total_rows(),
            npv_ready=bool(self.app_presenter.get_npv_report()),
            report_dir=self.app_presenter.generated_reports_dir(),
            last_files=len(self._last_paths),
        )

    def export(self, mode_id: str) -> dict[str, Path]:
        self.mode_by_id(mode_id)
        self._last_paths = self.app_presenter.export_selected_fragment(mode_id)
        return dict(self._last_paths)

    def open_folder(self) -> Path:
        return self.app_presenter.open_generated_reports_folder()

    def last_paths(self) -> dict[str, Path]:
        return dict(self._last_paths)

    @staticmethod
    def path_rows(paths: Mapping[str, Path]) -> list[dict[str, str]]:
        return [
            {"kind": str(kind), "path": str(path)}
            for kind, path in paths.items()
        ]
