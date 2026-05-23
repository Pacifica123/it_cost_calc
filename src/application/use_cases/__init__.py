"""Application use case public API with lazy imports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "BuildDecisionReportUseCase": "application.use_cases.build_decision_report",
    "BuildNpvReportUseCase": "application.use_cases.build_npv_report",
    "CalculateElectricityCostsUseCase": "application.use_cases.calculate_electricity_costs",
    "ExportCostReportUseCase": "application.use_cases.export_cost_report",
    "LoadDemoDatasetUseCase": "application.use_cases.load_demo_dataset",
    "PrepareCostSummaryUseCase": "application.use_cases.prepare_cost_summary",
    "RunGeneticOptimizationUseCase": "application.use_cases.run_genetic_optimization",
    "RunGeneticAhpRankingUseCase": "application.use_cases.run_genetic_ahp_ranking",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
