"""Application service public API with lazy imports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "AnalysisScopeProfileService": "application.services.analysis_scope_profile_service",
    "AnalysisScopeProfile": "application.services.analysis_scope_profile_service",
    "CandidateConfigurationService": "application.services.candidate_configuration_service",
    "CostAggregationService": "application.services.cost_aggregation_service",
    "CostSummaryService": "application.services.cost_summary_service",
    "DecisionReportService": "application.services.decision_report_service",
    "DemoControlScenarioService": "application.services.demo_control_scenario_service",
    "ElectricityCostService": "application.services.electricity_cost_service",
    "EntityCatalogService": "application.services.entity_catalog_service",
    "EquipmentService": "application.services.equipment_service",
    "GeneticOptimizationService": "application.services.genetic_optimization_service",
    "GeneticAhpRankingService": "application.services.genetic_ahp_ranking_service",
    "NPVReportService": "application.services.npv_report_service",
    "RuntimeEntityNormalizationService": "application.services.runtime_entity_normalization_service",
    "SolutionComponentAnalyticsIntegrationService": "application.services.solution_component_analytics_service",
    "SolutionComponentNormalizationService": "application.services.solution_component_normalization_service",
    "SolutionComponentRuntimeService": "application.services.solution_component_runtime_service",
    "TCOModelService": "application.services.tco_model_service",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
