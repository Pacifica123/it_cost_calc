from __future__ import annotations

from typing import Any, Mapping, Sequence

from application.services.decision_report_service import DecisionReportService


class BuildDecisionReportUseCase:
    def __init__(self, report_service: DecisionReportService):
        self.report_service = report_service

    def execute(
        self,
        *,
        project: Mapping[str, Any] | None = None,
        entities: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
        cost_totals: Mapping[str, Any] | None = None,
        candidate_configurations: Sequence[Mapping[str, Any]] | None = None,
        ahp_result: Mapping[str, Any] | None = None,
        criteria_importance_result: Mapping[str, Any] | None = None,
        genetic_result: Mapping[str, Any] | None = None,
        genetic_ahp_result: Mapping[str, Any] | None = None,
        npv_report: Mapping[str, Any] | None = None,
        assumptions: Sequence[str] | None = None,
        risks: Sequence[Mapping[str, Any]] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.report_service.build_report(
            project=project,
            entities=entities,
            cost_totals=cost_totals,
            candidate_configurations=candidate_configurations,
            ahp_result=ahp_result,
            criteria_importance_result=criteria_importance_result,
            genetic_result=genetic_result,
            genetic_ahp_result=genetic_ahp_result,
            npv_report=npv_report,
            assumptions=assumptions,
            risks=risks,
            metadata=metadata,
        )
