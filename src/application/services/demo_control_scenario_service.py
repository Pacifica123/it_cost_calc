"""Control scenario builder for roadmap stage 8.

The service keeps demonstration data, educational examples and regression
invariants tied to the same application-level model without making the UI know
about fixture roles.  It builds a deterministic snapshot from runtime-like
demo entities and verifies that they can pass through normalization,
``CandidateConfiguration``, TCO and ``DecisionReport``.
"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from typing import Any, Mapping, Sequence

from application.services.candidate_configuration_service import CandidateConfigurationService
from application.services.decision_demo_service import DecisionDemoDataService
from application.services.decision_report_service import DecisionReportService
from application.services.genetic_optimization_service import GeneticOptimizationService
from application.services.runtime_entity_normalization_service import RuntimeEntityNormalizationService
from domain.decision.criteria_importance.pipeline import run_importance_pipeline
from shared.constants import (
    ANALYSIS_SCOPE_SOFTWARE,
    ANALYSIS_SCOPE_TECHNICAL,
    CAPITAL_COST_CATEGORIES,
    OPERATIONAL_COST_CATEGORIES,
)


class DemoControlScenarioService:
    """Build and validate a reproducible demo/control scenario.

    Stage 8 does not add a new analytical method.  It makes the demo fixture a
    first-class scenario: the same rows that are loaded into CAPEX/OPEX tabs are
    normalized, converted into candidate configurations, enriched with TCO and
    then assembled into a partial DecisionReport.
    """

    SCENARIO_ID = "it_solution_demo_control_v1"
    SCENARIO_ROLE = "demonstration_control"

    def __init__(
        self,
        *,
        normalizer: RuntimeEntityNormalizationService | None = None,
        demo_data_service: DecisionDemoDataService | None = None,
        candidate_configuration_service: CandidateConfigurationService | None = None,
        decision_report_service: DecisionReportService | None = None,
    ):
        self.normalizer = normalizer or RuntimeEntityNormalizationService()
        self.demo_data_service = demo_data_service or DecisionDemoDataService()
        self.candidate_configuration_service = (
            candidate_configuration_service or CandidateConfigurationService()
        )
        self.decision_report_service = decision_report_service or DecisionReportService(
            self.candidate_configuration_service
        )

    def build(
        self,
        entities: Mapping[str, Sequence[Mapping[str, Any]]],
        *,
        scenario_id: str = SCENARIO_ID,
        scenario_title: str = "Демо-контроль выбора ИТ-решения",
    ) -> dict[str, Any]:
        """Return a deterministic scenario payload without writing files."""

        normalized_entities = self.normalizer.normalize_entities(
            {
                str(category): [dict(row) for row in rows]
                for category, rows in dict(entities).items()
            }
        )
        demo_payload = self.demo_data_service.build(normalized_entities)
        scoped_payloads = demo_payload.get("scoped_payloads", {})
        candidate_configurations = self._combined_candidate_pool(scoped_payloads)
        criteria_reports = self._criteria_reports(scoped_payloads)
        runtime_candidate = self.candidate_configuration_service.from_runtime_entities(
            normalized_entities,
            scope=None,
            candidate_id="demo_runtime_current",
            name="Демо-набор компонентов после нормализации",
        )
        decision_report = self.decision_report_service.build_report(
            project={
                "id": scenario_id,
                "title": scenario_title,
                "goal": "Проверить, что демонстрационный набор проходит общую цепочку выбора ИТ-решения.",
            },
            entities=normalized_entities,
            cost_totals=runtime_candidate.totals,
            candidate_configurations=candidate_configurations,
            criteria_importance_result=criteria_reports.get(ANALYSIS_SCOPE_TECHNICAL),
            assumptions=[
                "Демо-сценарий использует фиксированные параметры аналитики и не зависит от текущего GUI-состояния.",
                "Учебные примеры AHP/GA хранятся отдельно и не подменяют пользовательский демонстрационный сценарий.",
            ],
            metadata={
                "scenario_id": scenario_id,
                "scenario_role": self.SCENARIO_ROLE,
                "input_fingerprint": self._fingerprint(
                    {
                        "entities": normalized_entities,
                        "reproducibility": self._reproducibility_contract(),
                    }
                ),
            },
        )
        payload = {
            "scenario_id": scenario_id,
            "scenario_role": self.SCENARIO_ROLE,
            "schema_version": 1,
            "fixture_roles": self._fixture_roles(),
            "reproducibility": self._reproducibility_contract(),
            "normalized_entities": normalized_entities,
            "scoped_payloads": deepcopy(scoped_payloads),
            "candidate_configurations": deepcopy(candidate_configurations),
            "criteria_reports": criteria_reports,
            "decision_report": decision_report,
            "invariants": self._invariants(
                normalized_entities=normalized_entities,
                scoped_payloads=scoped_payloads,
                candidate_configurations=candidate_configurations,
                criteria_reports=criteria_reports,
                decision_report=decision_report,
            ),
        }
        payload["validation"] = self.validate(payload)
        return payload

    def validate(self, scenario_payload: Mapping[str, Any]) -> dict[str, Any]:
        """Validate stage-8 invariants for smoke checks and devctl manifest."""

        errors: list[str] = []
        normalized_entities = scenario_payload.get("normalized_entities", {})
        if not isinstance(normalized_entities, Mapping):
            errors.append("normalized_entities must be a mapping")
            normalized_entities = {}

        for category in CAPITAL_COST_CATEGORIES + OPERATIONAL_COST_CATEGORIES:
            rows = normalized_entities.get(category, [])
            for index, row in enumerate(rows):
                if not isinstance(row, Mapping):
                    errors.append(f"{category}[{index}] is not a mapping")
                    continue
                if not row.get("scope"):
                    errors.append(f"{category}[{index}] has no scope")
                if not row.get("component_type"):
                    errors.append(f"{category}[{index}] has no component_type")

        scoped_payloads = scenario_payload.get("scoped_payloads", {})
        if not isinstance(scoped_payloads, Mapping):
            errors.append("scoped_payloads must be a mapping")
            scoped_payloads = {}
        for scope in (ANALYSIS_SCOPE_TECHNICAL, ANALYSIS_SCOPE_SOFTWARE):
            payload = scoped_payloads.get(scope, {})
            candidates = payload.get("candidate_configurations", []) if isinstance(payload, Mapping) else []
            if not candidates:
                errors.append(f"{scope} scoped payload has no candidate_configurations")

        candidate_configurations = scenario_payload.get("candidate_configurations", [])
        if not candidate_configurations:
            errors.append("combined candidate_configurations pool is empty")
        for index, candidate in enumerate(candidate_configurations or []):
            if not isinstance(candidate, Mapping):
                errors.append(f"candidate_configurations[{index}] is not a mapping")
                continue
            if not candidate.get("scope"):
                errors.append(f"candidate_configurations[{index}] has no scope")
            totals = candidate.get("totals", {})
            if not isinstance(totals, Mapping) or "tco" not in totals:
                errors.append(f"candidate_configurations[{index}] has no totals.tco")

        decision_report = scenario_payload.get("decision_report", {})
        report_candidates = (
            decision_report.get("candidate_configurations", [])
            if isinstance(decision_report, Mapping)
            else []
        )
        if not report_candidates:
            errors.append("DecisionReport has no candidate_configurations")

        reproducibility = scenario_payload.get("reproducibility", {})
        if not isinstance(reproducibility, Mapping):
            errors.append("reproducibility must be a mapping")
        elif reproducibility.get("ga", {}).get("seed") is None:
            errors.append("GA seed is not fixed")

        return {
            "status": "ok" if not errors else "failed",
            "errors": errors,
        }

    def _combined_candidate_pool(
        self,
        scoped_payloads: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        by_id: dict[str, dict[str, Any]] = {}
        for scope in (ANALYSIS_SCOPE_TECHNICAL, ANALYSIS_SCOPE_SOFTWARE):
            payload = scoped_payloads.get(scope, {})
            if not isinstance(payload, Mapping):
                continue
            for candidate in payload.get("candidate_configurations", []) or []:
                if not isinstance(candidate, Mapping):
                    continue
                candidate_id = str(candidate.get("id") or candidate.get("name") or len(by_id) + 1)
                by_id[f"{scope}:{candidate_id}"] = deepcopy(dict(candidate))
        return list(by_id.values())

    def _criteria_reports(self, scoped_payloads: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        reports: dict[str, dict[str, Any]] = {}
        for scope in (ANALYSIS_SCOPE_TECHNICAL, ANALYSIS_SCOPE_SOFTWARE):
            payload = scoped_payloads.get(scope, {})
            if not isinstance(payload, Mapping):
                continue
            case_data = payload.get("criteria_case")
            if isinstance(case_data, Mapping):
                reports[scope] = run_importance_pipeline(dict(deepcopy(case_data)))
        return reports

    def _invariants(
        self,
        *,
        normalized_entities: Mapping[str, Sequence[Mapping[str, Any]]],
        scoped_payloads: Mapping[str, Any],
        candidate_configurations: Sequence[Mapping[str, Any]],
        criteria_reports: Mapping[str, Any],
        decision_report: Mapping[str, Any],
    ) -> dict[str, Any]:
        scope_counts: dict[str, int] = {}
        component_type_counts: dict[str, int] = {}
        known_rows = 0
        for category in CAPITAL_COST_CATEGORIES + OPERATIONAL_COST_CATEGORIES:
            for row in normalized_entities.get(category, []):
                known_rows += 1
                scope = str(row.get("scope") or "missing")
                component_type = str(row.get("component_type") or "missing")
                scope_counts[scope] = scope_counts.get(scope, 0) + 1
                component_type_counts[component_type] = component_type_counts.get(component_type, 0) + 1

        candidate_count_by_scope: dict[str, int] = {}
        for scope, payload in dict(scoped_payloads).items():
            if isinstance(payload, Mapping):
                candidate_count_by_scope[str(scope)] = len(payload.get("candidate_configurations", []) or [])

        return {
            "known_runtime_rows": known_rows,
            "scope_counts": scope_counts,
            "component_type_counts": component_type_counts,
            "candidate_count_by_scope": candidate_count_by_scope,
            "combined_candidate_count": len(candidate_configurations),
            "tco_candidate_count": sum(
                1
                for candidate in candidate_configurations
                if isinstance(candidate.get("totals"), Mapping) and candidate["totals"].get("tco")
            ),
            "criteria_report_scopes": sorted(criteria_reports),
            "decision_report_candidate_count": len(decision_report.get("candidate_configurations", []) or []),
            "decision_report_warning_count": len(decision_report.get("warnings", []) or []),
        }

    def _fixture_roles(self) -> dict[str, Any]:
        return {
            "educational": {
                "purpose": "Пояснение отдельных методик на малых примерах.",
                "examples": [
                    "data/examples/ahp/test_confs.json",
                    "data/examples/optimization/ga_demo_output.json",
                ],
            },
            "demonstration": {
                "purpose": "Сквозной пользовательский сценарий загрузки данных и показа выбора ИТ-решения.",
                "fixture": "data/fixtures/demo_dataset.json",
            },
            "regression": {
                "purpose": "Проверка инвариантов без тестового раннера и без записи cache-файлов.",
                "fixture": "data/fixtures/regression/demo_control_invariants.json",
            },
        }

    def _reproducibility_contract(self) -> dict[str, Any]:
        ga_defaults = deepcopy(GeneticOptimizationService.DEFAULT_GA_PARAMS)
        return {
            "ga": {
                "seed": ga_defaults.get("seed", 42),
                "pop_size": ga_defaults.get("pop_size"),
                "generations": ga_defaults.get("generations"),
                "mutation_rate": ga_defaults.get("mutation_rate"),
                "elite": ga_defaults.get("elite"),
                "stagnation_limit": ga_defaults.get("stagnation_limit"),
                "init_mode": ga_defaults.get("init_mode"),
                "top_solutions_limit": ga_defaults.get("top_solutions_limit"),
            },
            "ahp": {
                "criteria_source": "AnalysisScopeProfileService.default_weights",
                "candidate_pool_source": "DecisionDemoDataService.scoped_payloads[*].candidate_configurations",
            },
            "hybrid_assessment": {
                "candidate_pool_source": "same_as_ga_ahp_export_payload",
                "rule": "compare GA and AHP leaders over one candidate pool",
            },
        }

    def _fingerprint(self, payload: Mapping[str, Any]) -> str:
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return sha256(raw.encode("utf-8")).hexdigest()


__all__ = ["DemoControlScenarioService"]
