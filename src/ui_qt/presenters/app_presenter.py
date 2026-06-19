from __future__ import annotations

from copy import deepcopy
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from application.services.analysis_scope_profile_service import AnalysisScopeProfileService
from application.services.cost_aggregation_service import CostAggregationService
from application.services.decision_report_service import DecisionReportService
from application.services.entity_catalog_service import EntityCatalogService
from application.services.genetic_optimization_service import GeneticOptimizationService
from application.services.hybrid_decision_assessment_service import HybridDecisionAssessmentService
from application.services.electricity_cost_service import ElectricityCostService
from application.services.equipment_service import EquipmentService
from application.services.npv_report_service import NPVReportService
from application.services.scoped_candidate_pool_service import CandidatePoolSnapshot, ScopedCandidatePoolService
from application.services.solution_component_runtime_service import SolutionComponentRuntimeService
from domain.decision.ahp.math_utils import ahp_priority_and_consistency, build_pairwise_matrix_from_scores
from domain.decision.criteria_importance.pipeline import run_importance_pipeline
from application.use_cases.build_decision_report import BuildDecisionReportUseCase
from application.use_cases.build_npv_report import BuildNpvReportUseCase
from application.use_cases.calculate_electricity_costs import CalculateElectricityCostsUseCase
from application.use_cases.load_demo_dataset import LoadDemoDatasetUseCase
from application.use_cases.run_genetic_optimization import RunGeneticOptimizationUseCase
from application.use_cases.export_cost_report import ExportCostReportUseCase
from application.use_cases.prepare_cost_summary import PrepareCostSummaryUseCase
from infrastructure.exporters.decision_report_exporter import (
    export_decision_report_csv,
    export_decision_report_json,
    export_decision_report_markdown,
    export_solution_component_csv,
)
from infrastructure.repositories.json_entity_repository import JsonEntityRepository
from infrastructure.storage import JsonFileStorage


@dataclass(frozen=True)
class QtRuntimePaths:
    """Filesystem paths used by the Qt adapter layer."""

    repo_root: Path
    runtime_entities_path: Path
    demo_dataset_path: Path
    demo_profiles_path: Path

    @classmethod
    def from_repo_root(
        cls,
        repo_root: str | Path | None = None,
        *,
        runtime_entities_path: str | Path | None = None,
        demo_dataset_path: str | Path | None = None,
        demo_profiles_path: str | Path | None = None,
    ) -> "QtRuntimePaths":
        root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
        root = root.resolve()
        data_root = root / "data"
        return cls(
            repo_root=root,
            runtime_entities_path=Path(
                runtime_entities_path or data_root / "generated" / "runtime_entities.json"
            ),
            demo_dataset_path=Path(demo_dataset_path or data_root / "fixtures" / "demo_dataset.json"),
            demo_profiles_path=Path(
                demo_profiles_path or data_root / "fixtures" / "demo_profiles.json"
            ),
        )


class QtAppPresenter:
    """Boundary between future Qt screens and existing application services.

    The presenter is deliberately free from QWidget, Treeview and tkinter types.
    It owns the JSON-backed runtime repository and exposes small operations that
    future screens can call without knowing storage details.
    """

    def __init__(
        self,
        *,
        repo_root: str | Path | None = None,
        runtime_entities_path: str | Path | None = None,
        demo_dataset_path: str | Path | None = None,
        demo_profiles_path: str | Path | None = None,
        storage: JsonFileStorage | None = None,
        repository: JsonEntityRepository | None = None,
    ) -> None:
        self.paths = QtRuntimePaths.from_repo_root(
            repo_root,
            runtime_entities_path=runtime_entities_path,
            demo_dataset_path=demo_dataset_path,
            demo_profiles_path=demo_profiles_path,
        )
        self.storage = storage or JsonFileStorage()
        self.repository = repository or JsonEntityRepository(
            self.paths.runtime_entities_path,
            self.storage,
        )
        self.catalog_service = EntityCatalogService(self.repository)
        self.equipment_service = EquipmentService(self.repository)
        self.solution_component_runtime_service = SolutionComponentRuntimeService(self.repository)
        self.electricity_cost_service = ElectricityCostService()
        self.electricity_calculator = CalculateElectricityCostsUseCase(
            self.electricity_cost_service,
        )
        self.cost_aggregation_service = CostAggregationService()
        self.cost_summary_preparer = PrepareCostSummaryUseCase(
            self.cost_aggregation_service,
            self.equipment_service,
            solution_component_runtime_service=self.solution_component_runtime_service,
        )
        self.export_cost_report_use_case = ExportCostReportUseCase(
            self.cost_summary_preparer,
            self.cost_aggregation_service,
        )
        self.npv_report_builder = BuildNpvReportUseCase(NPVReportService())
        self.decision_report_builder = BuildDecisionReportUseCase(DecisionReportService())
        self.genetic_optimization_service = GeneticOptimizationService(self.repository)
        self.hybrid_assessment_service = HybridDecisionAssessmentService()
        self.genetic_optimizer = RunGeneticOptimizationUseCase(
            self.genetic_optimization_service,
        )
        self.analysis_scope_profile_service = AnalysisScopeProfileService()
        self.scoped_candidate_pool_service = ScopedCandidatePoolService(
            profile_service=self.analysis_scope_profile_service,
        )
        self.demo_loader = LoadDemoDatasetUseCase(self.storage, self.equipment_service)
        self._electricity_profile: dict[str, float] = {
            "hours_per_day": 8.0,
            "working_days": 22.0,
            "cost_per_kwh": 1.0,
        }
        self._electricity_report: dict[str, Any] = {"total_cost": 0.0, "items": []}
        self._npv_report: dict[str, Any] = {}
        self._ga_results: dict[str, dict[str, Any]] = {}
        self._ahp_results: dict[str, dict[str, Any]] = {}
        self._pareto_results: dict[str, dict[str, Any]] = {}
        self._hybrid_results: dict[str, dict[str, Any]] = {}

    @property
    def entities(self) -> dict[str, list[dict[str, Any]]]:
        return self.repository.entities

    def entities_snapshot(self) -> dict[str, list[dict[str, Any]]]:
        return deepcopy(self.repository.entities)

    def list_entities(self, entity_name: str) -> list[dict[str, Any]]:
        return deepcopy(self.catalog_service.list_entities(entity_name))

    def get_entity(self, entity_name: str, row_index: int) -> dict[str, Any]:
        return deepcopy(self.catalog_service.get_entity(entity_name, row_index))

    def add_entity(self, entity_name: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        row = self.catalog_service.add_entity(entity_name, dict(payload))
        self._reset_derived_results()
        return deepcopy(row)

    def update_entity(
        self,
        entity_name: str,
        row_index: int,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        row = self.catalog_service.update_entity(entity_name, row_index, dict(payload))
        self._reset_derived_results()
        return deepcopy(row)

    def delete_entity(self, entity_name: str, row_index: int) -> dict[str, Any]:
        row = self.catalog_service.delete_entity(entity_name, row_index)
        self._reset_derived_results()
        return deepcopy(row)

    def replace_entities(self, payload: Mapping[str, list[Mapping[str, Any]]]) -> None:
        normalized_payload = {
            str(entity_name): [dict(row) for row in rows]
            for entity_name, rows in dict(payload).items()
        }
        self.equipment_service.replace_all(normalized_payload)
        self._reset_derived_results()

    def load_demo_dataset(self, path: str | Path | None = None) -> dict[str, list[dict[str, Any]]]:
        dataset = self.demo_loader.execute(path or self.paths.demo_dataset_path)
        self._reset_derived_results()
        return deepcopy(dataset)

    def list_demo_profiles(self) -> list[dict[str, Any]]:
        """Return selectable demo data profiles for the Qt shell."""

        registry_path = self.paths.demo_profiles_path
        if not registry_path.exists():
            return [
                {
                    "id": "default",
                    "title": "Обычный демо-набор",
                    "description": "Fallback profile for the default demo fixture.",
                    "path": str(self.paths.demo_dataset_path),
                }
            ]
        payload = self.storage.read(registry_path)
        profiles = payload.get("profiles", []) if isinstance(payload, Mapping) else []
        result: list[dict[str, Any]] = []
        for item in profiles:
            if not isinstance(item, Mapping):
                continue
            profile_id = str(item.get("id") or "").strip()
            raw_path = item.get("path")
            if not profile_id or not raw_path:
                continue
            result.append(
                {
                    "id": profile_id,
                    "title": str(item.get("title") or profile_id),
                    "description": str(item.get("description") or ""),
                    "path": str(raw_path),
                }
            )
        return result

    def load_demo_profile(self, profile_id: str) -> dict[str, list[dict[str, Any]]]:
        """Load one registered demo profile by id."""

        requested_id = str(profile_id or "default")
        for profile in self.list_demo_profiles():
            if profile.get("id") != requested_id:
                continue
            profile_path = Path(str(profile.get("path")))
            if not profile_path.is_absolute():
                profile_path = self.paths.repo_root / profile_path
            return self.load_demo_dataset(profile_path)
        available = ", ".join(profile["id"] for profile in self.list_demo_profiles())
        raise ValueError(f"Неизвестный demo-профиль: {requested_id}. Доступно: {available}")

    def save(self) -> Path:
        return self.repository.save()

    def total_rows(self) -> int:
        return sum(len(rows) for rows in self.repository.entities.values())

    def has_runtime_data(self) -> bool:
        return self.total_rows() > 0

    def _reset_derived_results(self) -> None:
        self._reset_electricity_result()
        self._npv_report = {}
        self._ga_results.clear()
        self._ahp_results.clear()
        self._pareto_results.clear()
        self._hybrid_results.clear()
        self.scoped_candidate_pool_service.clear("technical")
        self.scoped_candidate_pool_service.clear("software")

    def _reset_electricity_result(self) -> None:
        self._electricity_report = {"total_cost": 0.0, "items": []}

    def list_energy_rows(self) -> list[dict[str, Any]]:
        """Return equipment rows that can participate in electricity cost calculation."""

        return deepcopy(self.equipment_service.list_energy_relevant_rows())

    def list_round_the_clock_equipment_names(self) -> set[str]:
        return set(self.equipment_service.list_round_the_clock_equipment_names())

    def calculate_electricity_costs(
        self,
        *,
        hours_per_day: float,
        working_days: float,
        cost_per_kwh: float,
        equipment_rows: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Calculate electricity costs and keep the latest result for other screens."""

        rows = deepcopy(equipment_rows if equipment_rows is not None else self.list_energy_rows())
        report = self.electricity_calculator.execute(
            equipment_rows=rows,
            hours_per_day=float(hours_per_day),
            working_days=float(working_days),
            cost_per_kwh=float(cost_per_kwh),
            round_the_clock_names=self.list_round_the_clock_equipment_names(),
        )
        self._electricity_profile = {
            "hours_per_day": float(hours_per_day),
            "working_days": float(working_days),
            "cost_per_kwh": float(cost_per_kwh),
        }
        self._electricity_report = deepcopy(report)
        self._npv_report = {}
        return deepcopy(report)

    def get_electricity_cost(self) -> float:
        return float(self._electricity_report.get("total_cost", 0.0) or 0.0)

    def get_electricity_profile(self) -> dict[str, float]:
        return dict(self._electricity_profile)

    def get_electricity_report(self) -> dict[str, Any]:
        return deepcopy(self._electricity_report)

    def prepare_cost_summary(self) -> dict[str, Any]:
        """Build a current CAPEX/OPEX/electricity summary for finance screens."""

        return self.cost_summary_preparer.execute(
            electricity_cost=self.get_electricity_cost(),
            electricity_profile=self.get_electricity_profile(),
        )

    def prepare_npv_basis_from_current_costs(
        self,
        *,
        horizon_years: int = 5,
        annual_effect: float = 0.0,
        discount_rate: float | None = None,
    ) -> dict[str, Any]:
        totals = self.prepare_cost_summary()
        return self.npv_report_builder.prepare_from_totals(
            totals,
            horizon_years=horizon_years,
            annual_effect=annual_effect,
            discount_rate=discount_rate,
        )

    def build_npv_report(
        self,
        *,
        investment: float,
        discount_rate: float,
        cash_flows: list[float],
        financial_basis: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        report = self.npv_report_builder.execute(
            investment=float(investment),
            discount_rate=float(discount_rate),
            cash_flows=[float(value) for value in cash_flows],
            financial_basis=financial_basis,
        )
        self._npv_report = deepcopy(report)
        return deepcopy(report)

    def get_npv_report(self) -> dict[str, Any]:
        return deepcopy(self._npv_report)


    def run_genetic_optimization(
        self,
        *,
        scope: str,
        max_budget: float | None = None,
        max_power: float | None = None,
        target_units: float | None = None,
        max_target_units: float | None = None,
        max_selected_items: int | None = None,
        ga_params: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run GA for one analysis scope and update the shared candidate pool."""

        result = self.genetic_optimizer.execute(
            source=self.repository,
            analysis_scope=scope,
            max_budget=max_budget,
            max_power=max_power,
            target_units=target_units,
            max_target_units=max_target_units,
            max_selected_items=max_selected_items,
            ga_params=dict(ga_params or {}),
        )
        scope_key = str(scope)
        self._ga_results[scope_key] = deepcopy(result)
        self.scoped_candidate_pool_service.replace_from_ga_result(
            scope_key,
            result,
            source_label=f"GA top-N {scope_key}",
        )
        return deepcopy(result)

    def get_ga_result(self, scope: str) -> dict[str, Any]:
        """Return latest GA result for a ПО/ТО scope."""

        return deepcopy(self._ga_results.get(str(scope), {}))

    def get_candidate_pool_snapshot(self, scope: str) -> CandidatePoolSnapshot:
        """Return latest shared candidate-pool snapshot for a ПО/ТО scope."""

        snapshot = self.scoped_candidate_pool_service.get(scope)
        return CandidatePoolSnapshot(
            scope=snapshot.scope,
            candidates=deepcopy(snapshot.candidates),
            source_label=snapshot.source_label,
            source_method=snapshot.source_method,
        )

    def run_ahp_for_candidate_pool(self, scope: str) -> dict[str, Any]:
        """Rank the current scoped candidate pool with standalone AHP."""

        scope_key = str(scope)
        case = self.scoped_candidate_pool_service.to_criteria_case(scope_key)
        snapshot = self.scoped_candidate_pool_service.get(scope_key)
        alternatives = list(case.get("alternatives", []))
        criteria = list(case.get("criteria", []))
        scores = case.get("scores", {}) if isinstance(case.get("scores"), Mapping) else {}
        if not alternatives:
            report = {
                "status": "skipped",
                "reason": "Общий пул альтернатив пуст.",
                "candidate_count": 0,
                "final": {"ranking": []},
                "candidate_configurations": [],
            }
            self._ahp_results[scope_key] = deepcopy(report)
            self._pareto_results.pop(scope_key, None)
            self._hybrid_results.pop(scope_key, None)
            return deepcopy(report)
        if not criteria:
            report = {
                "status": "skipped",
                "reason": "Нет критериев AHP для текущей области.",
                "candidate_count": len(alternatives),
                "final": {"ranking": []},
                "candidate_configurations": deepcopy(snapshot.candidates),
            }
            self._ahp_results[scope_key] = deepcopy(report)
            self._pareto_results.pop(scope_key, None)
            self._hybrid_results.pop(scope_key, None)
            return deepcopy(report)

        profile = self.analysis_scope_profile_service.get_profile(scope_key)
        weights_by_id = {
            str(criterion_id): float(value)
            for criterion_id, value in profile.default_weights.items()
        }
        raw_weights = [max(0.0, weights_by_id.get(str(item.get("id")), 1.0)) for item in criteria]
        weights_total = sum(raw_weights) or float(len(raw_weights) or 1)
        criteria_weights = [value / weights_total for value in raw_weights]

        alt_weights_per_criterion: list[list[float]] = []
        consistency_per_criterion: list[dict[str, Any]] = []
        for criterion in criteria:
            criterion_id = str(criterion.get("id"))
            criterion_scores = [
                self._finite_float(
                    scores.get(str(alternative.get("id")), {}).get(criterion_id)
                    if isinstance(scores.get(str(alternative.get("id"))), Mapping)
                    else 0.0
                )
                for alternative in alternatives
            ]
            matrix = build_pairwise_matrix_from_scores(criterion_scores, saaty_cap=True)
            weights, lam, ci_value, cr_value = ahp_priority_and_consistency(matrix)
            alt_weights_per_criterion.append([float(value) for value in weights.tolist()])
            consistency_per_criterion.append(
                {
                    "criterion": criterion_id,
                    "lambda": float(lam),
                    "CI": float(ci_value),
                    "CR": float(cr_value),
                }
            )

        final_scores = []
        for alt_index, _alternative in enumerate(alternatives):
            score = 0.0
            for criterion_index, criterion_weight in enumerate(criteria_weights):
                score += criterion_weight * alt_weights_per_criterion[criterion_index][alt_index]
            final_scores.append(score)
        score_total = sum(final_scores)
        if score_total > 0:
            final_scores = [value / score_total for value in final_scores]
        elif final_scores:
            final_scores = [1.0 / len(final_scores) for _value in final_scores]

        candidate_by_id = {str(candidate.get("id")): candidate for candidate in snapshot.candidates}
        ranking = []
        ordered = sorted(
            enumerate(alternatives),
            key=lambda item: (-final_scores[item[0]], str(item[1].get("name") or item[1].get("id"))),
        )
        for rank, (alt_index, alternative) in enumerate(ordered, start=1):
            candidate_id = str(alternative.get("id") or f"candidate_{alt_index + 1}")
            candidate = candidate_by_id.get(candidate_id, {})
            metadata = candidate.get("metadata", {}) if isinstance(candidate.get("metadata"), Mapping) else {}
            metrics = candidate.get("metrics", {}) if isinstance(candidate.get("metrics"), Mapping) else {}
            ranking.append(
                {
                    "rank": rank,
                    "id": candidate_id,
                    "name": str(alternative.get("name") or candidate.get("name") or candidate_id),
                    "score": float(final_scores[alt_index]),
                    "ga_rank": self._positive_int(metadata.get("rank"), fallback=alt_index + 1),
                    "ga_score": metrics.get("ga_score", metrics.get("score")),
                    "totals": deepcopy(candidate.get("totals", {})),
                    "selected_items": deepcopy(candidate.get("components", [])),
                }
            )

        report = {
            "status": "ok",
            "method": "scoped_pool_standalone_ahp",
            "candidate_count": len(alternatives),
            "source_label": snapshot.source_label,
            "criteria": {
                "names": [str(item.get("id")) for item in criteria],
                "labels": [str(item.get("name") or item.get("id")) for item in criteria],
                "directions": [str(item.get("direction") or "max") for item in criteria],
                "weights": criteria_weights,
                "weights_source": "analysis_scope_profile",
            },
            "alternatives": {
                "ids": [str(item.get("id")) for item in alternatives],
                "weights_per_criterion": alt_weights_per_criterion,
                "consistency_per_criterion": consistency_per_criterion,
            },
            "final": {
                "ranking": ranking,
                "winner_id": ranking[0]["id"] if ranking else None,
                "winner_name": ranking[0]["name"] if ranking else None,
                "winner": ranking[0] if ranking else None,
            },
            "candidate_configurations": deepcopy(snapshot.candidates),
            "metadata": {
                "analysis_scope": scope_key,
                "candidate_pool_source": snapshot.source_label,
                "source_method": snapshot.source_method,
                "note": "Standalone AHP uses the current scoped candidate pool only.",
            },
        }
        self._ahp_results[scope_key] = deepcopy(report)
        self._pareto_results.pop(scope_key, None)
        self._hybrid_results.pop(scope_key, None)
        return deepcopy(report)

    def get_ahp_result(self, scope: str) -> dict[str, Any]:
        return deepcopy(self._ahp_results.get(str(scope), {}))

    def run_pareto_for_candidate_pool(self, scope: str) -> dict[str, Any]:
        """Run Pareto/criteria-importance over the current scoped candidate pool."""

        scope_key = str(scope)
        snapshot = self.scoped_candidate_pool_service.get(scope_key)
        if not snapshot.candidates:
            report = {
                "status": "skipped",
                "reason": "Общий пул альтернатив пуст.",
                "ranking": [],
                "final_nondominated": [],
                "candidate_configurations": [],
            }
            self._pareto_results[scope_key] = deepcopy(report)
            self._hybrid_results.pop(scope_key, None)
            return deepcopy(report)

        case = self.scoped_candidate_pool_service.to_criteria_case(scope_key)
        report = run_importance_pipeline(case)
        report["status"] = "ok"
        report.setdefault("metadata", {})
        if isinstance(report["metadata"], dict):
            report["metadata"].update(
                {
                    "analysis_scope": scope_key,
                    "candidate_pool_source": snapshot.source_label,
                    "source_method": snapshot.source_method,
                }
            )
        self._pareto_results[scope_key] = deepcopy(report)
        self._hybrid_results.pop(scope_key, None)
        return deepcopy(report)

    def get_pareto_result(self, scope: str) -> dict[str, Any]:
        return deepcopy(self._pareto_results.get(str(scope), {}))


    def run_hybrid_assessment(self, scope: str, *, lambda_value: float = 0.5) -> dict[str, Any]:
        """Build a Hybrid summary for the current scoped GA/AHP/Pareto state."""

        scope_key = str(scope)
        snapshot = self.scoped_candidate_pool_service.get(scope_key)
        result = self.hybrid_assessment_service.assess(
            snapshot.candidates,
            ahp_report=self.get_ahp_result(scope_key),
            pareto_report=self.get_pareto_result(scope_key) or None,
            lambda_value=lambda_value,
            source_label=snapshot.source_label,
        )
        result.setdefault("metadata", {})
        if isinstance(result["metadata"], dict):
            result["metadata"].update(
                {
                    "analysis_scope": scope_key,
                    "candidate_pool_source": snapshot.source_label,
                    "source_method": snapshot.source_method,
                }
            )
        self._hybrid_results[scope_key] = deepcopy(result)
        return deepcopy(result)

    def get_hybrid_result(self, scope: str) -> dict[str, Any]:
        return deepcopy(self._hybrid_results.get(str(scope), {}))

    @staticmethod
    def _finite_float(value: Any, default: float = 0.0) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return default
        return numeric if numeric == numeric and numeric not in {float("inf"), float("-inf")} else default

    @staticmethod
    def _positive_int(value: Any, *, fallback: int) -> int:
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            return fallback
        return numeric if numeric > 0 else fallback


    def generated_reports_dir(self) -> Path:
        export_root = self.paths.repo_root / "data" / "generated"
        export_root.mkdir(parents=True, exist_ok=True)
        return export_root

    def build_export_dashboard_summary(self) -> str:
        totals = self.prepare_cost_summary()
        lines = [
            "Сводка экспорта",
            "==============",
            f"CAPEX: {self._format_money(totals.get('total_capital', 0.0))}",
            f"OPEX разовый: {self._format_money(totals.get('total_operational_one_time', 0.0))}",
            f"OPEX месяц: {self._format_money(totals.get('total_operational_monthly', 0.0))}",
            f"Энергия: {self._format_money(totals.get('electricity_costs', 0.0))}",
        ]
        tco = totals.get("tco")
        if isinstance(tco, Mapping):
            lines.extend(
                [
                    f"TCO: {self._format_money(tco.get('total_ownership_cost', 0.0))}",
                    f"OPEX год: {self._format_money(tco.get('annual_opex', 0.0))}",
                ]
            )

        npv_report = self.get_npv_report()
        lines.append("")
        if npv_report:
            lines.append(f"NPV: {self._format_money(npv_report.get('npv', 0.0))}")
        else:
            lines.append("NPV: нет расчёта")

        lines.append("")
        lines.append(self._decision_status_line())
        return "\n".join(lines)

    def export_selected_fragment(self, mode: str) -> dict[str, Path]:
        export_root = self.generated_reports_dir()
        if mode == "full_decision_report":
            return self.export_decision_report()
        if mode == "raw_costs_csv":
            return {"csv": self.export_costs_csv(export_root / "total_costs.csv")}
        if mode == "cost_summary":
            return {
                "markdown": self._write_markdown_export(
                    export_root / "cost_summary_context.md",
                    "Сводка затрат",
                    self._cost_summary_markdown(),
                )
            }
        if mode == "technical_analysis":
            return {
                "markdown": self._write_markdown_export(
                    export_root / "technical_analysis_context.md",
                    "Аналитика ТО",
                    self._scope_analysis_markdown("technical"),
                )
            }
        if mode == "software_analysis":
            return {
                "markdown": self._write_markdown_export(
                    export_root / "software_analysis_context.md",
                    "Аналитика ПО",
                    self._scope_analysis_markdown("software"),
                )
            }
        if mode == "npv":
            return {
                "markdown": self._write_markdown_export(
                    export_root / "npv_context.md",
                    "NPV-контекст",
                    self._npv_markdown(),
                )
            }
        raise ValueError(f"Unknown export mode: {mode!r}")

    def export_costs_csv(self, filename: str | Path | None = None) -> Path:
        path = Path(filename) if filename is not None else self.generated_reports_dir() / "total_costs.csv"
        self.export_cost_report_use_case.execute(
            path,
            electricity_cost=self.get_electricity_cost(),
            electricity_profile=self.get_electricity_profile(),
        )
        return path

    def export_decision_report(self) -> dict[str, Path]:
        export_root = self.generated_reports_dir()
        scoped_inputs = self._scoped_decision_report_inputs()
        report = self.decision_report_builder.execute(
            project={
                "id": "current-it-solution-choice",
                "title": "Итоговый отчёт выбора ИТ-решения",
                "goal": "Свести компоненты, стоимость, NPV и анализ ПО/ТО.",
            },
            entities=self.entities_snapshot(),
            cost_totals=self.prepare_cost_summary(),
            candidate_configurations=scoped_inputs["candidate_configurations"],
            ahp_result=scoped_inputs["ahp_result"],
            criteria_importance_result=scoped_inputs["criteria_importance_result"],
            genetic_result=scoped_inputs["genetic_result"],
            genetic_ahp_result=None,
            hybrid_assessment_result=scoped_inputs["hybrid_assessment_result"],
            npv_report=self.get_npv_report() or None,
            metadata={
                "source": "qt_export_screen",
                "qt_migration_stage": "hybrid_screen",
                "scoped_analysis_state": self._scoped_analysis_state(),
            },
        )
        paths = {
            "json": export_decision_report_json(report, export_root / "decision_report.json"),
            "markdown": export_decision_report_markdown(report, export_root / "decision_report.md"),
            "csv": export_decision_report_csv(report, export_root / "decision_report_candidates.csv"),
            "components_csv": export_solution_component_csv(
                report,
                export_root / "decision_report_solution_components.csv",
            ),
        }
        return paths

    def open_generated_reports_folder(self) -> Path:
        export_root = self.generated_reports_dir()
        try:
            if sys.platform.startswith("win"):
                os.startfile(export_root)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(export_root)])
            else:
                subprocess.Popen(["xdg-open", str(export_root)])
        except Exception as exc:  # noqa: BLE001 - GUI needs a readable message.
            raise RuntimeError(f"Не удалось открыть папку: {export_root}") from exc
        return export_root


    def _scoped_decision_report_inputs(self) -> dict[str, Any]:
        scopes = ("software", "technical")
        ga_results: dict[str, dict[str, Any]] = {}
        ahp_results: dict[str, dict[str, Any]] = {}
        pareto_results: dict[str, dict[str, Any]] = {}
        hybrid_results: dict[str, dict[str, Any]] = {}
        candidates: list[dict[str, Any]] = []

        for scope in scopes:
            snapshot = self.scoped_candidate_pool_service.get(scope)
            for index, candidate in enumerate(snapshot.candidates, start=1):
                exported = deepcopy(candidate)
                metadata = dict(exported.get("metadata") or {})
                metadata.setdefault("candidate_pool_scope", scope)
                metadata.setdefault("candidate_pool_source", snapshot.source_label)
                metadata.setdefault("candidate_pool_method", snapshot.source_method)
                metadata.setdefault("candidate_pool_position", index)
                exported["metadata"] = metadata
                exported.setdefault("scope", scope)
                candidates.append(exported)

            ga_result = self.get_ga_result(scope)
            if ga_result:
                ga_results[scope] = ga_result
            ahp_result = self.get_ahp_result(scope)
            if ahp_result:
                ahp_results[scope] = ahp_result
            pareto_result = self.get_pareto_result(scope)
            if pareto_result:
                pareto_results[scope] = pareto_result
            hybrid_result = self.get_hybrid_result(scope)
            if hybrid_result:
                hybrid_results[scope] = hybrid_result

        return {
            "candidate_configurations": candidates or None,
            "ahp_result": {"by_scope": ahp_results} if ahp_results else None,
            "criteria_importance_result": {"by_scope": pareto_results} if pareto_results else None,
            "genetic_result": {"by_scope": ga_results} if ga_results else None,
            "hybrid_assessment_result": {"by_scope": hybrid_results} if hybrid_results else None,
        }

    def _scoped_analysis_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {}
        for scope in ("software", "technical"):
            snapshot = self.scoped_candidate_pool_service.get(scope)
            state[scope] = {
                "candidate_pool_count": len(snapshot.candidates),
                "ga_status": self.get_ga_result(scope).get("status", "not_run"),
                "ahp_status": self.get_ahp_result(scope).get("status", "not_run"),
                "pareto_status": self.get_pareto_result(scope).get("status", "not_run"),
                "hybrid_status": self.get_hybrid_result(scope).get("status", "not_run"),
            }
        return state

    def _decision_status_line(self) -> str:
        parts = []
        for scope, label in (("software", "ПО"), ("technical", "ТО")):
            hybrid = self.get_hybrid_result(scope)
            if hybrid.get("status") == "ok":
                parts.append(f"{label}: Hybrid готов")
            elif self.get_pareto_result(scope).get("status") == "ok":
                parts.append(f"{label}: ждёт Hybrid")
        return "; ".join(parts) if parts else "ПО/ТО: нет итогового Hybrid."

    def _write_markdown_export(self, path: Path, title: str, body: str) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {title}\n\n{body.rstrip()}\n", encoding="utf-8")
        return path

    def _cost_summary_markdown(self) -> str:
        totals = self.prepare_cost_summary()
        payload = json.dumps(totals, ensure_ascii=False, indent=2, default=str)
        return (
            "Фрагмент фиксирует только затраты. Полное обоснование требует GA, AHP, "
            "Pareto, Hybrid и NPV.\n\n"
            f"- CAPEX: {self._format_money(totals.get('total_capital', 0.0))}\n"
            f"- OPEX разовый: {self._format_money(totals.get('total_operational_one_time', 0.0))}\n"
            f"- OPEX месяц: {self._format_money(totals.get('total_operational_monthly', 0.0))}\n"
            f"- Энергия: {self._format_money(totals.get('electricity_costs', 0.0))}\n\n"
            "```json\n"
            f"{payload}\n"
            "```"
        )

    def _scope_analysis_markdown(self, scope: str) -> str:
        labels = {"technical": "ТО", "software": "ПО"}
        return (
            f"Аналитика {labels.get(scope, scope)} будет доступна после переноса маршрута "
            "Данные → GA → AHP → Pareto → Гибрид. Этот файл создан как маркер "
            "частичного экспорта в Qt-контуре."
        )

    def _npv_markdown(self) -> str:
        npv_report = self.get_npv_report()
        if not npv_report:
            return "NPV ещё не рассчитан. Сначала выполните NPV-анализ."
        return (
            "Экспортируется только NPV. Он не заменяет GA/AHP/Pareto/Hybrid.\n\n"
            "```json\n"
            f"{json.dumps(npv_report, ensure_ascii=False, indent=2, default=str)}\n"
            "```"
        )

    @staticmethod
    def _format_money(value: Any) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = 0.0
        return f"{number:,.2f} ₽".replace(",", " ")

    def table_rows(self, entity_name: str) -> list[dict[str, Any]]:
        """Return plain rows ready for Qt table models."""

        return self.list_entities(entity_name)
