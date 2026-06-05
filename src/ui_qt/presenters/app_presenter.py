from __future__ import annotations

from copy import deepcopy
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from application.services.cost_aggregation_service import CostAggregationService
from application.services.decision_report_service import DecisionReportService
from application.services.entity_catalog_service import EntityCatalogService
from application.services.genetic_optimization_service import GeneticOptimizationService
from application.services.electricity_cost_service import ElectricityCostService
from application.services.equipment_service import EquipmentService
from application.services.npv_report_service import NPVReportService
from application.services.scoped_candidate_pool_service import CandidatePoolSnapshot, ScopedCandidatePoolService
from application.services.solution_component_runtime_service import SolutionComponentRuntimeService
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

    @classmethod
    def from_repo_root(
        cls,
        repo_root: str | Path | None = None,
        *,
        runtime_entities_path: str | Path | None = None,
        demo_dataset_path: str | Path | None = None,
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
        storage: JsonFileStorage | None = None,
        repository: JsonEntityRepository | None = None,
    ) -> None:
        self.paths = QtRuntimePaths.from_repo_root(
            repo_root,
            runtime_entities_path=runtime_entities_path,
            demo_dataset_path=demo_dataset_path,
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
        self.genetic_optimizer = RunGeneticOptimizationUseCase(
            self.genetic_optimization_service,
        )
        self.scoped_candidate_pool_service = ScopedCandidatePoolService()
        self.demo_loader = LoadDemoDatasetUseCase(self.storage, self.equipment_service)
        self._electricity_profile: dict[str, float] = {
            "hours_per_day": 8.0,
            "working_days": 22.0,
            "cost_per_kwh": 1.0,
        }
        self._electricity_report: dict[str, Any] = {"total_cost": 0.0, "items": []}
        self._npv_report: dict[str, Any] = {}
        self._ga_results: dict[str, dict[str, Any]] = {}

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
        ga_params: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run GA for one analysis scope and update the shared candidate pool."""

        result = self.genetic_optimizer.execute(
            source=self.repository,
            analysis_scope=scope,
            max_budget=max_budget,
            max_power=max_power,
            target_units=target_units,
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
        lines.append("GA/AHP/Pareto/Hybrid будут подключены после переноса ПО/ТО.")
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
        report = self.decision_report_builder.execute(
            project={
                "id": "current-it-solution-choice",
                "title": "Итоговый отчёт выбора ИТ-решения",
                "goal": "Свести компоненты, стоимость, NPV и будущий анализ ПО/ТО.",
            },
            entities=self.entities_snapshot(),
            cost_totals=self.prepare_cost_summary(),
            candidate_configurations=None,
            ahp_result=None,
            criteria_importance_result=None,
            genetic_result=None,
            genetic_ahp_result=None,
            hybrid_assessment_result=None,
            npv_report=self.get_npv_report() or None,
            metadata={
                "source": "qt_export_screen",
                "qt_migration_stage": "export_screen",
                "analysis_note": "ПО/ТО analysis screens are not migrated yet.",
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
