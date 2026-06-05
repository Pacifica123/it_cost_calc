from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from application.services.cost_aggregation_service import CostAggregationService
from application.services.entity_catalog_service import EntityCatalogService
from application.services.electricity_cost_service import ElectricityCostService
from application.services.equipment_service import EquipmentService
from application.services.npv_report_service import NPVReportService
from application.services.solution_component_runtime_service import SolutionComponentRuntimeService
from application.use_cases.build_npv_report import BuildNpvReportUseCase
from application.use_cases.calculate_electricity_costs import CalculateElectricityCostsUseCase
from application.use_cases.load_demo_dataset import LoadDemoDatasetUseCase
from application.use_cases.prepare_cost_summary import PrepareCostSummaryUseCase
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
        self.npv_report_builder = BuildNpvReportUseCase(NPVReportService())
        self.demo_loader = LoadDemoDatasetUseCase(self.storage, self.equipment_service)
        self._electricity_profile: dict[str, float] = {
            "hours_per_day": 8.0,
            "working_days": 22.0,
            "cost_per_kwh": 1.0,
        }
        self._electricity_report: dict[str, Any] = {"total_cost": 0.0, "items": []}
        self._npv_report: dict[str, Any] = {}

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

    def table_rows(self, entity_name: str) -> list[dict[str, Any]]:
        """Return plain rows ready for Qt table models."""

        return self.list_entities(entity_name)
