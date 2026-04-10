from pathlib import Path

from bootstrap import load_demo_data
from application.services.cost_aggregation_service import CostAggregationService
from application.services.equipment_service import EquipmentService
from application.use_cases.export_cost_report import ExportCostReportUseCase
from application.use_cases.prepare_cost_summary import PrepareCostSummaryUseCase
from infrastructure.repositories.json_entity_repository import JsonEntityRepository
from infrastructure.storage import JsonFileStorage


def test_demo_dataset_can_be_loaded_and_exported_to_csv(tmp_path: Path):
    root = Path(__file__).resolve().parents[2]
    runtime_path = tmp_path / "runtime_entities.json"
    export_path = tmp_path / "total_costs.csv"

    dataset = load_demo_data(
        repo_root=root,
        fixture_path=root / "data" / "fixtures" / "demo_dataset.json",
        runtime_entities_path=runtime_path,
    )

    repository = JsonEntityRepository(runtime_path, JsonFileStorage())
    equipment_service = EquipmentService(repository)
    aggregation_service = CostAggregationService()
    prepare_cost_summary = PrepareCostSummaryUseCase(aggregation_service, equipment_service)
    export_cost_report = ExportCostReportUseCase(prepare_cost_summary, aggregation_service)

    totals = export_cost_report.execute(export_path, electricity_cost=321.5)
    exported_text = export_path.read_text(encoding="utf-8")

    assert runtime_path.exists()
    assert export_path.exists()
    assert dataset["server"][0]["name"] == "Rock w9102p"
    assert totals == {
        "total_capital": 1084392.0,
        "total_operational_one_time": 0.0,
        "total_operational_monthly": 26104.0,
        "electricity_costs": 321.5,
    }
    assert "ОБЩИЙ ИТОГ" in exported_text
    assert "Rock w9102p" in exported_text
