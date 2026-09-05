from __future__ import annotations

import json
from pathlib import Path

from application.services.catalog_procurement_benchmark_service import (
    apply_procurement_benchmarks,
    load_procurement_observations,
)
from application.services.catalog_staging_service import CatalogStagingService
from application.services.decision_report_service import DecisionReportService
from infrastructure.exporters.decision_report_exporter import build_decision_report_markdown


def _supplier_item() -> dict:
    return {
        "item_id": "supplier-router-r1",
        "title": "MikroTik Router R1",
        "category": "router",
        "source": "supplier-a",
        "source_product_id": "r1",
        "identity": {"brand": "MikroTik", "model": "Router R1", "mpn": "R1-PN"},
        "offer": {
            "price": 15000,
            "currency": "RUB",
            "availability": "in_stock",
            "observed_at": "2026-09-05T00:00:00+00:00",
            "price_kind": "supplier_price",
        },
        "attributes": {"lan_ports": 4, "max_power_watts": 12},
    }


def test_eis_benchmark_is_statistical_evidence_not_offer(tmp_path: Path) -> None:
    supplier = tmp_path / "supplier.json"
    supplier.write_text(json.dumps({"schema_version": 2, "items": [_supplier_item()]}), encoding="utf-8")
    staging = CatalogStagingService(tmp_path / "staging.json")
    staging.stage_file(supplier)

    xml = tmp_path / "contracts.xml"
    xml.write_text(
        """<?xml version='1.0' encoding='UTF-8'?>
        <contract>
          <regNum>0123456789</regNum>
          <publishDate>2026-08-20</publishDate>
          <product><name>MikroTik Router R1 R1-PN</name><unitPrice>12000</unitPrice><quantity>2</quantity></product>
          <product><name>MikroTik Router R1 R1-PN</name><unitPrice>14000</unitPrice><quantity>1</quantity></product>
        </contract>""",
        encoding="utf-8",
    )

    observations = load_procurement_observations(xml)
    assert [item.unit_price_rub for item in observations] == [12000.0, 14000.0]

    manifest = tmp_path / "manifest.json"
    summary = apply_procurement_benchmarks(
        staging,
        observations,
        source_location=str(xml),
        manifest_path=manifest,
    )
    assert summary.matched_records == 1
    assert summary.identity_matches == 1

    record = staging.list_records()[0]
    item = record["catalog_item"]
    benchmark = item["procurement_benchmark"]
    assert benchmark["median_rub"] == 13000.0
    assert benchmark["observation_count"] == 2
    assert benchmark["match_level"] == "identity"
    assert item["offer"]["price"] == 15000
    assert len(item["offers"]) == 1
    assert manifest.exists()


def test_procurement_benchmark_reaches_runtime_and_decision_report(tmp_path: Path) -> None:
    supplier = tmp_path / "supplier.json"
    supplier.write_text(json.dumps({"schema_version": 2, "items": [_supplier_item()]}), encoding="utf-8")
    staging = CatalogStagingService(tmp_path / "staging.json")
    records = staging.stage_file(supplier)
    staging_id = records[0]["staging_id"]

    xml = tmp_path / "contracts.xml"
    xml.write_text(
        "<root><product><name>MikroTik Router R1 R1-PN</name><unitPrice>12500</unitPrice></product></root>",
        encoding="utf-8",
    )
    apply_procurement_benchmarks(staging, load_procurement_observations(xml))
    staging.set_status(staging_id, "approved")
    _target, row = __import__(
        "application.services.catalog_staging_service", fromlist=["catalog_item_to_runtime_row"]
    ).catalog_item_to_runtime_row(staging.list_records()[0])

    report = DecisionReportService().build_report(entities={"network": [row]})
    component = report["catalog_data_quality"]["components"][0]
    assert component["procurement_benchmark"]["median_rub"] == 12500.0
    markdown = build_decision_report_markdown(report)
    assert "ЕИС median" in markdown
    assert "12 500" in markdown or "12500" in markdown
