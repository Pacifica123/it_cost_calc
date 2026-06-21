from application.services.decision_report_service import DecisionReportService
from domain import DecisionReport
from infrastructure.exporters.decision_report_exporter import (
    build_decision_report_csv_rows,
    build_decision_report_json_payload,
    build_decision_report_markdown,
)


def _catalog_workstation() -> dict:
    return {
        "name": "Catalog Workstation",
        "quantity": 1,
        "price": 70000,
        "scope": "technical",
        "component_type": "workstation",
        "origin": "catalog",
        "catalog_item_id": "workstation-1",
        "ram_gb": 16,
        "cpu_cores": 8,
        "max_power_watts": 250,
        "metric_warnings": ["compute metric storage_gb is missing"],
        "catalog_metadata": {
            "source": "dns",
            "source_product_id": "source-1",
            "offer": {"observed_at": "2026-06-21T12:00:00Z"},
            "field_provenance": {"attributes": {"method": "dns-product-specs"}},
            "parser_metadata": {
                "parse_source": "dns-product-specs",
                "confidence": 0.77,
                "parse_warnings": ["storage type was not recognized"],
            },
            "manual_overrides": {"price": 70000},
        },
    }


def test_decision_report_exposes_catalog_quality_and_provenance() -> None:
    report = DecisionReportService().build_report(entities={"client": [_catalog_workstation()]})
    quality = report["catalog_data_quality"]

    assert quality["summary"] == {
        "catalog_components_total": 1,
        "complete_metrics": 0,
        "incomplete_metrics": 1,
        "with_warnings": 1,
        "with_manual_overrides": 1,
    }
    component = quality["components"][0]
    assert component["missing_metrics"] == ["storage_gb"]
    assert component["confidence"] == 0.77
    assert component["manual_override_fields"] == ["price"]
    assert any("неполные технические метрики" in warning for warning in report["warnings"])

    restored = DecisionReport.from_dict(report).to_dict()
    assert restored["catalog_data_quality"] == quality


def test_catalog_quality_is_visible_in_all_decision_report_exports() -> None:
    report = DecisionReportService().build_report(entities={"client": [_catalog_workstation()]})
    payload = build_decision_report_json_payload(report)
    markdown = build_decision_report_markdown(report)
    csv_rows = build_decision_report_csv_rows(report)

    assert payload["catalog_data_quality"]["summary"]["incomplete_metrics"] == 1
    assert "Качество данных каталога" in markdown
    assert "storage_gb" in markdown
    assert "total_ram_gb" in csv_rows[0]["technical_metrics"]
    assert "storage_gb is missing" in csv_rows[0]["metric_warnings"]
