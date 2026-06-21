from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from application.services.catalog_staging_service import (  # noqa: E402
    CatalogStagingService,
    STAGING_APPROVED,
    catalog_item_to_runtime_row,
)
from application.services.decision_report_service import DecisionReportService  # noqa: E402
from infrastructure.exporters.decision_report_exporter import (  # noqa: E402
    build_decision_report_csv_rows,
    build_decision_report_json_payload,
    build_decision_report_markdown,
)
from tools.catalog_parser.catalog_builder import save_catalog  # noqa: E402
from tools.catalog_parser.sources.dns_snapshot import build_catalog_from_dns_snapshot  # noqa: E402


def main() -> int:
    snapshot = ROOT / "data" / "examples" / "parser" / "dns_snapshot"
    catalog = build_catalog_from_dns_snapshot(snapshot)

    with tempfile.TemporaryDirectory(prefix="catalog_report_check_") as temp_dir:
        root = Path(temp_dir)
        catalog_path = save_catalog(catalog, root / "catalog.json")
        staging = CatalogStagingService(root / "staging.json")
        records = staging.stage_file(catalog_path)
        entities: dict[str, list[dict]] = {}
        for record in records:
            approved = staging.set_status(record["staging_id"], STAGING_APPROVED)
            category, runtime_row = catalog_item_to_runtime_row(approved)
            entities.setdefault(category, []).append(runtime_row)

        report = DecisionReportService().build_report(entities=entities)
        quality = report["catalog_data_quality"]
        summary = quality["summary"]
        assert summary["catalog_components_total"] == 2
        assert summary["complete_metrics"] == 2
        assert summary["with_warnings"] == 1

        router = next(
            item for item in quality["components"] if item["component_type"] == "network_device"
        )
        assert router["source"] == "dns"
        assert router["parse_source"] == "dns-product-specs+title-best-effort"
        assert router["confidence"] > 0
        assert router["field_provenance"]
        assert any("Wi-Fi standards" in warning for warning in router["metric_warnings"])

        payload = build_decision_report_json_payload(report)
        markdown = build_decision_report_markdown(report)
        csv_rows = build_decision_report_csv_rows(report)
        assert payload["catalog_data_quality"]["summary"] == summary
        assert "Качество данных каталога" in markdown
        assert "dns-product-specs+title-best-effort" in markdown
        assert "technical_metrics" in csv_rows[0]
        assert "metric_warnings" in csv_rows[0]

    print(
        "Catalog DecisionReport check passed: DNS snapshot, staging, provenance, "
        "metric completeness and JSON/Markdown/CSV exports."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
