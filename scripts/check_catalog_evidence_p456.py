"""Portable P4/P5/P6 smoke check; requires only the Python standard library."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from application.services.catalog_browser_capture_service import capture_browser_content
from application.services.catalog_commercial_quote_service import import_commercial_quote
from application.services.catalog_procurement_benchmark_service import (
    apply_procurement_benchmarks,
    load_procurement_observations,
)
from application.services.catalog_staging_service import CatalogStagingService


def main() -> int:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        staging = CatalogStagingService(root / "staging.json")
        supplier = root / "supplier.json"
        supplier.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "items": [
                        {
                            "title": "MikroTik Router R1",
                            "category": "router",
                            "source": "supplier-a",
                            "source_product_id": "r1",
                            "identity": {"brand": "MikroTik", "model": "Router R1", "mpn": "R1-PN"},
                            "offer": {"price": 15000, "currency": "RUB", "availability": "in_stock", "observed_at": "2026-09-05T00:00:00+00:00", "price_kind": "supplier_price"},
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        staging.stage_file(supplier)

        quote = root / "quote.csv"
        with quote.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["name", "price", "brand", "model", "mpn"])
            writer.writeheader()
            writer.writerow({"name": "MikroTik Router R1", "price": "16200", "brand": "MikroTik", "model": "Router R1", "mpn": "R1-PN"})
        import_commercial_quote(staging, quote, supplier_name="Integrator", quote_number="42", quote_date="2026-08-20")
        assert staging.list_records()[0]["catalog_item"]["offer"]["price_kind"] == "commercial_quote"

        xml = root / "eis.xml"
        xml.write_text("<root><product><name>MikroTik Router R1 R1-PN</name><unitPrice>13000</unitPrice></product></root>", encoding="utf-8")
        observations = load_procurement_observations(xml)
        apply_procurement_benchmarks(staging, observations)
        item = staging.list_records()[0]["catalog_item"]
        assert item["procurement_benchmark"]["median_rub"] == 13000.0
        assert item["offer"]["price"] == 16200.0

        html = '<script type="application/ld+json">{"@type":"Product","name":"MikroTik Router R1","brand":{"name":"MikroTik"},"mpn":"R1-PN","model":"Router R1","category":"router","offers":{"price":"14990","priceCurrency":"RUB","availability":"https://schema.org/InStock"}}</script>'
        captured = capture_browser_content(html, source_url="https://shop.example/r1")
        assert captured.item["identity"]["mpn"] == "R1-PN"
        assert captured.item["offer"]["price"] == 14990.0

    print("P4/P5/P6 portable smoke: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
