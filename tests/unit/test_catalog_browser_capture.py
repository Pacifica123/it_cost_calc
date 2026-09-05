from __future__ import annotations

import json
from pathlib import Path

from application.services.catalog_browser_capture_service import capture_browser_content
from application.services.catalog_staging_service import CatalogStagingService


HTML = """
<html><head><title>Router R1</title>
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "Product",
  "name": "MikroTik Router R1",
  "brand": {"@type": "Brand", "name": "MikroTik"},
  "model": "Router R1",
  "mpn": "R1-PN",
  "sku": "SKU-77",
  "category": "router",
  "offers": {
    "@type": "Offer",
    "price": "14990",
    "priceCurrency": "RUB",
    "availability": "https://schema.org/InStock",
    "url": "https://shop.example/r1"
  }
}
</script></head><body></body></html>
"""


def test_browser_capture_uses_saved_structured_data_without_network() -> None:
    captured = capture_browser_content(HTML, source_url="https://shop.example/r1", region="Кемерово")
    assert captured.item["title"] == "MikroTik Router R1"
    assert captured.item["identity"]["mpn"] == "R1-PN"
    assert captured.item["offer"]["price"] == 14990.0
    assert captured.item["offer"]["availability"] == "in_stock"
    assert captured.source_context["capture_method"] == "ordinary_browser"
    assert captured.source_context["location"] == "https://shop.example/r1"


def test_browser_capture_can_be_staged_and_federated(tmp_path: Path) -> None:
    supplier = tmp_path / "supplier.json"
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
                        "offer": {
                            "price": 15100,
                            "currency": "RUB",
                            "availability": "in_stock",
                            "observed_at": "2026-09-05T00:00:00+00:00",
                            "price_kind": "supplier_price",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    staging = CatalogStagingService(tmp_path / "staging.json")
    staging.stage_file(supplier)

    captured = capture_browser_content(HTML, source_url="https://shop.example/r1")
    capture_file = tmp_path / "capture.json"
    capture_file.write_text(
        json.dumps({"schema_version": 2, "items": [captured.item]}),
        encoding="utf-8",
    )
    staging.stage_file(capture_file, source_context=captured.source_context)
    item = staging.list_records()[0]["catalog_item"]
    assert item["federation"]["source_count"] == 2
    assert len(item["offers"]) == 2
    # Supplier price outranks retail capture even when capture is newer.
    assert item["offer"]["price_kind"] == "supplier_price"
