"""Portable P3 smoke: no network, pytest, Qt or third-party imports required."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from application.services.catalog_enrichment_service import (  # noqa: E402
    apply_specification_source,
    parse_icecat_specification,
)
from application.services.catalog_staging_service import CatalogStagingService  # noqa: E402


def feature(name: str, value, unit: str = "") -> dict:
    return {
        "RawValue": value,
        "Feature": {
            "Name": {"Value": name},
            "Measure": {"Signs": {"_": unit}},
        },
    }


def main() -> int:
    payload = {
        "msg": "OK",
        "data": {
            "GeneralInfo": {
                "IcecatId": 42,
                "Brand": "Vendor",
                "BrandPartCode": "R-1",
                "GTIN": ["4601234567890"],
            },
            "FeaturesGroups": [{
                "Features": [
                    feature("Ethernet LAN (RJ-45) ports", 4),
                    feature("Ethernet LAN data rates", "10,100,1000", "Mbit/s"),
                    feature("Power consumption (max)", 15, "W"),
                    feature("IPv6 support", "Yes"),
                ]
            }],
        },
    }
    specification = parse_icecat_specification(
        payload,
        requested_identity={"gtin": "4601234567890"},
        matched_by="gtin",
        observed_at="2026-09-05T08:00:00+00:00",
    )
    assert specification["metrics"]["lan_ports"] == 4
    assert specification["metrics"]["lan_speed_mbps"] == 1000
    assert specification["metrics"]["ipv6_support"] is True

    with tempfile.TemporaryDirectory(prefix="catalog-icecat-") as temp_dir:
        root = Path(temp_dir)
        feed = root / "feed.json"
        feed.write_text(
            json.dumps({
                "schema_version": 2,
                "items": [{
                    "item_id": "router-1",
                    "title": "Vendor Router",
                    "category": "router",
                    "source": "supplier",
                    "source_product_id": "r1",
                    "identity": {"brand": "Vendor", "mpn": "R-1", "gtin": "4601234567890"},
                    "offer": {"price": 9000, "currency": "RUB", "price_kind": "supplier_price"},
                    "attributes": {"max_power_watts": 12},
                }],
            }),
            encoding="utf-8",
        )
        service = CatalogStagingService(root / "staging.json")
        record = service.stage_file(feed)[0]
        enriched = apply_specification_source(record["source_catalog_item"], specification)
        updated = service.apply_source_item_updates({record["staging_id"]: enriched})[0]
        assert updated["catalog_item"]["attributes"]["lan_ports"] == 4
        assert updated["catalog_item"]["attributes"]["max_power_watts"] == 12
        assert "attributes.max_power_watts" in updated["catalog_item"]["specification_summary"]["conflicts"]

        # Supplier refresh must not discard the specification layer.
        feed_payload = json.loads(feed.read_text(encoding="utf-8"))
        feed_payload["items"][0]["offer"]["price"] = 8500
        feed.write_text(json.dumps(feed_payload), encoding="utf-8")
        refreshed = service.stage_file(feed)[0]
        assert refreshed["catalog_item"]["offer"]["price"] == 8500
        assert refreshed["catalog_item"]["attributes"]["lan_ports"] == 4
        assert refreshed["catalog_item"]["specification_sources"][0]["source"] == "icecat"

    print("Catalog Icecat P3 smoke: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
