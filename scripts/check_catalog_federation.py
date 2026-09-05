from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from application.services.catalog_federation_service import federate_catalog_items  # noqa: E402
from application.services.catalog_staging_service import CatalogStagingService  # noqa: E402


def _row(source: str, product_id: str, price: int, observed_at: str) -> dict:
    return {
        "item_id": f"{source}-{product_id}",
        "title": "MikroTik Router R1",
        "category": "router",
        "source": source,
        "source_product_id": product_id,
        "identity": {"brand": "MikroTik", "mpn": "R1-PN", "model": "Router R1"},
        "offer": {
            "price": price,
            "currency": "RUB",
            "availability": "in_stock",
            "observed_at": observed_at,
            "price_kind": "supplier_price",
        },
        "attributes": {"lan_ports": 4, "max_power_watts": 12},
    }


def main() -> int:
    merged = federate_catalog_items(
        [
            _row("supplier-a", "a-1", 12000, "2026-09-04T08:00:00+00:00"),
            _row("supplier-b", "b-1", 11500, "2026-09-05T08:00:00+00:00"),
        ]
    )
    assert len(merged) == 1
    assert len(merged[0]["offers"]) == 2
    assert merged[0]["offer"]["source"] == "supplier-b"
    assert merged[0]["price_summary"]["median_rub"] == 11750

    with tempfile.TemporaryDirectory(prefix="catalog_federation_check_") as temp:
        base = Path(temp)
        first = base / "first.json"
        second = base / "second.json"
        first.write_text(
            json.dumps({"schema_version": 2, "items": [_row(
                "supplier-a", "a-1", 12000, "2026-09-04T08:00:00+00:00"
            )]}),
            encoding="utf-8",
        )
        second.write_text(
            json.dumps({"schema_version": 2, "items": [_row(
                "supplier-b", "b-1", 11500, "2026-09-05T08:00:00+00:00"
            )]}),
            encoding="utf-8",
        )
        service = CatalogStagingService(base / "staging.json")
        initial = service.stage_file(first)
        initial_id = initial[0]["staging_id"]
        federated = service.stage_file(second)
        assert len(federated) == 1
        assert federated[0]["staging_id"] == initial_id
        assert federated[0]["catalog_item"]["federation"]["source_count"] == 2

    print("Catalog federation check passed: strict identity, offers[], effective price and refresh.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
