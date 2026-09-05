from __future__ import annotations

import csv
import json
from pathlib import Path

from application.services.catalog_commercial_quote_service import import_commercial_quote
from application.services.catalog_staging_service import CatalogStagingService


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
        "attributes": {"lan_ports": 4},
    }


def test_commercial_quote_becomes_highest_trust_effective_offer(tmp_path: Path) -> None:
    supplier = tmp_path / "supplier.json"
    supplier.write_text(json.dumps({"schema_version": 2, "items": [_supplier_item()]}), encoding="utf-8")
    staging = CatalogStagingService(tmp_path / "staging.json")
    staging.stage_file(supplier)

    quote = tmp_path / "quote.csv"
    with quote.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["Наименование", "Цена", "Бренд", "Модель", "Артикул производителя"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "Наименование": "MikroTik Router R1",
                "Цена": "16200",
                "Бренд": "MikroTik",
                "Модель": "Router R1",
                "Артикул производителя": "R1-PN",
            }
        )

    summary = import_commercial_quote(
        staging,
        quote,
        supplier_name="ООО Интегратор",
        quote_number="KP-42",
        quote_date="2026-08-20",
        region="Кемеровская область",
    )
    assert summary.records_total == 1

    item = staging.list_records()[0]["catalog_item"]
    assert len(item["offers"]) == 2
    assert item["offer"]["price_kind"] == "commercial_quote"
    assert item["offer"]["price"] == 16200.0
    assert item["offer"]["availability"] == "quoted"
    provenance = item["field_provenance"]["feed"]
    assert provenance["source_type"] == "commercial_quote"
    assert provenance["quote_number"] == "KP-42"
