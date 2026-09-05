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

from application.services.catalog_source_registry import CatalogSourceRegistry  # noqa: E402
from application.services.catalog_staging_service import CatalogStagingService  # noqa: E402


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="itcost-feed-check-") as raw_tmp:
        tmp = Path(raw_tmp)
        feed = tmp / "catalog.yml"
        feed.write_text(
            """<yml_catalog><shop><categories><category id="1">Маршрутизаторы</category></categories>
<offers><offer id="r1" available="true"><name>Router R1</name><vendor>Vendor</vendor>
<vendorCode>R1</vendorCode><categoryId>1</categoryId><price>10000</price>
<currencyId>RUB</currencyId></offer></offers></shop></yml_catalog>""",
            encoding="utf-8",
        )
        records = CatalogStagingService(tmp / "staging.json").stage_file(
            feed,
            source_context={
                "id": "smoke-supplier",
                "name": "Smoke Supplier",
                "location": "https://example.invalid/catalog.yml",
                "format": "yml",
                "price_kind": "supplier_price",
                "observed_at": "2026-09-05T00:00:00+00:00",
            },
        )
        if len(records) != 1:
            raise SystemExit("feed smoke: expected one record")
        item = records[0]["catalog_item"]
        if item["category"] != "router" or item["source"] != "smoke-supplier":
            raise SystemExit("feed smoke: normalization failed")
        if item["offer"].get("observed_at") != "2026-09-05T00:00:00+00:00":
            raise SystemExit("feed smoke: observed_at lost")
        if item["field_provenance"].get("price", {}).get("method") != "feed:yml":
            raise SystemExit("feed smoke: provenance lost")

        presets = tmp / "presets.json"
        presets.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "sources": [
                        {
                            "id": "preset",
                            "name": "Preset",
                            "location": "https://example.invalid/price.xlsx",
                            "format": "xlsx",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        registry = CatalogSourceRegistry(tmp / "sources.json", presets_path=presets)
        registry.save_source(
            {
                "id": "custom",
                "name": "Custom",
                "location": str(feed),
                "format": "yml",
            }
        )
        if {source["id"] for source in registry.list_sources()} != {"preset", "custom"}:
            raise SystemExit("feed smoke: registry merge failed")

    print("Catalog feed ingestion smoke: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
