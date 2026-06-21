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

from application.services.catalog_staging_service import (  # noqa: E402
    CatalogStagingService,
    STAGING_APPROVED,
    STAGING_BLOCKED,
    catalog_item_to_runtime_row,
)
from tools.catalog_parser.sources.dns_examples import (  # noqa: E402
    build_catalog_from_example_snapshots,
)


def main() -> int:
    catalog = build_catalog_from_example_snapshots()
    assert catalog["schema_version"] == 2
    assert catalog["stats"]["items_total"] >= 100
    assert all("offer" in item and "identity" in item for item in catalog["items"])

    with tempfile.TemporaryDirectory(prefix="catalog_staging_check_") as temp_dir:
        root = Path(temp_dir)
        source = root / "catalog.json"
        source.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "items": [
                        {
                            "item_id": "check-router",
                            "title": "Check Router",
                            "category": "router",
                            "source": "smoke",
                            "identity": {"brand": "Check", "model": "R1"},
                            "offer": {
                                "price": 5000,
                                "currency": "RUB",
                                "observed_at": "2026-06-21T12:00:00Z",
                            },
                            "attributes": {
                                "lan_ports": 4,
                                "lan_speed_mbps": 1000,
                                "max_power_watts": 12,
                            },
                        },
                        {
                            "item_id": "check-cpu",
                            "title": "Check CPU",
                            "category": "cpu",
                            "source": "smoke",
                            "offer": {
                                "price": 20000,
                                "currency": "RUB",
                                "observed_at": "2026-06-21T12:00:00Z",
                            },
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        service = CatalogStagingService(root / "staging.json")
        records = service.stage_file(source)
        assert records[0]["status"] != STAGING_BLOCKED
        assert records[1]["status"] == STAGING_BLOCKED
        approved = service.set_status(records[0]["staging_id"], STAGING_APPROVED)
        category, row = catalog_item_to_runtime_row(approved)
        assert category == "network"
        assert row["origin"] == "catalog"
        assert row["lan_ports"] == 4
        edited = service.update_record(
            records[1]["staging_id"],
            {
                "title": "Check Workstation",
                "category": "prebuilt_pc",
                "price": "65000",
                "currency": "RUB",
                "target_category": "client",
                "target_component_type": "workstation",
                "quantity": "2",
                "client_seats": "2",
                "attributes": {
                    "ram_gb": "16",
                    "cpu_cores": "8",
                    "storage_gb": "512",
                    "max_power_watts": "250",
                },
            },
        )
        assert edited["status"] == "pending"
        assert not edited["validation_errors"]
        assert edited["source_catalog_item"]["category"] == "cpu"
        approved_workstation = service.set_status(
            edited["staging_id"],
            STAGING_APPROVED,
        )
        category, workstation = catalog_item_to_runtime_row(approved_workstation)
        assert category == "client"
        assert workstation["quantity"] == 2
        assert workstation["client_seats"] == 2

    print(
        "Catalog staging check passed: "
        f"schema v2, {catalog['stats']['items_total']} example items, "
        "review overrides and guarded runtime import."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
