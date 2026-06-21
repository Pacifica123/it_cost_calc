from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from application.services.catalog_staging_service import CatalogStagingService  # noqa: E402
from tools.catalog_parser.catalog_builder import save_catalog  # noqa: E402
from tools.catalog_parser.sources.dns_snapshot import build_catalog_from_dns_snapshot  # noqa: E402


def main() -> int:
    snapshot = ROOT / "data" / "examples" / "parser" / "dns_snapshot"
    catalog = build_catalog_from_dns_snapshot(snapshot)
    assert catalog["schema_version"] == 2
    assert catalog["stats"]["items_total"] == 2

    by_category = {item["category"]: item for item in catalog["items"]}
    assert by_category["router"]["attributes"]["lan_ports"] == 4
    assert by_category["prebuilt_pc"]["attributes"]["ram_gb"] == 32
    assert by_category["prebuilt_pc"]["attributes"]["storage_gb"] == 3072

    with tempfile.TemporaryDirectory(prefix="dns_snapshot_check_") as temp_dir:
        root = Path(temp_dir)
        catalog_path = save_catalog(catalog, root / "catalog.json")
        records = CatalogStagingService(root / "staging.json").stage_file(catalog_path)
        assert len(records) == 2
        assert all(not record["validation_errors"] for record in records)

    print("DNS snapshot import check passed: 2 JSON-LD products, catalog v2 and staging.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
