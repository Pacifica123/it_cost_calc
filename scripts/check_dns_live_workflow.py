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

from application.services.catalog_staging_service import CatalogStagingService  # noqa: E402
from tools.catalog_parser.catalog_builder import save_catalog  # noqa: E402
from tools.catalog_parser.sources.dns_live import (  # noqa: E402
    DnsLiveOptions,
    capture_dns_snapshot,
)
from tools.catalog_parser.sources.dns_snapshot import build_catalog_from_dns_snapshot  # noqa: E402

SEARCH_HTML = '<a href="/product/check-router/">Check Router</a>'
PRODUCT_HTML = """
<script type="application/ld+json">
{"@type":"Product","name":"Check Router [4 LAN, 1000 Мбит/с, Wi-Fi 1200 Мбит/с, IPv6]",
"brand":"Check","model":"R1","offers":{"price":4999,"priceCurrency":"RUB",
"availability":"https://schema.org/InStock"},"additionalProperty":[
{"name":"Количество LAN портов","value":"4"},{"name":"Скорость LAN","value":"1 Гбит/с"},
{"name":"Максимальная скорость Wi-Fi","value":"1200 Мбит/с"},
{"name":"Поддержка IPv6","value":"Да"},
{"name":"Максимальная потребляемая мощность","value":"12 Вт"}]}
</script>
"""


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="dns_live_workflow_") as temp_dir:
        root = Path(temp_dir)
        options = DnsLiveOptions(
            snapshot_dir=root / "snapshot",
            profile_dir=root / "profile",
            categories=("routers",),
            per_category_limit=1,
            time_limit_seconds=60,
            request_delay_seconds=0,
            region="smoke-region",
        )
        capture_dns_snapshot(
            options,
            fetch=lambda url: SEARCH_HTML if "/search/" in url else PRODUCT_HTML,
            progress=lambda _message: None,
        )
        catalog = build_catalog_from_dns_snapshot(options.snapshot_dir)
        catalog_path = save_catalog(catalog, root / "equipment_catalog.json")
        records = CatalogStagingService(root / "staging.json").stage_file(catalog_path)
        manifest = json.loads(
            (options.snapshot_dir / "snapshot_manifest.json").read_text(encoding="utf-8")
        )
        assert len(manifest["items"]) == 1
        assert catalog["schema_version"] == 2
        assert catalog["items"][0]["attributes"]["lan_ports"] == 4
        assert len(records) == 1 and not records[0]["validation_errors"]

    print("DNS live workflow check passed: capture, replay, catalog v2 and staging.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
