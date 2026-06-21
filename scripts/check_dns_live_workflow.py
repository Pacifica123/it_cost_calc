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
from tools.catalog_parser.dns_browser import ensure_playwright_browser  # noqa: E402
from tools.catalog_parser.sources.dns_live import (  # noqa: E402
    DNS_CATALOG_URLS,
    DnsAccessDeniedError,
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
DNS_403_HTML = "<title>HTTP 403</title><h1>403 Error</h1><p>Forbidden</p>"


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
        requested_urls: list[str] = []

        def fetch(url: str) -> str:
            requested_urls.append(url)
            return SEARCH_HTML if "/catalog/" in url or "/search/" in url else PRODUCT_HTML

        capture_dns_snapshot(
            options,
            fetch=fetch,
            progress=lambda _message: None,
        )
        catalog = build_catalog_from_dns_snapshot(options.snapshot_dir)
        catalog_path = save_catalog(catalog, root / "equipment_catalog.json")
        records = CatalogStagingService(root / "staging.json").stage_file(catalog_path)
        manifest = json.loads(
            (options.snapshot_dir / "snapshot_manifest.json").read_text(encoding="utf-8")
        )
        assert len(manifest["items"]) == 1
        assert requested_urls[0] == DNS_CATALOG_URLS["routers"]
        assert catalog["schema_version"] == 2
        assert catalog["items"][0]["attributes"]["lan_ports"] == 4
        assert len(records) == 1 and not records[0]["validation_errors"]

        denied_options = DnsLiveOptions(
            snapshot_dir=root / "denied_snapshot",
            profile_dir=root / "profile",
            categories=("routers", "servers"),
            per_category_limit=1,
            time_limit_seconds=60,
            request_delay_seconds=0,
        )
        calls = 0

        def denied_fetch(_url: str) -> str:
            nonlocal calls
            calls += 1
            return DNS_403_HTML

        try:
            capture_dns_snapshot(
                denied_options,
                fetch=denied_fetch,
                progress=lambda _message: None,
            )
        except DnsAccessDeniedError:
            pass
        else:
            raise AssertionError("DNS 403 must be classified as access denied")
        denied_manifest = json.loads(
            (denied_options.snapshot_dir / "snapshot_manifest.json").read_text(encoding="utf-8")
        )
        assert calls == 1
        assert denied_manifest["capture"]["failure"]["status_code"] == 403

        browser_executable = root / "fake-chromium"
        install_commands: list[list[str]] = []

        def install_browser(command: list[str]) -> int:
            install_commands.append(command)
            browser_executable.touch()
            return 0

        ensure_playwright_browser(
            "chromium",
            progress=lambda _message: None,
            probe=lambda _engine: browser_executable,
            installer=install_browser,
        )
        assert install_commands == [
            [sys.executable, "-m", "playwright", "install", "chromium"]
        ]

    print(
        "DNS live workflow check passed: capture, replay, catalog v2, staging "
        "direct category URL, browser bootstrap and early HTTP 403 diagnostics."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
