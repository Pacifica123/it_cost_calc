import json
from pathlib import Path

from tools.catalog_parser.sources.dns_live import (
    DnsLiveOptions,
    capture_dns_snapshot,
    parse_dns_search_html,
)
from tools.catalog_parser.sources.dns_snapshot import build_catalog_from_dns_snapshot
from tools.catalog_parser.cli import build_parser


SEARCH_HTML = """
<html><body>
  <a href="/product/router-one/">Router One</a>
  <a href="/product/router-one/"><span>Router One duplicate</span></a>
  <a href="https://example.test/product/external/">External</a>
  <a href="https://fake-dns-shop.ru/product/external/">Lookalike</a>
</body></html>
"""

PRODUCT_HTML = """
<html><head><script type="application/ld+json">
{
  "@context":"https://schema.org",
  "@type":"Product",
  "name":"Router One [4 LAN, 1000 Мбит/с, Wi-Fi 1200 Мбит/с, IPv6]",
  "brand":"Example",
  "model":"R1",
  "offers":{"price":"4999","priceCurrency":"RUB","availability":"https://schema.org/InStock"},
  "additionalProperty":[
    {"name":"Количество LAN портов","value":"4"},
    {"name":"Скорость LAN","value":"1 Гбит/с"},
    {"name":"Максимальная скорость Wi-Fi","value":"1200 Мбит/с"},
    {"name":"Поддержка IPv6","value":"Да"},
    {"name":"Максимальная потребляемая мощность","value":"12 Вт"}
  ]
}
</script></head><body></body></html>
"""


def test_search_parser_keeps_unique_dns_product_urls() -> None:
    items = parse_dns_search_html(SEARCH_HTML, limit=10)
    assert items == [
        {"url": "https://www.dns-shop.ru/product/router-one/", "title": "Router One"}
    ]


def test_capture_creates_replayable_snapshot_and_catalog(tmp_path: Path) -> None:
    options = DnsLiveOptions(
        snapshot_dir=tmp_path / "snapshot",
        profile_dir=tmp_path / "profile",
        categories=("routers",),
        per_category_limit=2,
        time_limit_seconds=60,
        request_delay_seconds=0,
        region="test-region",
    )

    def fetch(url: str) -> str:
        return SEARCH_HTML if "/search/" in url else PRODUCT_HTML

    manifest_path = capture_dns_snapshot(options, fetch=fetch, progress=lambda _message: None)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["capture"]["categories"] == ["routers"]
    assert manifest["region"] == "test-region"
    assert len(manifest["items"]) == 1
    assert (options.snapshot_dir / manifest["items"][0]["file"]).exists()

    catalog = build_catalog_from_dns_snapshot(options.snapshot_dir)
    assert catalog["stats"]["items_total"] == 1
    item = catalog["items"][0]
    assert item["category"] == "router"
    assert item["attributes"]["lan_ports"] == 4
    assert item["offer"]["region"] == "test-region"


def test_cli_exposes_bounded_dns_live_options() -> None:
    args = build_parser().parse_args(
        [
            "--mode",
            "dns-live",
            "--categories",
            "routers,servers",
            "--limit",
            "5",
            "--time-limit",
            "120",
            "--snapshot-output",
            "snapshot",
            "--profile",
            "profile",
        ]
    )
    assert args.mode == "dns-live"
    assert args.limit == 5
    assert args.time_limit == 120
