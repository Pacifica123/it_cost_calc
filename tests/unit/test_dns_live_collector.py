import io
import json
from contextlib import redirect_stdout
from pathlib import Path

from tools.catalog_parser.dns_browser import DnsBrowserPage
from tools.catalog_parser.sources.dns_live import (
    DNS_BASE_URL,
    DNS_CATALOG_URLS,
    DnsAccessDeniedError,
    DnsLiveOptions,
    capture_dns_snapshot,
    parse_dns_search_html,
)
from tools.catalog_parser.sources.dns_snapshot import build_catalog_from_dns_snapshot
from tools.catalog_parser.cli import build_parser
from tools.catalog_parser import cli
from tools.catalog_parser.sources import dns_live


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

DNS_403_HTML = """
<!DOCTYPE html><html lang="ru"><head>
<meta charset="utf-8"><title>HTTP 403</title></head><body>
<div class="title">403 Error</div><div class="sub-title">Forbidden</div>
<p>Доступ к сайту www.dns-shop.ru запрещен.</p>
</body></html>
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

    requested_urls: list[str] = []

    def fetch(url: str) -> str:
        requested_urls.append(url)
        return SEARCH_HTML if "/catalog/" in url or "/search/" in url else PRODUCT_HTML

    manifest_path = capture_dns_snapshot(options, fetch=fetch, progress=lambda _message: None)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["capture"]["status"] == "completed"
    assert manifest["capture"]["categories"] == ["routers"]
    assert manifest["capture"]["browser_engine"] == "firefox"
    assert manifest["capture"]["category_sources"]["routers"]["mode"] == "catalog-url"
    assert requested_urls[0] == DNS_CATALOG_URLS["routers"]
    assert manifest["region"] == "test-region"
    assert len(manifest["items"]) == 1
    assert (options.snapshot_dir / manifest["items"][0]["file"]).exists()

    catalog = build_catalog_from_dns_snapshot(options.snapshot_dir)
    assert catalog["stats"]["items_total"] == 1
    item = catalog["items"][0]
    assert item["category"] == "router"
    assert item["attributes"]["lan_ports"] == 4
    assert item["offer"]["region"] == "test-region"


def test_capture_stops_after_first_dns_403_and_writes_failure_manifest(tmp_path: Path) -> None:
    options = DnsLiveOptions(
        snapshot_dir=tmp_path / "snapshot",
        profile_dir=tmp_path / "profile",
        categories=("routers", "prebuilt_pcs", "servers"),
        per_category_limit=3,
        time_limit_seconds=60,
        request_delay_seconds=0,
    )
    requested_urls: list[str] = []

    def fetch(url: str) -> str:
        requested_urls.append(url)
        return DNS_403_HTML

    try:
        capture_dns_snapshot(options, fetch=fetch, progress=lambda _message: None)
    except DnsAccessDeniedError as exc:
        assert exc.exit_code == 3
        assert exc.manifest_path == options.snapshot_dir / "snapshot_manifest.json"
    else:
        raise AssertionError("DNS 403 must stop collection")

    assert len(requested_urls) == 1
    manifest = json.loads(
        (options.snapshot_dir / "snapshot_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["capture"]["status"] == "failed"
    assert manifest["capture"]["failure"]["kind"] == "access_denied"
    assert manifest["capture"]["failure"]["status_code"] == 403
    assert manifest["capture"]["failure"]["stage"] == "hard_block"
    assert "окончательную страницу HTTP 403" in manifest["capture"]["failure"]["message"]
    assert (options.snapshot_dir / "search" / "routers.html").exists()
    assert not (options.snapshot_dir / "search" / "prebuilt_pcs.html").exists()


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
    assert args.browser_engine == "firefox"


def test_capture_keeps_partial_result_when_later_category_is_denied(tmp_path: Path) -> None:
    options = DnsLiveOptions(
        snapshot_dir=tmp_path / "snapshot",
        profile_dir=tmp_path / "profile",
        categories=("routers", "prebuilt_pcs"),
        per_category_limit=1,
        time_limit_seconds=60,
        request_delay_seconds=0,
    )

    def fetch(url: str) -> str:
        if url == DNS_CATALOG_URLS["routers"]:
            return SEARCH_HTML
        if "/product/" in url:
            return PRODUCT_HTML
        return DNS_403_HTML

    manifest_path = capture_dns_snapshot(options, fetch=fetch, progress=lambda _message: None)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["capture"]["status"] == "partial"
    assert manifest["capture"]["failure"]["status_code"] == 403
    assert len(manifest["items"]) == 1


def test_html_403_takes_precedence_over_transport_401(tmp_path: Path) -> None:
    options = DnsLiveOptions(
        snapshot_dir=tmp_path / "snapshot",
        profile_dir=tmp_path / "profile",
        categories=("routers",),
        per_category_limit=1,
        time_limit_seconds=60,
        request_delay_seconds=0,
    )

    def fetch(url: str) -> DnsBrowserPage:
        return DnsBrowserPage(
            requested_url=url,
            final_url=url,
            status_code=401,
            title="HTTP 403",
            html=DNS_403_HTML.replace("</body>", "<p>Guru meditation</p></body>"),
        )

    try:
        capture_dns_snapshot(options, fetch=fetch, progress=lambda _message: None)
    except DnsAccessDeniedError:
        pass
    else:
        raise AssertionError("DNS hard block must stop collection")

    manifest = json.loads(
        (options.snapshot_dir / "snapshot_manifest.json").read_text(encoding="utf-8")
    )
    failure = manifest["capture"]["failure"]
    assert failure["status_code"] == 403
    assert failure["response_status_code"] == 401
    assert failure["html_status_code"] == 403
    assert failure["stage"] == "hard_block"
    assert failure["protection_provider"] == "qrator"


def test_cli_reports_expected_access_denial_without_traceback(tmp_path: Path) -> None:
    manifest_path = tmp_path / "snapshot" / "snapshot_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("{}", encoding="utf-8")
    original = dns_live.build_catalog_from_live_dns

    def denied(_options, *, progress):
        raise DnsAccessDeniedError("DNS вернул HTTP 403", manifest_path=manifest_path)

    dns_live.build_catalog_from_live_dns = denied
    output = io.StringIO()
    try:
        with redirect_stdout(output):
            exit_code = cli.main(
                [
                    "--mode",
                    "dns-live",
                    "--snapshot-output",
                    str(tmp_path / "snapshot"),
                    "--profile",
                    str(tmp_path / "profile"),
                    "--output",
                    str(tmp_path / "catalog.json"),
                ]
            )
    finally:
        dns_live.build_catalog_from_live_dns = original

    assert exit_code == 3
    assert "Ошибка DNS-сбора: DNS вернул HTTP 403" in output.getvalue()
    assert str(manifest_path) in output.getvalue()
    assert "Traceback" not in output.getvalue()


def test_live_catalog_warms_up_dns_homepage_before_capture(tmp_path: Path) -> None:
    options = DnsLiveOptions(
        snapshot_dir=tmp_path / "snapshot",
        profile_dir=tmp_path / "profile",
        categories=("routers",),
        per_category_limit=1,
        time_limit_seconds=60,
        request_delay_seconds=0,
    )
    events: list[str] = []

    class FakeBrowser:
        def __init__(self, **_kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def fetch(self, url: str) -> str:
            events.append(url)
            if url == DNS_BASE_URL + "/":
                return "<html><body>home</body></html>"
            if "/catalog/" in url:
                return SEARCH_HTML
            return PRODUCT_HTML

        def __exit__(self, _exc_type, _exc, _traceback) -> None:
            pass

    original_session = dns_live.DnsBrowserSession
    dns_live.DnsBrowserSession = FakeBrowser
    try:
        payload = dns_live.build_catalog_from_live_dns(options, progress=lambda _message: None)
    finally:
        dns_live.DnsBrowserSession = original_session

    assert events[0] == DNS_BASE_URL + "/"
    assert events[1] == DNS_CATALOG_URLS["routers"]
    assert payload["stats"]["items_total"] == 1
