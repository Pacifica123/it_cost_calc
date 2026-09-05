import base64
import json
from pathlib import Path

from tools.catalog_parser.cli import build_parser
from tools.catalog_parser.market_browser import YandexMarketBrowserPage
from tools.catalog_parser.sources.yandex_market_capture import (
    build_catalog_from_yandex_market_har,
    build_catalog_from_yandex_market_html,
)
from tools.catalog_parser.sources.yandex_market_live import (
    YANDEX_MARKET_CATEGORY_URLS,
    YandexMarketAccessDeniedError,
    YandexMarketLiveOptions,
    capture_yandex_market_snapshot,
    parse_yandex_market_listing_html,
)
from tools.catalog_parser.sources.yandex_market_snapshot import (
    build_catalog_from_yandex_market_snapshot,
    parse_yandex_market_product_html,
)

PRODUCT_URL = "https://market.yandex.ru/card/test-router/123456789"
LISTING_HTML = f"""
<html><body>
  <a href="/card/test-router/123456789?cpc=secret"><span>Router One</span></a>
  <a href="{PRODUCT_URL}?show-uid=duplicate">Router One duplicate</a>
  <a href="https://evil.test/card/external/999">External</a>
  <a href="https://market.yandex.ru.evil.test/card/lookalike/888">Lookalike</a>
</body></html>
"""
PRODUCT_HTML = f"""
<html><head>
  <link rel="canonical" href="{PRODUCT_URL}">
  <script type="application/ld+json">
  {{
    "@context":"https://schema.org",
    "@type":"Product",
    "name":"Роутер Test R1",
    "sku":"123456789",
    "brand":{{"name":"Test"}},
    "offers":{{
      "price":"7299",
      "priceCurrency":"RUB",
      "availability":"https://schema.org/InStock"
    }},
    "additionalProperty":[
      {{"name":"Количество LAN-портов","value":"4"}},
      {{"name":"Скорость Ethernet","value":"1 Гбит/с"}},
      {{"name":"Мощность блока питания","value":"18 Вт"}}
    ]
  }}
  </script>
</head><body><h1>Роутер Test R1</h1></body></html>
"""
CAPTCHA_HTML = "<html><body><div id='smartcaptcha'>Подтвердите, что запросы отправляли вы</div></body></html>"


def _har_entry(url: str, body: str, *, encoding: str | None = None) -> dict:
    text = base64.b64encode(body.encode()).decode() if encoding == "base64" else body
    return {
        "startedDateTime": "2026-08-22T08:00:00Z",
        "request": {
            "url": url,
            "headers": [{"name": "Cookie", "value": "private-cookie"}],
            "cookies": [{"name": "Session_id", "value": "private-cookie"}],
            "postData": {"text": "private-post-data"},
        },
        "response": {
            "status": 200,
            "headers": [{"name": "Set-Cookie", "value": "private-response"}],
            "content": {"mimeType": "text/html", "text": text, "encoding": encoding},
        },
    }


def test_listing_parser_keeps_unique_public_market_card_urls() -> None:
    items = parse_yandex_market_listing_html(LISTING_HTML, limit=10)
    assert items == [{"url": PRODUCT_URL, "title": "Router One"}]


def test_product_parser_reads_json_ld_and_specs() -> None:
    item, warnings = parse_yandex_market_product_html(PRODUCT_HTML)
    assert item["title"] == "Роутер Test R1"
    assert item["price_int"] == 7299
    assert item["source_product_id"] == "123456789"
    assert item["specs"]["Количество LAN-портов"] == "4"
    assert warnings == []


def test_live_capture_creates_replayable_market_snapshot(tmp_path: Path) -> None:
    options = YandexMarketLiveOptions(
        snapshot_dir=tmp_path / "snapshot",
        profile_dir=tmp_path / "profile",
        categories=("routers",),
        per_category_limit=2,
        time_limit_seconds=60,
        request_delay_seconds=0,
        region="Москва",
    )

    requested: list[str] = []

    def fetch(url: str) -> str:
        requested.append(url)
        return LISTING_HTML if "/category/" in url else PRODUCT_HTML

    manifest_path = capture_yandex_market_snapshot(
        options,
        fetch=fetch,
        progress=lambda _message: None,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["source"] == "yandex_market"
    assert manifest["capture"]["status"] == "completed"
    assert requested[0] == YANDEX_MARKET_CATEGORY_URLS["routers"]
    assert len(manifest["items"]) == 1

    catalog = build_catalog_from_yandex_market_snapshot(options.snapshot_dir)
    assert catalog["stats"]["items_total"] == 1
    item = catalog["items"][0]
    assert item["source"] == "yandex_market"
    assert item["category"] == "router"
    assert item["attributes"]["lan_ports"] == 4
    assert item["attributes"]["lan_speed_mbps"] == 1000
    assert item["offer"]["region"] == "Москва"


def test_snapshot_skips_generic_market_shell_instead_of_creating_placeholder(
    tmp_path: Path,
) -> None:
    snapshot_dir = tmp_path / "snapshot"
    products_dir = snapshot_dir / "products"
    products_dir.mkdir(parents=True)
    shell_file = products_dir / "routers_001.html"
    shell_file.write_text(
        "<html><head><title>Яндекс Маркет</title></head><body>Каталог</body></html>",
        encoding="utf-8",
    )
    (snapshot_dir / "snapshot_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": "yandex_market",
                "observed_at": "2026-09-05T00:00:00Z",
                "items": [
                    {
                        "file": "products/routers_001.html",
                        "category": "routers",
                        "url": PRODUCT_URL,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    catalog = build_catalog_from_yandex_market_snapshot(snapshot_dir)

    assert catalog["stats"]["items_total"] == 0
    assert catalog["items"] == []
    assert any(
        "skipped low-quality product page" in warning
        for warning in catalog["sources"][0]["warnings"]
    )


def test_live_capture_classifies_market_captcha(tmp_path: Path) -> None:
    options = YandexMarketLiveOptions(
        snapshot_dir=tmp_path / "snapshot",
        profile_dir=tmp_path / "profile",
        categories=("routers",),
        per_category_limit=1,
        time_limit_seconds=60,
        request_delay_seconds=0,
    )

    def fetch(url: str) -> YandexMarketBrowserPage:
        return YandexMarketBrowserPage(
            requested_url=url,
            final_url="https://market.yandex.ru/showcaptcha",
            status_code=200,
            title="Ой!",
            html=CAPTCHA_HTML,
        )

    try:
        capture_yandex_market_snapshot(options, fetch=fetch, progress=lambda _message: None)
    except YandexMarketAccessDeniedError as exc:
        assert exc.exit_code == 3
    else:
        raise AssertionError("Yandex Market CAPTCHA must stop collection")
    manifest = json.loads(
        (options.snapshot_dir / "snapshot_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["capture"]["failure"]["stage"] == "challenge_not_completed"


def test_market_html_and_har_import_are_offline_and_drop_secrets(tmp_path: Path) -> None:
    html_path = tmp_path / "card.html"
    html_path.write_text(PRODUCT_HTML, encoding="utf-8")
    html_payload = build_catalog_from_yandex_market_html(html_path, region="Кемерово")
    assert html_payload["items"][0]["price_rub"] == 7299
    assert html_payload["items"][0]["offer"]["region"] == "Кемерово"

    har = {
        "log": {
            "entries": [
                _har_entry(YANDEX_MARKET_CATEGORY_URLS["routers"], LISTING_HTML),
                _har_entry(PRODUCT_URL, PRODUCT_HTML, encoding="base64"),
                _har_entry("https://example.test/card/ignored/1", "private-external-body"),
            ]
        }
    }
    har_path = tmp_path / "capture.har"
    har_path.write_text(json.dumps(har), encoding="utf-8")
    har_payload = build_catalog_from_yandex_market_har(har_path, region="Кемерово")
    assert har_payload["stats"]["items_total"] == 1
    assert har_payload["items"][0]["source"] == "yandex_market"
    serialized = json.dumps(har_payload, ensure_ascii=False)
    assert "private-cookie" not in serialized
    assert "private-post-data" not in serialized
    assert "private-external-body" not in serialized


def test_cli_exposes_all_yandex_market_modes() -> None:
    parser = build_parser()
    assert parser.parse_args(["--mode", "yandex-market-snapshot", "--input", "s"]).mode == (
        "yandex-market-snapshot"
    )
    assert parser.parse_args(["--mode", "yandex-market-har", "--input", "x.har"]).mode == (
        "yandex-market-har"
    )
    assert parser.parse_args(["--mode", "yandex-market-html", "--input", "x.html"]).mode == (
        "yandex-market-html"
    )
    args = parser.parse_args(
        [
            "--mode",
            "yandex-market-live",
            "--categories",
            "routers,servers",
            "--snapshot-output",
            "snapshot",
            "--profile",
            "profile",
        ]
    )
    assert args.mode == "yandex-market-live"
    assert args.limit == 10
