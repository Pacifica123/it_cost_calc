import json
from pathlib import Path

from tools.catalog_parser.http_session import CatalogHttpResponse
from tools.catalog_parser.sources.yandex_market_http_live import (
    YandexMarketHttpLiveOptions,
    build_catalog_from_http_yandex_market,
)

_LISTING = '''
<html><body>
<a href="/card/test-router/123456789" title="Test Router">Test Router</a>
</body></html>
'''
_PRODUCT = '''
<html><head>
<link rel="canonical" href="https://market.yandex.ru/card/test-router/123456789">
<script type="application/ld+json">
{"@context":"https://schema.org","@type":"Product","name":"Test Router AX3000","sku":"123456789","brand":{"name":"Test"},"offers":{"@type":"Offer","price":"6490","priceCurrency":"RUB","availability":"https://schema.org/InStock"},"additionalProperty":[{"name":"Количество LAN-портов","value":"4"}]}
</script>
</head><body><h1>Test Router AX3000</h1></body></html>
'''


class _FakeMarketSession:
    def __init__(self, **_kwargs) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        return None

    def get(self, url: str, **_kwargs) -> CatalogHttpResponse:
        if "/category/routery" in url:
            body = _LISTING
        elif "/card/test-router/123456789" in url:
            body = _PRODUCT
        else:
            body = "<html><title>Маркет</title></html>"
        return CatalogHttpResponse(url, url, 200, body, "text/html")


def test_yandex_market_http_live_reuses_snapshot_parser(tmp_path: Path) -> None:
    payload = build_catalog_from_http_yandex_market(
        YandexMarketHttpLiveOptions(
            snapshot_dir=tmp_path / "snapshot",
            categories=("routers",),
            per_category_limit=1,
            time_limit_seconds=60,
            region="Кемерово",
            request_delay_seconds=0,
        ),
        session_factory=_FakeMarketSession,
        sleep=lambda _seconds: None,
    )

    assert payload["stats"]["items_total"] == 1
    item = payload["items"][0]
    assert item["source"] == "yandex_market"
    assert item["price_rub"] == 6490
    assert item["offer"]["region"] == "Кемерово"
    manifest = json.loads(
        (tmp_path / "snapshot" / "snapshot_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["capture"]["mode"] == "user-initiated-http"
    assert manifest["capture"]["status"] == "completed"
