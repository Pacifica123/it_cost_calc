import json
from pathlib import Path

from tools.catalog_parser.http_session import CatalogHttpResponse
from tools.catalog_parser.sources.dns_http_live import (
    DnsHttpLiveOptions,
    build_catalog_from_http_dns,
    parse_dns_product_buy_batches,
)

_UUID = "11111111-2222-3333-4444-555555555555"
_HASH = "a" * 40


def _catalog_payload() -> str:
    batch = [
        {"type": "product-buy", "hash": _HASH, "timeout": 10},
        [
            {
                "id": "as-AbCd12",
                "data": {"id": _UUID, "type": 4, "params": {"hideButtons": True}},
            }
        ],
    ]
    return json.dumps(
        {
            "result": True,
            "assets": {
                "inlineJs": {
                    "nonce": "window.AjaxState.register(" + json.dumps([batch]) + ");"
                }
            },
        }
    )


def test_dns_http_parser_preserves_hash_and_original_containers() -> None:
    batches = parse_dns_product_buy_batches(_catalog_payload())

    assert len(batches) == 1
    product_hash, containers = batches[0]
    assert product_hash == _HASH
    assert containers[0]["id"] == "as-AbCd12"
    assert containers[0]["data"]["id"] == _UUID


class _FakeDnsSession:
    def __init__(self, **_kwargs) -> None:
        self.posts: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        return None

    def get(self, url: str, **_kwargs) -> CatalogHttpResponse:
        if url.endswith("/catalog/markdown/"):
            body = "{}"
        elif "wi-fi-routery" in url:
            body = _catalog_payload()
        else:
            body = "<html><title>DNS</title></html>"
        return CatalogHttpResponse(url, url, 200, body, "text/html")

    def post_form(self, url: str, *, data: str, **_kwargs) -> CatalogHttpResponse:
        self.posts.append(data)
        body = json.dumps(
            {
                "result": True,
                "data": {
                    "states": [
                        {
                            "id": "as-AbCd12",
                            "data": {
                                "id": _UUID,
                                "name": "Wi-Fi роутер Test AX3000",
                                "price": {"current": 7999, "previous": 8999},
                                "brand": "Test",
                                "model": "AX3000",
                            },
                        }
                    ]
                },
            }
        )
        return CatalogHttpResponse(url, url, 200, body, "application/json")


def test_dns_http_live_builds_catalog_without_playwright(tmp_path: Path) -> None:
    payload = build_catalog_from_http_dns(
        DnsHttpLiveOptions(
            snapshot_dir=tmp_path / "snapshot",
            categories=("routers",),
            per_category_limit=1,
            time_limit_seconds=60,
            region="Кемерово",
            request_delay_seconds=0,
        ),
        session_factory=_FakeDnsSession,
        sleep=lambda _seconds: None,
    )

    assert payload["stats"]["items_total"] == 1
    item = payload["items"][0]
    assert item["source"] == "dns"
    assert item["source_product_id"] == _UUID
    assert item["price_rub"] == 7999
    assert item["offer"]["region"] == "Кемерово"
    manifest = json.loads(
        (tmp_path / "snapshot" / "snapshot_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["capture"]["mode"] == "user-initiated-http-json"
    assert manifest["capture"]["status"] == "completed"


def test_parse_dns_batches_handles_nested_arrays_and_strings() -> None:
    raw = json.dumps(
        {
            "assets": {
                "inlineJs": {
                    "nonce": (
                        'before(); AjaxState.register('
                        + json.dumps(
                            [[
                                {"type": "product-buy", "hash": "b" * 40, "timeout": 10},
                                [{
                                    "id": "as-nested",
                                    "data": {
                                        "id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                                        "type": 4,
                                        "params": {"tags": ["one]", "two"]},
                                    },
                                }],
                            ]]
                        )
                        + '); after();'
                    )
                }
            }
        }
    )
    batches = parse_dns_product_buy_batches(raw)
    assert len(batches) == 1
    assert batches[0][0] == "b" * 40
    assert batches[0][1][0]["id"] == "as-nested"


def test_dns_access_detection_does_not_reject_incidental_qrator_text() -> None:
    from tools.catalog_parser.http_session import CatalogHttpResponse
    from tools.catalog_parser.sources.dns_http_live import _access_failure

    response = CatalogHttpResponse(
        requested_url="https://www.dns-shop.ru/catalog/example/",
        final_url="https://www.dns-shop.ru/catalog/example/",
        status_code=200,
        text='{"note":"regular page may mention Qrator or captcha documentation"}',
        content_type="application/json",
    )
    assert _access_failure(response) is None
