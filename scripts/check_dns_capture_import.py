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

from tools.catalog_parser.sources.dns_capture import build_catalog_from_dns_har  # noqa: E402


def main() -> int:
    key = "4a01ec7c-088b-11f1-93fb-0050569d9cb8"
    category_html = (
        f'<div class="catalog-product" data-product="{key}">'
        '<a class="catalog-product__name" href="/product/4a01ec7c088bd9cb/test/" '
        'title="ПК Capture Test [16 ГБ DDR5]">ПК Capture Test</a></div>'
    )
    prices = {
        "data": {
            "states": [
                {"data": {"id": key, "name": "ПК Capture Test", "price": {"current": 50000}}}
            ]
        }
    }

    def entry(url: str, text: str, mime: str) -> dict:
        return {
            "startedDateTime": "2026-06-22T07:00:00Z",
            "request": {"url": url, "headers": [{"name": "Cookie", "value": "ignored"}]},
            "response": {"status": 200, "content": {"mimeType": mime, "text": text}},
        }

    har = {
        "log": {
            "entries": [
                entry(
                    "https://www.dns-shop.ru/catalog/17a8932c16404e77/personalnye-komputery/",
                    category_html,
                    "text/html",
                ),
                entry(
                    "https://www.dns-shop.ru/ajax-state/product-buy/",
                    json.dumps(prices),
                    "application/json",
                ),
            ]
        }
    }
    with tempfile.TemporaryDirectory(prefix="dns_capture_check_") as directory:
        path = Path(directory) / "capture.har"
        path.write_text(json.dumps(har), encoding="utf-8")
        payload = build_catalog_from_dns_har(path, region="smoke-region")
    assert payload["stats"]["items_total"] == 1
    assert payload["items"][0]["price_rub"] == 50000
    assert payload["items"][0]["offer"]["region"] == "smoke-region"
    assert "ignored" not in json.dumps(payload)
    print("DNS capture import check passed: local HAR, price merge and sensitive headers ignored.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
