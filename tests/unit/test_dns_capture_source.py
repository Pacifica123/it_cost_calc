import base64
import json
from pathlib import Path

from tools.catalog_parser.cli import build_parser
from tools.catalog_parser.sources.dns_capture import (
    build_catalog_from_dns_har,
    build_catalog_from_dns_html,
)


PRODUCT_KEY = "4a01ec7c-088b-11f1-93fb-0050569d9cb8"
PRODUCT_URL = "https://www.dns-shop.ru/product/4a01ec7c088bd9cb/pk-test/"
PRODUCT_HTML = f"""
<html><head><link rel="canonical" href="{PRODUCT_URL}"></head><body>
<div class="product-card" data-product-card="{PRODUCT_KEY}">
  <h1 class="product-card-top__title" data-product-title>ПК Test One</h1>
  <div class="product-card-top__specs-item">
    <div class="product-card-top__specs-item-title">Оперативная память</div>
    <div class="product-card-top__specs-item-content">16 ГБ DDR5</div>
  </div>
  <div class="product-card-top__code">Код товара: 5663770</div>
</div></body></html>
"""
CATEGORY_HTML = f"""
<div class="catalog-product" data-product="{PRODUCT_KEY}">
  <a class="catalog-product__name" href="/product/4a01ec7c088bd9cb/pk-test/"
     title="ПК Test One [16 ГБ DDR5, SSD 1 ТБ]">ПК Test One</a>
  <div class="catalog-product__spec">
    <div class="catalog-product__spec-label">SSD</div>
    <div class="catalog-product__spec-value">1 ТБ</div>
  </div>
</div>
"""


def _entry(url: str, body: str, mime: str, *, encoding: str | None = None) -> dict:
    text = base64.b64encode(body.encode()).decode() if encoding == "base64" else body
    return {
        "startedDateTime": "2026-06-22T07:00:00Z",
        "request": {
            "url": url,
            "headers": [{"name": "Cookie", "value": "secret-cookie"}],
            "cookies": [{"name": "secret", "value": "secret-cookie"}],
            "postData": {"text": "secret-post-data"},
        },
        "response": {
            "status": 200,
            "headers": [{"name": "Set-Cookie", "value": "secret-response"}],
            "content": {"mimeType": mime, "text": text, "encoding": encoding},
        },
    }


def test_har_import_merges_category_price_microdata_and_pwa(tmp_path: Path) -> None:
    microdata = {
        "data": {
            "@type": "Product",
            "name": "ПК Test One",
            "sku": "5663770",
            "brand": {"name": "Test Brand"},
            "offers": {
                "url": PRODUCT_URL,
                "price": 84299,
                "priceCurrency": "RUB",
                "availability": "https://schema.org/InStock",
            },
        }
    }
    pwa = {
        "data": {
            "guid": PRODUCT_KEY,
            "code": "5663770",
            "name": "ПК Test One",
            "price": 84299,
            "characteristics": {
                "Процессор": [{"title": "Общее количество ядер", "value": "6"}],
                "Накопители": [
                    {
                        "title": "Конфигурация твердотельных накопителей (SSD)",
                        "value": "1 ТБ",
                    }
                ],
            },
            "topOpinion": {"userName": "must-not-be-copied"},
        }
    }
    product_buy = {
        "data": {
            "states": [
                {
                    "data": {
                        "id": PRODUCT_KEY,
                        "name": "ПК Test One",
                        "price": {"current": 84299},
                    }
                }
            ]
        }
    }
    har = {
        "log": {
            "entries": [
                _entry(
                    "https://www.dns-shop.ru/catalog/17a8932c16404e77/personalnye-komputery/",
                    CATEGORY_HTML,
                    "text/html",
                ),
                _entry(PRODUCT_URL, PRODUCT_HTML, "text/html"),
                _entry(
                    f"https://www.dns-shop.ru/product/microdata/{PRODUCT_KEY}/",
                    json.dumps(microdata),
                    "application/json",
                    encoding="base64",
                ),
                _entry(
                    "https://www.dns-shop.ru/ajax-state/product-buy/",
                    json.dumps(product_buy),
                    "application/json",
                ),
                _entry(
                    "https://www.dns-shop.ru/pwa/pwa/get-product/?id=5663770",
                    json.dumps(pwa),
                    "application/json",
                ),
                _entry(
                    "https://example.test/product/ignored/",
                    "secret-external-body",
                    "text/html",
                ),
            ]
        }
    }
    path = tmp_path / "capture.har"
    path.write_text(json.dumps(har), encoding="utf-8")

    payload = build_catalog_from_dns_har(path, region="Кемерово")

    assert payload["stats"]["items_total"] == 1
    item = payload["items"][0]
    assert item["title"] == "ПК Test One"
    assert item["price_rub"] == 84299
    assert item["source_product_id"] == "5663770"
    assert item["offer"]["region"] == "Кемерово"
    assert item["attributes"]["cpu_cores"] == 6
    assert item["attributes"]["ram_gb"] == 16
    assert item["attributes"]["storage_gb"] == 1024
    serialized = json.dumps(payload, ensure_ascii=False)
    assert "secret-cookie" not in serialized
    assert "secret-post-data" not in serialized
    assert "must-not-be-copied" not in serialized


def test_saved_product_html_builds_partial_catalog(tmp_path: Path) -> None:
    path = tmp_path / "product.htm"
    path.write_text(PRODUCT_HTML, encoding="utf-8")

    payload = build_catalog_from_dns_html(path, region="Москва")

    item = payload["items"][0]
    assert item["title"] == "ПК Test One"
    assert item["source_product_id"] == "5663770"
    assert item["price_rub"] is None
    assert item["attributes"]["ram_gb"] == 16
    assert item["offer"]["region"] == "Москва"
    assert "Цена отсутствует" in " ".join(item["review"]["warnings"])


def test_cli_exposes_offline_dns_capture_modes() -> None:
    parser = build_parser()
    assert parser.parse_args(["--mode", "dns-har", "--input", "capture.har"]).mode == "dns-har"
    assert parser.parse_args(["--mode", "dns-html", "--input", "product.htm"]).mode == "dns-html"
