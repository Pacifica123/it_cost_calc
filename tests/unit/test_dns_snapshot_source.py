import json
from pathlib import Path

import pytest

from tools.catalog_parser.sources.dns_snapshot import (
    build_catalog_from_dns_snapshot,
    parse_dns_product_html,
)


def test_parse_json_ld_product_and_offer() -> None:
    item, warnings = parse_dns_product_html(
        """
        <script type="application/ld+json">
        {"@graph":[{"@type":"Product","name":"Router X","brand":{"name":"Brand"},
        "model":"X1","gtin13":"123","offers":{"price":"4999","priceCurrency":"RUB",
        "availability":"https://schema.org/InStock"},"additionalProperty":[
        {"name":"Скорость LAN","value":"1 Гбит/с"}]}]}
        </script>
        """
    )
    assert warnings == []
    assert item["title"] == "Router X"
    assert item["price_int"] == 4999
    assert item["availability"] == "in_stock"
    assert item["specs"]["Скорость LAN"] == "1 Гбит/с"


def test_invalid_json_ld_uses_html_fallback() -> None:
    item, warnings = parse_dns_product_html(
        '<html><head><meta itemprop="price" content="1200"></head>'
        '<body><script type="application/ld+json">{bad}</script><h1>Fallback</h1>'
        '<dl><dt>Объем SSD</dt><dd>512 ГБ</dd></dl></body></html>'
    )
    assert item["title"] == "Fallback"
    assert item["price_int"] == 1200
    assert item["specs"] == {"Объем SSD": "512 ГБ"}
    assert any("invalid" in warning for warning in warnings)
    assert any("fallback" in warning for warning in warnings)


def test_build_snapshot_rejects_parent_path(tmp_path: Path) -> None:
    (tmp_path / "snapshot_manifest.json").write_text(
        json.dumps(
            {"schema_version": 1, "items": [{"file": "../outside.html", "category": "routers"}]}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsafe snapshot file path"):
        build_catalog_from_dns_snapshot(tmp_path)
