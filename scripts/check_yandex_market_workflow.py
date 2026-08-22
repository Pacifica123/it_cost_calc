#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for value in (str(ROOT), str(SRC)):
    if value not in sys.path:
        sys.path.insert(0, value)

from tools.catalog_parser.sources.yandex_market_live import (  # noqa: E402
    YandexMarketLiveOptions,
    capture_yandex_market_snapshot,
)
from tools.catalog_parser.sources.yandex_market_snapshot import (  # noqa: E402
    build_catalog_from_yandex_market_snapshot,
)

PRODUCT_URL = "https://market.yandex.ru/card/check-router/101010"
LISTING_HTML = f'<a href="{PRODUCT_URL}?tracking=ignored">Check Router</a>'
PRODUCT_HTML = f"""
<html><head><link rel="canonical" href="{PRODUCT_URL}">
<script type="application/ld+json">{{
  "@context":"https://schema.org", "@type":"Product", "name":"Роутер Check R1",
  "sku":"101010", "offers":{{"price":5999,"priceCurrency":"RUB"}},
  "additionalProperty":[{{"name":"Количество LAN-портов","value":"4"}}]
}}</script></head><body><h1>Роутер Check R1</h1></body></html>
"""


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="yandex_market_check_") as directory:
        root = Path(directory)
        options = YandexMarketLiveOptions(
            snapshot_dir=root / "snapshot",
            profile_dir=root / "profile",
            categories=("routers",),
            per_category_limit=1,
            time_limit_seconds=60,
            request_delay_seconds=0,
            region="smoke-region",
        )

        def fetch(url: str) -> str:
            return LISTING_HTML if "/category/" in url else PRODUCT_HTML

        manifest_path = capture_yandex_market_snapshot(
            options,
            fetch=fetch,
            progress=lambda _message: None,
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        catalog = build_catalog_from_yandex_market_snapshot(options.snapshot_dir)
        item = catalog["items"][0]
        assert manifest["source"] == "yandex_market"
        assert catalog["stats"]["items_total"] == 1
        assert item["source"] == "yandex_market"
        assert item["attributes"]["lan_ports"] == 4
        assert item["offer"]["region"] == "smoke-region"
    print(
        "Yandex Market workflow check passed: bounded capture, replay, catalog v2 and provenance."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
