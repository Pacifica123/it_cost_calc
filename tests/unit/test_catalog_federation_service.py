from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from application.services.catalog_federation_service import (
    federate_catalog_items,
    identity_candidates,
    offer_freshness,
    select_effective_offer,
)
from application.services.catalog_staging_service import CatalogStagingService


def _item(
    *,
    source: str,
    product_id: str,
    price: int,
    observed_at: str,
    brand: str = "MikroTik",
    mpn: str = "R1-PN",
    model: str = "Router R1",
    gtin: str = "",
    availability: str = "in_stock",
) -> dict:
    identity = {"brand": brand, "mpn": mpn, "model": model}
    if gtin:
        identity["gtin"] = gtin
    return {
        "item_id": f"{source}-{product_id}",
        "title": f"{brand} {model}",
        "category": "router",
        "source": source,
        "source_product_id": product_id,
        "identity": identity,
        "offer": {
            "price": price,
            "currency": "RUB",
            "availability": availability,
            "observed_at": observed_at,
            "price_kind": "supplier_price",
            "url": f"https://{source}.example/{product_id}",
        },
        "attributes": {"lan_ports": 4, "max_power_watts": 12},
    }


def test_identity_priority_is_gtin_then_brand_mpn_then_brand_model():
    candidates = identity_candidates(
        _item(
            source="a",
            product_id="1",
            price=100,
            observed_at="2026-09-05T08:00:00+00:00",
            gtin="4601234567890",
        )
    )

    assert [kind for kind, _key in candidates] == ["gtin", "brand_mpn", "brand_model"]


def test_federation_merges_two_sources_and_keeps_price_observations():
    items = [
        _item(
            source="supplier-a",
            product_id="a-1",
            price=12000,
            observed_at="2026-09-04T08:00:00+00:00",
        ),
        _item(
            source="supplier-b",
            product_id="b-7",
            price=11500,
            observed_at="2026-09-05T08:00:00+00:00",
        ),
    ]

    merged = federate_catalog_items(items)

    assert len(merged) == 1
    item = merged[0]
    assert item["source"] == "federated"
    assert item["federation"]["source_count"] == 2
    assert item["federation"]["matched_by"] == ["brand_mpn", "brand_model"]
    assert len(item["offers"]) == 2
    assert item["offer"]["source"] == "supplier-b"
    assert item["offer"]["price"] == 11500
    assert item["price_summary"]["min_rub"] == 11500
    assert item["price_summary"]["median_rub"] == 11750
    assert item["price_summary"]["max_rub"] == 12000


def test_effective_offer_prefers_available_before_newer_out_of_stock():
    selected = select_effective_offer(
        [
            {
                "source": "available",
                "price": 12000,
                "currency": "RUB",
                "availability": "in_stock",
                "observed_at": "2026-09-01T00:00:00+00:00",
                "price_kind": "supplier_price",
            },
            {
                "source": "newer-but-empty",
                "price": 11000,
                "currency": "RUB",
                "availability": "out_of_stock",
                "observed_at": "2026-09-05T00:00:00+00:00",
                "price_kind": "supplier_price",
            },
        ]
    )

    assert selected["source"] == "available"


def test_offer_freshness_boundaries_are_explicit():
    now = datetime(2026, 9, 5, tzinfo=UTC)

    assert offer_freshness({"observed_at": "2026-08-20T00:00:00+00:00"}, now=now) == "fresh"
    assert offer_freshness({"observed_at": "2026-07-20T00:00:00+00:00"}, now=now) == "aging"
    assert offer_freshness({"observed_at": "2026-05-01T00:00:00+00:00"}, now=now) == "stale"
    assert offer_freshness({}, now=now) == "unknown"


def test_title_alone_never_merges_products():
    left = _item(
        source="a",
        product_id="1",
        price=10000,
        observed_at="2026-09-05T00:00:00+00:00",
        brand="",
        mpn="",
        model="",
    )
    right = _item(
        source="b",
        product_id="2",
        price=11000,
        observed_at="2026-09-05T00:00:00+00:00",
        brand="",
        mpn="",
        model="",
    )
    left["title"] = right["title"] = "Generic Router"

    assert len(federate_catalog_items([left, right])) == 2


def test_staging_accumulates_sources_and_refreshes_one_source(tmp_path: Path):
    first = tmp_path / "supplier_a.json"
    second = tmp_path / "supplier_b.json"

    first.write_text(
        json.dumps({"schema_version": 2, "items": [
            _item(
                source="supplier-a",
                product_id="a-1",
                price=12000,
                observed_at="2026-09-04T08:00:00+00:00",
            )
        ]}),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps({"schema_version": 2, "items": [
            _item(
                source="supplier-b",
                product_id="b-1",
                price=11500,
                observed_at="2026-09-05T08:00:00+00:00",
            )
        ]}),
        encoding="utf-8",
    )

    service = CatalogStagingService(tmp_path / "staging.json")
    first_records = service.stage_file(first)
    first_staging_id = first_records[0]["staging_id"]

    merged_records = service.stage_file(second)
    assert len(merged_records) == 1
    assert merged_records[0]["staging_id"] == first_staging_id
    assert merged_records[0]["catalog_item"]["federation"]["source_count"] == 2
    assert len(merged_records[0]["catalog_item"]["offers"]) == 2

    payload = json.loads(first.read_text(encoding="utf-8"))
    payload["items"][0]["offer"]["price"] = 10500
    payload["items"][0]["offer"]["observed_at"] = "2026-09-06T08:00:00+00:00"
    first.write_text(json.dumps(payload), encoding="utf-8")

    refreshed = service.stage_file(first)
    assert len(refreshed) == 1
    assert refreshed[0]["staging_id"] == first_staging_id
    assert len(refreshed[0]["catalog_item"]["offers"]) == 2
    assert refreshed[0]["catalog_item"]["offer"]["source"] == "supplier-a"
    assert refreshed[0]["catalog_item"]["offer"]["price"] == 10500


def test_different_mpn_does_not_merge_even_when_brand_matches():
    items = [
        _item(
            source="a",
            product_id="1",
            price=10000,
            observed_at="2026-09-05T00:00:00+00:00",
            mpn="R1",
            model="",
        ),
        _item(
            source="b",
            product_id="2",
            price=10000,
            observed_at="2026-09-05T00:00:00+00:00",
            mpn="R2",
            model="",
        ),
    ]

    assert len(federate_catalog_items(items)) == 2
