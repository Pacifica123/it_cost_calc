from tools.catalog_parser.catalog_builder import build_catalog_payload, deduplicate_items, normalize_dns_snapshot
from tools.catalog_parser.catalog_schema import CatalogSourceInfo


SAMPLE_SNAPSHOT = {
    "routers": [
        {
            "title": "Wi-Fi роутер TP-Link Archer AX1500",
            "url": "https://example.test/router-1",
            "price_int": 3799,
            "specs": {"wifi": "ax"},
        }
    ],
    "prebuilt_pcs": [],
    "components": [
        {
            "title": "Процессор AMD Ryzen 7 7800X3D OEM",
            "url": "https://example.test/cpu-1",
            "price_int": 30999,
            "specs": {"socket": "AM5"},
            "type": "cpus",
        },
        {
            "title": "Процессор AMD Ryzen 7 7800X3D OEM",
            "url": "https://example.test/cpu-1",
            "price_int": 30999,
            "specs": {"socket": "AM5"},
            "type": "cpus",
        },
    ],
    "compatible_builds": [],
}



def test_normalize_dns_snapshot_maps_categories_and_fields() -> None:
    items = normalize_dns_snapshot(SAMPLE_SNAPSHOT, snapshot_name="sample.json")
    assert len(items) == 3
    router = items[0]
    cpu = items[1]
    assert router.category == "router"
    assert cpu.category == "cpu"
    assert cpu.attributes["socket"] == "AM5"



def test_deduplicate_and_build_catalog_payload() -> None:
    items = deduplicate_items(normalize_dns_snapshot(SAMPLE_SNAPSHOT, snapshot_name="sample.json"))
    payload = build_catalog_payload(
        items=items,
        sources=[
            CatalogSourceInfo(
                source="dns",
                snapshot_name="sample.json",
                mode="unit-test",
                items_before_dedup=3,
                items_after_dedup=2,
            )
        ],
        generated_by="unit-test",
    )
    assert payload["stats"]["items_total"] == 2
    assert payload["stats"]["by_category"]["cpu"] == 1
    assert payload["sources"][0]["items_after_dedup"] == 2
