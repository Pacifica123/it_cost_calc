from __future__ import annotations

import json
from pathlib import Path

import pytest

from application.services.catalog_enrichment_service import (
    IcecatClient,
    IcecatIdentityMismatch,
    apply_specification_source,
    icecat_lookup_identity,
    parse_icecat_specification,
    replace_specification_source,
)
from application.services.catalog_staging_service import (
    CatalogStagingService,
    STAGING_APPROVED,
    catalog_item_to_runtime_row,
)


def _feature(name: str, value, unit: str = "", feature_id: str = "1") -> dict:
    return {
        "RawValue": value,
        "PresentationValue": f"{value} {unit}".strip(),
        "Feature": {
            "ID": feature_id,
            "Name": {"Value": name, "Language": "EN"},
            "Measure": {"Signs": {"_": unit, "Language": "EN"}},
        },
    }


def _payload(*, gtin: str = "4601234567890", brand: str = "Vendor", mpn: str = "PC-16") -> dict:
    return {
        "msg": "OK",
        "data": {
            "GeneralInfo": {
                "IcecatId": 12345,
                "Title": "Vendor Office PC",
                "Brand": brand,
                "BrandPartCode": mpn,
                "GTIN": [gtin],
                "Category": {"Name": {"Value": "PC/workstation"}},
            },
            "FeaturesGroups": [
                {
                    "Features": [
                        _feature("Internal memory", 16, "GB", "ram"),
                        _feature("Processor cores", 8, "", "cpu"),
                        _feature("Total storage capacity", 1, "TB", "storage"),
                        _feature("Power consumption (max)", 250, "W", "power"),
                        _feature("Ethernet LAN (RJ-45) ports", 2, "", "ports"),
                        _feature("Ethernet LAN data rates", "10,100,1000", "Mbit/s", "lan"),
                        _feature("Maximum WLAN data transfer rate", 1.2, "Gbit/s", "wifi"),
                        _feature("IPv6 support", "Yes", "", "ipv6"),
                    ]
                }
            ],
        },
    }


def _catalog_payload(price: int = 65000) -> dict:
    return {
        "schema_version": 2,
        "items": [
            {
                "item_id": "supplier-pc-1",
                "title": "Vendor Office PC",
                "category": "prebuilt_pc",
                "source": "supplier-a",
                "source_product_id": "pc-1",
                "identity": {
                    "brand": "Vendor",
                    "mpn": "PC-16",
                    "gtin": "4601234567890",
                },
                "offer": {
                    "price": price,
                    "currency": "RUB",
                    "availability": "in_stock",
                    "observed_at": "2026-09-05T00:00:00+00:00",
                    "price_kind": "supplier_price",
                },
                "attributes": {"max_power_watts": 240},
            }
        ],
    }


def test_lookup_prefers_gtin_before_brand_mpn():
    lookup = icecat_lookup_identity(
        {"identity": {"gtin": "460 1234567890", "brand": "Vendor", "mpn": "PC-16"}}
    )

    assert lookup == {"matched_by": "gtin", "gtin": "4601234567890"}


def test_parse_icecat_maps_normalized_metrics_and_units():
    specification = parse_icecat_specification(
        _payload(),
        requested_identity={"gtin": "4601234567890"},
        matched_by="gtin",
        observed_at="2026-09-05T08:00:00+00:00",
    )

    assert specification["identity"]["brand"] == "Vendor"
    assert specification["identity"]["mpn"] == "PC-16"
    assert specification["metrics"] == {
        "ram_gb": 16,
        "cpu_cores": 8,
        "storage_gb": 1024,
        "max_power_watts": 250,
        "lan_ports": 2,
        "lan_speed_mbps": 1000,
        "wifi_total_mbps": 1200,
        "ipv6_support": True,
    }
    assert specification["mapped_features"]["ram_gb"]["name"] == "Internal memory"


def test_parse_icecat_rejects_mismatched_identity():
    with pytest.raises(IcecatIdentityMismatch):
        parse_icecat_specification(
            _payload(gtin="4609999999999"),
            requested_identity={"gtin": "4601234567890"},
            matched_by="gtin",
        )


def test_enrichment_fills_only_missing_metrics_and_records_conflict():
    item = _catalog_payload()["items"][0]
    specification = parse_icecat_specification(
        _payload(),
        requested_identity={"gtin": "4601234567890"},
        matched_by="gtin",
    )

    enriched = apply_specification_source(item, specification)

    assert enriched["attributes"]["ram_gb"] == 16
    assert enriched["attributes"]["max_power_watts"] == 240
    assert enriched["specification_sources"][0]["conflicts"]["attributes.max_power_watts"] == {
        "catalog": 240,
        "icecat": 250,
    }
    assert enriched["field_provenance"]["specifications"]["ram_gb"]["source"] == "icecat"
    assert "max_power_watts" not in enriched["field_provenance"]["specifications"]


def test_replace_specification_source_removes_previous_icecat_owned_values():
    item = _catalog_payload()["items"][0]
    first = parse_icecat_specification(
        _payload(),
        requested_identity={"gtin": "4601234567890"},
        matched_by="gtin",
    )
    enriched = apply_specification_source(item, first)
    second = dict(first)
    second["metrics"] = {**first["metrics"], "ram_gb": 32}

    refreshed, changed = replace_specification_source(enriched, second)

    assert changed is True
    assert refreshed["attributes"]["ram_gb"] == 32
    assert len(refreshed["specification_sources"]) == 1


def test_staging_preserves_icecat_enrichment_across_supplier_refresh(tmp_path: Path):
    source = tmp_path / "supplier.json"
    source.write_text(json.dumps(_catalog_payload()), encoding="utf-8")
    service = CatalogStagingService(tmp_path / "staging.json")
    record = service.stage_file(source)[0]
    service.set_status(record["staging_id"], STAGING_APPROVED)

    specification = parse_icecat_specification(
        _payload(),
        requested_identity={"gtin": "4601234567890"},
        matched_by="gtin",
        observed_at="2026-09-05T09:00:00+00:00",
    )
    enriched, _changed = replace_specification_source(record["source_catalog_item"], specification)
    updated = service.apply_source_item_updates({record["staging_id"]: enriched})[0]

    assert updated["status"] == "pending"
    assert updated["source_catalog_item"]["attributes"]["ram_gb"] == 16
    assert updated["source_catalog_item"]["attributes"]["max_power_watts"] == 240

    source.write_text(json.dumps(_catalog_payload(price=62000)), encoding="utf-8")
    refreshed = service.stage_file(source)[0]

    assert refreshed["catalog_item"]["offer"]["price"] == 62000
    assert refreshed["source_catalog_item"]["attributes"]["ram_gb"] == 16
    assert refreshed["source_catalog_item"]["attributes"]["max_power_watts"] == 240
    assert refreshed["source_catalog_item"]["specification_summary"]["sources"] == ["Open Icecat"]


def test_runtime_keeps_specification_provenance(tmp_path: Path):
    source = tmp_path / "supplier.json"
    source.write_text(json.dumps(_catalog_payload()), encoding="utf-8")
    service = CatalogStagingService(tmp_path / "staging.json")
    record = service.stage_file(source)[0]
    specification = parse_icecat_specification(
        _payload(),
        requested_identity={"gtin": "4601234567890"},
        matched_by="gtin",
    )
    enriched, _changed = replace_specification_source(record["source_catalog_item"], specification)
    updated = service.apply_source_item_updates({record["staging_id"]: enriched})[0]
    approved = service.set_status(updated["staging_id"], STAGING_APPROVED)

    _category, runtime = catalog_item_to_runtime_row(approved)

    assert runtime["ram_gb"] == 16
    assert runtime["catalog_metadata"]["specification_sources"][0]["source"] == "icecat"
    assert runtime["catalog_metadata"]["specification_summary"]["filled_metrics"]


class _FakeResponse:
    status_code = 200

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def json(self) -> dict:
        return self._payload


class _FakeSession:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[tuple[str, dict]] = []

    def get(self, url: str, **kwargs):
        self.calls.append((url, kwargs))
        return _FakeResponse(self.payload)


def test_client_uses_gtin_and_token_header_without_mutating_item():
    session = _FakeSession(_payload())
    client = IcecatClient(
        username="student",
        api_token="secret-token",
        session=session,
    )
    item = _catalog_payload()["items"][0]

    result = client.lookup(item)

    assert result.matched_by == "gtin"
    _url, kwargs = session.calls[0]
    assert kwargs["params"]["GTIN"] == "4601234567890"
    assert "Brand" not in kwargs["params"]
    assert kwargs["headers"]["api-token"] == "secret-token"
    assert "secret-token" not in json.dumps(result.specification)
