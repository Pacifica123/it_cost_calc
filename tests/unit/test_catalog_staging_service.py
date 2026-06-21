import json
import zipfile
from pathlib import Path

import pytest

from application.services.catalog_staging_service import (
    CATALOG_STAGING_SCHEMA_VERSION,
    CatalogStagingService,
    STAGING_APPROVED,
    STAGING_BLOCKED,
    catalog_item_to_runtime_row,
    load_catalog_rows,
    staging_record_readiness,
)
from ui_qt.presenters import CatalogStagingPresenter, QtAppPresenter


def _catalog_payload() -> dict:
    return {
        "schema_version": 2,
        "items": [
            {
                "item_id": "router-1",
                "title": "Router One",
                "category": "router",
                "source": "fixture",
                "identity": {"brand": "Example", "model": "R1"},
                "offer": {
                    "price": 5000,
                    "currency": "RUB",
                    "availability": "in_stock",
                    "observed_at": "2026-06-21T12:00:00Z",
                    "url": "https://example.test/router-1",
                },
                "attributes": {
                    "lan_ports": 4,
                    "lan_speed_mbps": 1000,
                    "max_power_watts": 12,
                },
            },
            {
                "item_id": "cpu-1",
                "title": "CPU One",
                "category": "cpu",
                "source": "fixture",
                "offer": {
                    "price": 20000,
                    "currency": "RUB",
                    "observed_at": "2026-06-21T12:00:00Z",
                },
            },
        ],
    }


def test_staging_blocks_parts_and_converts_approved_router(tmp_path: Path):
    source = tmp_path / "catalog.json"
    source.write_text(json.dumps(_catalog_payload()), encoding="utf-8")
    service = CatalogStagingService(tmp_path / "staging.json")

    records = service.stage_file(source)

    assert len(records) == 2
    assert records[0]["status"] != STAGING_BLOCKED
    assert records[1]["status"] == STAGING_BLOCKED
    approved = service.set_status(records[0]["staging_id"], STAGING_APPROVED)
    category, row = catalog_item_to_runtime_row(approved)
    assert category == "network"
    assert row["component_type"] == "network_device"
    assert row["origin"] == "catalog"
    assert row["lan_ports"] == 4


def test_staging_refuses_to_approve_blocked_item(tmp_path: Path):
    source = tmp_path / "catalog.json"
    source.write_text(json.dumps(_catalog_payload()), encoding="utf-8")
    service = CatalogStagingService(tmp_path / "staging.json")
    records = service.stage_file(source)

    with pytest.raises(ValueError, match="блокирующ"):
        service.set_status(records[1]["staging_id"], STAGING_APPROVED)


def test_flat_csv_is_supported(tmp_path: Path):
    source = tmp_path / "catalog.csv"
    source.write_text(
        "name;category;price;currency;brand;model;ram_gb;cpu_cores;storage_gb;max_power_watts\n"
        "Office PC;prebuilt_pc;65000;RUB;Example;P1;16;8;512;250\n",
        encoding="utf-8",
    )

    rows = load_catalog_rows(source)
    records = CatalogStagingService(tmp_path / "staging.json").stage_file(source)

    assert rows[0]["name"] == "Office PC"
    assert records[0]["catalog_item"]["attributes"]["ram_gb"] == 16
    assert records[0]["validation_errors"] == []


def test_schema_v1_catalog_remains_supported(tmp_path: Path):
    source = tmp_path / "legacy.json"
    source.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "items": [
                    {
                        "item_id": "legacy-router",
                        "title": "Legacy Router",
                        "category": "router",
                        "source": "legacy",
                        "price_rub": 3000,
                        "currency": "RUB",
                        "attributes": {"lan_ports": 4, "max_power_watts": 9},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    record = CatalogStagingService(tmp_path / "staging.json").stage_file(source)[0]

    assert record["catalog_item"]["source_schema_version"] == 1
    assert record["catalog_item"]["offer"]["price"] == 3000
    assert record["validation_errors"] == []


def test_simple_xlsx_price_list_is_supported(tmp_path: Path):
    source = tmp_path / "catalog.xlsx"
    workbook_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
      xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
      <sheets><sheet name="Catalog" sheetId="1" r:id="rId1"/></sheets>
    </workbook>"""
    rels_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
      <Relationship Id="rId1"
        Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet"
        Target="worksheets/sheet1.xml"/>
    </Relationships>"""
    sheet_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
      <sheetData>
        <row r="1">
          <c r="A1" t="inlineStr"><is><t>name</t></is></c>
          <c r="B1" t="inlineStr"><is><t>category</t></is></c>
          <c r="C1" t="inlineStr"><is><t>price</t></is></c>
          <c r="D1" t="inlineStr"><is><t>currency</t></is></c>
          <c r="E1" t="inlineStr"><is><t>max_power_watts</t></is></c>
        </row>
        <row r="2">
          <c r="A2" t="inlineStr"><is><t>Branch Router</t></is></c>
          <c r="B2" t="inlineStr"><is><t>router</t></is></c>
          <c r="C2"><v>8200</v></c>
          <c r="D2" t="inlineStr"><is><t>RUB</t></is></c>
          <c r="E2"><v>15</v></c>
        </row>
      </sheetData>
    </worksheet>"""
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("xl/workbook.xml", workbook_xml)
        archive.writestr("xl/_rels/workbook.xml.rels", rels_xml)
        archive.writestr("xl/worksheets/sheet1.xml", sheet_xml)

    rows = load_catalog_rows(source)
    record = CatalogStagingService(tmp_path / "staging.json").stage_file(source)[0]

    assert rows[0]["name"] == "Branch Router"
    assert record["catalog_item"]["offer"]["price"] == 8200
    assert record["catalog_item"]["attributes"]["max_power_watts"] == 15


def test_presenter_imports_only_approved_rows_and_avoids_duplicates(tmp_path: Path):
    source = tmp_path / "catalog.json"
    source.write_text(json.dumps(_catalog_payload()), encoding="utf-8")
    app = QtAppPresenter(
        repo_root=Path(__file__).resolve().parents[2],
        runtime_entities_path=tmp_path / "runtime.json",
    )
    presenter = CatalogStagingPresenter(
        app,
        staging_path=tmp_path / "staging.json",
    )
    presenter.stage_file(source)
    presenter.approve(0)

    first = presenter.import_approved()
    second_presenter = CatalogStagingPresenter(
        app,
        staging_path=tmp_path / "second-staging.json",
    )
    second_presenter.stage_file(source)
    second_presenter.approve(0)
    second = second_presenter.import_approved()

    assert first == {"imported": 1, "skipped": 0}
    assert second == {"imported": 0, "skipped": 1}
    assert app.list_entities("network")[0]["catalog_item_id"] == "router-1"


def test_editor_unblocks_item_and_preserves_runtime_inputs(tmp_path: Path):
    source = tmp_path / "catalog.json"
    source.write_text(json.dumps(_catalog_payload()), encoding="utf-8")
    service = CatalogStagingService(tmp_path / "staging.json")
    cpu_record = service.stage_file(source)[1]

    edited = service.update_record(
        cpu_record["staging_id"],
        {
            "title": "Office PC",
            "category": "prebuilt_pc",
            "price": "65000",
            "currency": "RUB",
            "target_category": "client",
            "target_component_type": "workstation",
            "quantity": "2",
            "client_seats": "2",
            "attributes": {
                "ram_gb": "16",
                "cpu_cores": "8",
                "storage_gb": "512",
                "max_power_watts": "250",
            },
        },
    )

    assert edited["status"] == "pending"
    assert edited["validation_errors"] == []
    assert edited["source_catalog_item"]["title"] == "CPU One"
    assert edited["manual_overrides"]["title"] == "Office PC"
    approved = service.set_status(edited["staging_id"], STAGING_APPROVED)
    category, row = catalog_item_to_runtime_row(approved)
    assert category == "client"
    assert row["quantity"] == 2
    assert row["client_seats"] == 2
    assert row["ram_gb"] == 16
    assert staging_record_readiness(approved) == "import_ready"


def test_manual_corrections_survive_catalog_refresh(tmp_path: Path):
    payload = _catalog_payload()
    source = tmp_path / "catalog.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    service = CatalogStagingService(tmp_path / "staging.json")
    record = service.stage_file(source)[0]
    values = {
        "title": "Router Corrected",
        "category": "router",
        "price": "6200",
        "currency": "RUB",
        "target_category": "network",
        "target_component_type": "network_device",
        "quantity": "3",
        "client_seats": "0",
        "attributes": {
            "lan_ports": "8",
            "lan_speed_mbps": "1000",
            "max_power_watts": "18",
        },
    }
    edited = service.update_record(record["staging_id"], values)
    service.set_status(edited["staging_id"], STAGING_APPROVED)

    payload["items"][0]["title"] = "Router Source Renamed"
    payload["items"][0]["offer"]["price"] = 5500
    source.write_text(json.dumps(payload), encoding="utf-8")
    refreshed = service.stage_file(source)[0]

    assert refreshed["source_catalog_item"]["title"] == "Router Source Renamed"
    assert refreshed["source_catalog_item"]["offer"]["price"] == 5500
    assert refreshed["catalog_item"]["title"] == "Router Corrected"
    assert refreshed["catalog_item"]["offer"]["price"] == 6200
    assert refreshed["runtime_inputs"]["quantity"] == 3
    assert refreshed["status"] == "pending"
    assert any("Источник изменился" in warning for warning in refreshed["validation_warnings"])
    saved = json.loads((tmp_path / "staging.json").read_text(encoding="utf-8"))
    assert saved["schema_version"] == CATALOG_STAGING_SCHEMA_VERSION == 2


def test_bulk_approval_skips_blocked_records(tmp_path: Path):
    source = tmp_path / "catalog.json"
    source.write_text(json.dumps(_catalog_payload()), encoding="utf-8")
    service = CatalogStagingService(tmp_path / "staging.json")
    records = service.stage_file(source)

    result = service.set_status_many(
        [record["staging_id"] for record in records],
        STAGING_APPROVED,
    )

    assert result == {"updated": 1, "blocked": 1, "skipped": 0}
    assert service.list_records()[0]["status"] == STAGING_APPROVED
    assert service.list_records()[1]["status"] == STAGING_BLOCKED


def test_numeric_metric_one_is_not_converted_to_boolean(tmp_path: Path):
    payload = _catalog_payload()
    payload["items"][0]["attributes"]["lan_ports"] = 1
    source = tmp_path / "catalog.json"
    source.write_text(json.dumps(payload), encoding="utf-8")

    record = CatalogStagingService(tmp_path / "staging.json").stage_file(source)[0]

    assert record["catalog_item"]["attributes"]["lan_ports"] == 1
    assert record["catalog_item"]["attributes"]["lan_ports"] is not True


def test_presenter_filters_rows_and_shows_manual_diff(tmp_path: Path):
    source = tmp_path / "catalog.json"
    source.write_text(json.dumps(_catalog_payload()), encoding="utf-8")
    app = QtAppPresenter(
        repo_root=Path(__file__).resolve().parents[2],
        runtime_entities_path=tmp_path / "runtime.json",
    )
    presenter = CatalogStagingPresenter(app, staging_path=tmp_path / "staging.json")
    records = presenter.stage_file(source)
    staging_id = records[0]["staging_id"]
    values = presenter.edit_values(staging_id)
    values["price"] = "6100"
    values["quantity"] = "2"
    presenter.save_edits(staging_id, values)

    blocked_rows = presenter.table_rows("blocked")
    pending_rows = presenter.table_rows("pending")
    details = presenter.details_text_by_id(staging_id)

    assert len(blocked_rows) == 1
    assert pending_rows[0]["staging_id"] == staging_id
    assert pending_rows[0]["readiness"] == "Нужна проверка"
    assert "Цена: 5000" in details
    assert "6100" in details
    assert "Количество: 1" in details
