import json
import zipfile
from pathlib import Path

from application.services.catalog_local_enrichment_service import (
    enrich_staging_records_locally,
)
from application.services.catalog_staging_service import (
    CatalogStagingService,
    iter_catalog_rows,
)


def _write_xlsx(path: Path, rows: int) -> None:
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
    data = [
        '<row r="1">'
        '<c r="A1" t="inlineStr"><is><t>Номенклатура</t></is></c>'
        '<c r="B1" t="inlineStr"><is><t>Товарная категория</t></is></c>'
        '<c r="C1" t="inlineStr"><is><t>Цена с НДС, руб.</t></is></c>'
        '<c r="D1" t="inlineStr"><is><t>PN</t></is></c>'
        '</row>'
    ]
    for index in range(1, rows + 1):
        row = index + 1
        data.append(
            f'<row r="{row}">'
            f'<c r="A{row}" t="inlineStr"><is><t>Маршрутизатор R{index} AX1200 4xRJ-45 1GbE 18 Вт</t></is></c>'
            f'<c r="B{row}" t="inlineStr"><is><t>Маршрутизаторы</t></is></c>'
            f'<c r="C{row}"><v>{8000 + index}</v></c>'
            f'<c r="D{row}" t="inlineStr"><is><t>R-{index}</t></is></c>'
            '</row>'
        )
    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'
        + "".join(data)
        + '</sheetData></worksheet>'
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("xl/workbook.xml", workbook_xml)
        archive.writestr("xl/_rels/workbook.xml.rels", rels_xml)
        archive.writestr("xl/worksheets/sheet1.xml", sheet_xml)


def test_supplier_price_alias_and_explicit_title_metrics_are_recovered(tmp_path: Path):
    source = tmp_path / "supplier.csv"
    source.write_text(
        "Номенклатура;Товарная категория;Цена с НДС, руб.;Бренд;PN\n"
        "Системный блок Office RAM 16GB SSD 512GB 8 ядер 300 Вт;Системные блоки;84 990;Example;PC-1\n",
        encoding="utf-8",
    )

    record = CatalogStagingService(tmp_path / "staging.json").stage_file(source)[0]
    item = record["catalog_item"]

    assert item["offer"]["price"] == 84990
    assert item["category"] == "prebuilt_pc"
    assert item["attributes"]["ram_gb"] == 16
    assert item["attributes"]["cpu_cores"] == 8
    assert item["attributes"]["storage_gb"] == 512
    assert item["attributes"]["max_power_watts"] == 300


def test_streaming_xlsx_honours_optional_scope_limit(tmp_path: Path):
    source = tmp_path / "large.xlsx"
    _write_xlsx(source, 1200)

    rows = list(iter_catalog_rows(source, max_rows=137))

    assert len(rows) == 137
    assert rows[0]["Номенклатура"].startswith("Маршрутизатор R1")
    assert rows[-1]["Цена с НДС, руб."] == "8137"


def test_large_staging_is_compact_and_paginated(tmp_path: Path):
    source = tmp_path / "large.xlsx"
    _write_xlsx(source, 1600)
    staging_path = tmp_path / "staging.json"
    service = CatalogStagingService(staging_path)

    records = service.stage_file(source)
    first_page, total = service.page_records(offset=0, limit=100)
    second_page, second_total = service.page_records(offset=100, limit=100)
    payload = json.loads(staging_path.read_text(encoding="utf-8"))

    assert len(records) == total == second_total == 1600
    assert len(first_page) == len(second_page) == 100
    assert first_page[0]["staging_id"] != second_page[0]["staging_id"]
    assert payload["schema_version"] == 3
    assert "catalog_item" not in payload["records"][0]
    assert "validation_errors" not in payload["records"][0]
    assert "source_observations" not in payload["records"][0]["source_catalog_item"]
    assert staging_path.stat().st_size < 8 * 1024 * 1024


def test_account_free_enrichment_fills_only_missing_values(tmp_path: Path):
    source = tmp_path / "catalog.csv"
    source.write_text(
        "name;category;price\n"
        "Branch Router;router;\n"
        "Office PC;prebuilt_pc;99000\n",
        encoding="utf-8",
    )
    service = CatalogStagingService(tmp_path / "staging.json")
    records = service.stage_file(source)

    summary = enrich_staging_records_locally(service)
    refreshed = service.list_records()
    router = next(record for record in refreshed if record["catalog_item"]["category"] == "router")
    pc = next(record for record in refreshed if record["catalog_item"]["category"] == "prebuilt_pc")

    assert summary.changed == 2
    assert router["catalog_item"]["attributes"]["lan_ports"] == 4
    assert router["catalog_item"]["offer"]["price"] > 0
    assert router["catalog_item"]["offer"]["price_kind"] == "estimated_price"
    assert pc["catalog_item"]["offer"]["price"] == 99000
    assert pc["catalog_item"]["offer"]["price_kind"] != "estimated_price"
    assert pc["catalog_item"]["attributes"]["ram_gb"] == 16
    assert any(
        source.get("source") == "offline-heuristic-v1"
        for source in pc["catalog_item"].get("specification_sources", [])
    )


def test_sqlite_read_model_serves_ui_without_loading_full_json(tmp_path: Path):
    source = tmp_path / "catalog.csv"
    source.write_text(
        "name;category;price;brand;mpn\n"
        "Маршрутизатор Alpha AX1800;router;12000;NetBrand;NB-A1\n"
        "Office PC RAM 16GB SSD 512GB;prebuilt_pc;78000;PCBrand;PC-16\n",
        encoding="utf-8",
    )
    staging_path = tmp_path / "catalog_staging.json"
    writer = CatalogStagingService(staging_path)
    writer.stage_file(source)
    assert writer.read_model.path.is_file()

    class NoFullJsonRead:
        def read(self, _path):
            raise AssertionError("UI projection must not read the complete staging JSON")

        def write(self, *_args, **_kwargs):
            raise AssertionError("read-only test")

    reader = CatalogStagingService(staging_path, storage=NoFullJsonRead())
    page, total = reader.page_projection(query="netbrand", offset=0, limit=10)
    counts = reader.summary_counts()
    record = reader.get_record(page[0]["staging_id"])

    assert total == 1
    assert page[0]["title"] == "Маршрутизатор Alpha AX1800"
    assert counts["total"] == 2
    assert record["catalog_item"]["identity"]["brand"] == "NetBrand"
    assert reader._records_cache is None
