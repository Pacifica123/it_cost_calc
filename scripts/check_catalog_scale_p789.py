"""Portable smoke checks for large-catalog staging/read-model/local enrichment."""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from application.services.catalog_local_enrichment_service import enrich_staging_records_locally
from application.services.catalog_staging_service import CatalogStagingService


def _write_catalog(path: Path, rows: int = 4200) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter=";")
        writer.writerow(("Артикул", "Наименование", "Категория", "Цена с НДС, руб."))
        for index in range(rows):
            if index % 2:
                name = f"Office PC RAM 16 GB SSD 512 GB 8 cores {index}"
                category = "Готовые компьютеры"
            else:
                name = f"Router AX1800 4xRJ-45 1 Gbit/s 18 Вт {index}"
                category = "Маршрутизаторы"
            writer.writerow((f"SKU-{index}", name, category, 10000 + index))


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="catalog-scale-p789-") as tmp:
        root = Path(tmp)
        source = root / "catalog.csv"
        staging = root / "catalog_staging.json"
        _write_catalog(source)
        worker = CatalogStagingService(staging)
        records = worker.stage_file(source)
        assert len(records) == 4200
        assert staging.stat().st_size < 12 * 1024 * 1024
        assert worker.read_model.path.is_file()

        # Simulate a fresh GUI process: no full-record cache is available.
        gui = CatalogStagingService(staging)
        page, total = gui.page_projection(offset=250, limit=250)
        assert total == 4200 and len(page) == 250
        assert gui._records_cache is None
        summary = gui.summary_counts()
        assert summary["total"] == 4200
        one = gui.get_record(page[0]["staging_id"])
        assert one["catalog_item"]["offer"]["price"] > 0
        assert gui._records_cache is None

        # Account-free enrichment remains available and never overwrites price.
        target_id = page[0]["staging_id"]
        result = enrich_staging_records_locally(gui, staging_ids=(target_id,))
        assert result.requested == 1
        refreshed = gui.get_record(target_id)
        assert refreshed["catalog_item"]["offer"]["price"] > 0

    print("catalog-scale-p789: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
