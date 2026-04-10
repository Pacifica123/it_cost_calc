from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from ..catalog_builder import build_catalog_payload, deduplicate_items, normalize_dns_snapshot
from ..catalog_schema import CatalogSourceInfo
from ..legacy import main as run_legacy_dns_parser



def build_catalog_from_live_dns() -> dict:
    """Запускает legacy-парсер DNS во временной директории и нормализует результат.

    Этот режим оставлен как техническая заготовка на будущее и не используется
    в стандартном тестовом контуре проекта.
    """

    with tempfile.TemporaryDirectory(prefix="dns_live_") as temp_dir:
        temp_path = Path(temp_dir)
        previous_cwd = Path.cwd()
        try:
            os.chdir(temp_path)
            run_legacy_dns_parser()
            raw_snapshot_path = temp_path / "dns_data_playwright.json"
            snapshot = json.loads(raw_snapshot_path.read_text(encoding="utf-8"))
        finally:
            os.chdir(previous_cwd)

    normalized_items = normalize_dns_snapshot(snapshot, snapshot_name="dns_data_playwright.json")
    deduplicated_items = deduplicate_items(normalized_items)
    return build_catalog_payload(
        items=deduplicated_items,
        sources=[
            CatalogSourceInfo(
                source="dns",
                snapshot_name="dns_data_playwright.json",
                mode="legacy-live",
                items_before_dedup=len(normalized_items),
                items_after_dedup=len(deduplicated_items),
            )
        ],
        generated_by="tools.catalog_parser.sources.dns_live",
    )
