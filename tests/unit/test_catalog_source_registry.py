from __future__ import annotations

import json
from pathlib import Path

from application.services.catalog_source_registry import CatalogSourceRegistry


def test_registry_merges_presets_and_saved_sources(tmp_path: Path):
    presets = tmp_path / "presets.json"
    presets.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sources": [
                    {
                        "id": "technocity",
                        "name": "ТехноСити",
                        "location": "https://example.test/price.xlsx",
                        "format": "xlsx",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    registry = CatalogSourceRegistry(tmp_path / "user.json", presets_path=presets)

    assert registry.list_sources()[0]["preset"] is True
    saved = registry.save_source(
        {
            "id": "my-supplier",
            "name": "Мой поставщик",
            "location": "https://supplier.test/catalog.yml",
            "format": "yml",
            "region": "Россия",
        }
    )

    assert saved["preset"] is False
    assert registry.get_source("my-supplier")["format"] == "yml"
    assert len(registry.list_sources()) == 2
