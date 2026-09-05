"""Persistent configuration for structured catalog feed sources."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from infrastructure.storage import JsonFileStorage

CATALOG_SOURCE_REGISTRY_VERSION = 1
SUPPORTED_FEED_FORMATS = ("auto", "xlsx", "csv", "yml", "xml")
SUPPORTED_DOWNLOAD_STRATEGIES = ("direct", "yandex_disk_public")


class CatalogSourceRegistry:
    """Merge bundled source presets with user-defined feed sources.

    Presets are immutable application resources. User sources are stored under the
    writable runtime data directory and may override a preset with the same id.
    """

    def __init__(
        self,
        user_path: str | Path,
        *,
        presets_path: str | Path | None = None,
        storage: JsonFileStorage | None = None,
    ) -> None:
        self.user_path = Path(user_path)
        self.presets_path = Path(presets_path) if presets_path is not None else None
        self.storage = storage or JsonFileStorage()

    def list_sources(self) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for raw in self._read_sources(self.presets_path):
            source = normalize_catalog_source(raw, preset=True)
            merged[source["id"]] = source
        for raw in self._read_sources(self.user_path):
            source = normalize_catalog_source(raw, preset=False)
            merged[source["id"]] = source
        return sorted(merged.values(), key=lambda item: (not bool(item.get("preset")), item["name"].lower()))

    def get_source(self, source_id: str) -> dict[str, Any]:
        requested = str(source_id or "").strip()
        for source in self.list_sources():
            if source["id"] == requested:
                return deepcopy(source)
        raise KeyError(requested)

    def save_source(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        source = normalize_catalog_source(raw, preset=False)
        existing = {
            item["id"]: item
            for item in self._read_sources(self.user_path)
            if isinstance(item, Mapping) and str(item.get("id") or "").strip()
        }
        existing[source["id"]] = {key: value for key, value in source.items() if key != "preset"}
        payload = {
            "schema_version": CATALOG_SOURCE_REGISTRY_VERSION,
            "sources": [existing[key] for key in sorted(existing)],
        }
        self.storage.write(self.user_path, payload)
        return deepcopy(source)

    def _read_sources(self, path: Path | None) -> list[dict[str, Any]]:
        if path is None or not path.exists():
            return []
        payload = self.storage.read(path)
        rows = payload.get("sources", []) if isinstance(payload, Mapping) else []
        return [dict(row) for row in rows if isinstance(row, Mapping)]


def normalize_catalog_source(raw: Mapping[str, Any], *, preset: bool = False) -> dict[str, Any]:
    source_id = _slug(str(raw.get("id") or raw.get("name") or ""))
    if not source_id:
        raise ValueError("У источника должен быть идентификатор.")
    name = str(raw.get("name") or source_id).strip()
    location = str(raw.get("location") or raw.get("url") or raw.get("path") or "").strip()
    if not location:
        raise ValueError("У источника должен быть URL или локальный путь.")
    feed_format = str(raw.get("format") or "auto").strip().lower()
    if feed_format not in SUPPORTED_FEED_FORMATS:
        raise ValueError(f"Неподдерживаемый формат источника: {feed_format}")
    strategy = str(raw.get("download_strategy") or "direct").strip().lower()
    if strategy not in SUPPORTED_DOWNLOAD_STRATEGIES:
        raise ValueError(f"Неподдерживаемая стратегия загрузки: {strategy}")
    price_kind = str(raw.get("price_kind") or "supplier_price").strip() or "supplier_price"
    return {
        "id": source_id,
        "name": name,
        "location": location,
        "format": feed_format,
        "region": str(raw.get("region") or "").strip(),
        "price_kind": price_kind,
        "download_strategy": strategy,
        "homepage": str(raw.get("homepage") or "").strip(),
        "notes": str(raw.get("notes") or "").strip(),
        "preset": bool(preset),
    }


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9а-яА-Я]+", "-", value.strip().lower()).strip("-")
    return re.sub(r"-+", "-", normalized)


__all__ = [
    "CATALOG_SOURCE_REGISTRY_VERSION",
    "CatalogSourceRegistry",
    "SUPPORTED_DOWNLOAD_STRATEGIES",
    "SUPPORTED_FEED_FORMATS",
    "normalize_catalog_source",
]
