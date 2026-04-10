from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..catalog_builder import build_catalog_payload, deduplicate_items, normalize_dns_snapshot
from ..catalog_schema import CatalogSourceInfo
from ..paths import DEFAULT_EXAMPLE_SNAPSHOT, DEFAULT_ROUTERS_SNAPSHOT



def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))



def _normalize_router_list(router_list: list[dict[str, Any]]) -> dict[str, Any]:
    return {"routers": router_list, "prebuilt_pcs": [], "components": [], "compatible_builds": []}



def build_catalog_from_example_snapshots() -> dict[str, Any]:
    snapshots: list[tuple[str, dict[str, Any], str]] = []

    if DEFAULT_EXAMPLE_SNAPSHOT.exists():
        snapshots.append((DEFAULT_EXAMPLE_SNAPSHOT.name, _load_json(DEFAULT_EXAMPLE_SNAPSHOT), "example"))

    if DEFAULT_ROUTERS_SNAPSHOT.exists():
        router_snapshot = _load_json(DEFAULT_ROUTERS_SNAPSHOT)
        if isinstance(router_snapshot, list) and router_snapshot:
            snapshots.append(
                (
                    DEFAULT_ROUTERS_SNAPSHOT.name,
                    _normalize_router_list(router_snapshot),
                    "example-router-list",
                )
            )

    all_items = []
    source_infos = []
    for snapshot_name, snapshot, mode in snapshots:
        normalized_items = normalize_dns_snapshot(snapshot, snapshot_name=snapshot_name)
        deduplicated_items = deduplicate_items(normalized_items)
        all_items.extend(deduplicated_items)
        source_infos.append(
            CatalogSourceInfo(
                source="dns",
                snapshot_name=snapshot_name,
                mode=mode,
                items_before_dedup=len(normalized_items),
                items_after_dedup=len(deduplicated_items),
            )
        )

    combined_items = deduplicate_items(all_items)
    return build_catalog_payload(
        items=combined_items,
        sources=source_infos,
        generated_by="tools.catalog_parser.sources.dns_examples",
    )
