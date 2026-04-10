from __future__ import annotations

import json
import re
from collections import Counter
from datetime import UTC, datetime
from hashlib import sha1
from pathlib import Path
from typing import Any, Iterable

from .catalog_schema import CatalogItem, CatalogSourceInfo

DNS_CATEGORY_MAP = {
    "routers": "router",
    "prebuilt_pcs": "prebuilt_pc",
    "components": "component",
    "motherboards": "motherboard",
    "cpus": "cpu",
    "gpus": "gpu",
    "rams": "ram",
    "psus": "psu",
    "ssds": "ssd",
    "hdds": "hdd",
    "cases": "case",
}


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9а-яА-Я]+", "-", value).strip("-").lower()
    return re.sub(r"-+", "-", slug) or "item"



def make_item_id(title: str, url: str | None) -> str:
    digest_source = url or title
    digest = sha1(digest_source.encode("utf-8")).hexdigest()[:10]
    return f"{slugify(title)[:64]}-{digest}"



def _normalize_dns_item(raw: dict[str, Any], *, bucket: str, snapshot_name: str) -> CatalogItem:
    source_category = raw.get("type") or bucket
    normalized_category = DNS_CATEGORY_MAP.get(source_category, DNS_CATEGORY_MAP.get(bucket, "component"))
    title = str(raw.get("title") or "Без названия").strip()
    url = raw.get("url")
    price_rub = raw.get("price_int")
    if not isinstance(price_rub, int):
        price_rub = None
    attributes = dict(raw.get("specs") or {})
    return CatalogItem(
        item_id=make_item_id(title, url),
        title=title,
        category=normalized_category,
        source="dns",
        source_category=source_category,
        price_rub=price_rub,
        currency="RUB",
        availability=str(raw.get("availability") or "unknown"),
        url=url,
        attributes=attributes,
        raw_snapshot=snapshot_name,
    )



def normalize_dns_snapshot(snapshot: dict[str, Any], *, snapshot_name: str) -> list[CatalogItem]:
    items: list[CatalogItem] = []
    for bucket_name, bucket_value in snapshot.items():
        if not isinstance(bucket_value, list):
            continue
        if bucket_name == "compatible_builds":
            continue
        for raw_item in bucket_value:
            if not isinstance(raw_item, dict):
                continue
            items.append(_normalize_dns_item(raw_item, bucket=bucket_name, snapshot_name=snapshot_name))
    return items



def deduplicate_items(items: Iterable[CatalogItem]) -> list[CatalogItem]:
    unique: dict[str, CatalogItem] = {}
    for item in items:
        dedup_key = item.url or item.item_id
        if dedup_key not in unique:
            unique[dedup_key] = item
    return list(unique.values())



def build_catalog_payload(
    *,
    items: Iterable[CatalogItem],
    sources: Iterable[CatalogSourceInfo],
    generated_by: str,
) -> dict[str, Any]:
    item_list = list(items)
    stats_by_category = Counter(item.category for item in item_list)
    stats_by_source = Counter(item.source for item in item_list)
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "generated_by": generated_by,
        "stats": {
            "items_total": len(item_list),
            "by_category": dict(sorted(stats_by_category.items())),
            "by_source": dict(sorted(stats_by_source.items())),
        },
        "sources": [source.to_dict() for source in sources],
        "items": [item.to_dict() for item in item_list],
    }



def save_catalog(payload: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path
