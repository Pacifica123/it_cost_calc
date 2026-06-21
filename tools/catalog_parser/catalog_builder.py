from __future__ import annotations

import json
import re
from collections import Counter
from datetime import UTC, datetime
from hashlib import sha1
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

from .catalog_schema import CatalogItem, CatalogSourceInfo
from .dns_network_metrics import merge_parsed_metrics_with_specs, parse_dns_network_title

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


def _source_product_id(url: str | None) -> str | None:
    if not url:
        return None
    parts = [part for part in urlparse(str(url)).path.split("/") if part]
    if "product" in parts:
        index = parts.index("product")
        if index + 1 < len(parts):
            return parts[index + 1]
    return parts[-1] if parts else None


def _first_text(raw: dict[str, Any], specs: dict[str, Any], *keys: str) -> str | None:
    lowered = {str(key).strip().lower(): value for key, value in specs.items()}
    for key in keys:
        value = raw.get(key)
        if value in (None, ""):
            value = lowered.get(key.lower())
        text = str(value or "").strip()
        if text:
            return text
    return None



def _normalize_dns_item(raw: dict[str, Any], *, bucket: str, snapshot_name: str) -> CatalogItem:
    source_category = raw.get("type") or bucket
    normalized_category = DNS_CATEGORY_MAP.get(source_category, DNS_CATEGORY_MAP.get(bucket, "component"))
    title = str(raw.get("title") or "Без названия").strip()
    url = raw.get("url")
    price_rub = raw.get("price_int")
    if not isinstance(price_rub, int):
        price_rub = None
    attributes = dict(raw.get("specs") or {})
    if normalized_category == "router":
        parse_result = parse_dns_network_title(title)
        attributes = merge_parsed_metrics_with_specs(attributes, parse_result)
    observed_at = datetime.now(UTC).isoformat()
    source_product_id = _source_product_id(url)
    identity = {
        key: value
        for key, value in {
            "brand": _first_text(raw, attributes, "brand", "производитель"),
            "model": _first_text(raw, attributes, "model", "модель"),
            "mpn": _first_text(raw, attributes, "mpn", "артикул производителя", "part number"),
            "gtin": _first_text(raw, attributes, "gtin", "ean", "штрихкод"),
        }.items()
        if value
    }
    parse_warnings = list(attributes.get("parse_warnings") or [])
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
        source_product_id=source_product_id,
        identity=identity,
        offer={
            "price": price_rub,
            "currency": "RUB",
            "availability": str(raw.get("availability") or "unknown"),
            "url": url,
            "region": str(raw.get("region") or ""),
            "observed_at": observed_at,
        },
        field_provenance={
            "title": {"source": "dns", "method": "snapshot", "confidence": 1.0},
            "price": {"source": "dns", "method": "snapshot", "confidence": 1.0},
            "attributes": {
                "source": "dns",
                "method": str(attributes.get("parse_source") or "snapshot-specs"),
                "confidence": float(attributes.get("confidence", 1.0) or 0.0),
            },
        },
        review={
            "status": "pending",
            "warnings": parse_warnings,
        },
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
        identity = item.identity
        dedup_key = (
            (f"gtin:{identity['gtin']}" if identity.get("gtin") else None)
            or (
                f"mpn:{identity.get('brand', '').lower()}:{identity['mpn'].lower()}"
                if identity.get("mpn")
                else None
            )
            or (
                f"model:{identity['brand'].lower()}:{identity['model'].lower()}"
                if identity.get("brand") and identity.get("model")
                else None
            )
            or (f"source:{item.source}:{item.source_product_id}" if item.source_product_id else None)
            or item.url
            or item.item_id
        )
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
        "schema_version": 2,
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
