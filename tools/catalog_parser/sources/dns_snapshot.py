from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from ..catalog_builder import build_catalog_payload, deduplicate_items, normalize_dns_snapshot
from ..catalog_schema import CatalogSourceInfo

MANIFEST_NAME = "snapshot_manifest.json"
_PRICE_RE = re.compile(r"\d+(?:[.,]\d+)?")


class _ProductHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.json_ld: list[str] = []
        self.meta: dict[str, str] = {}
        self.h1_parts: list[str] = []
        self.title_parts: list[str] = []
        self.specs: dict[str, str] = {}
        self._capture: str | None = None
        self._parts: list[str] = []
        self._last_dt: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {str(key).lower(): str(value or "") for key, value in attrs}
        if tag == "script" and attributes.get("type", "").lower() == "application/ld+json":
            self._capture, self._parts = "json_ld", []
        elif tag in {"h1", "title", "dt", "dd"}:
            self._capture, self._parts = tag, []
        elif tag in {"meta", "link"}:
            key = attributes.get("itemprop") or attributes.get("property") or attributes.get("name")
            value = attributes.get("content") or attributes.get("href")
            if key and value:
                self.meta[key.lower()] = value

    def handle_data(self, data: str) -> None:
        if self._capture:
            self._parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._capture != tag and not (tag == "script" and self._capture == "json_ld"):
            return
        value = " ".join("".join(self._parts).split())
        if self._capture == "json_ld" and value:
            self.json_ld.append(value)
        elif self._capture == "h1" and value:
            self.h1_parts.append(value)
        elif self._capture == "title" and value:
            self.title_parts.append(value)
        elif self._capture == "dt":
            self._last_dt = value or None
        elif self._capture == "dd" and value and self._last_dt:
            self.specs[self._last_dt] = value
            self._last_dt = None
        self._capture, self._parts = None, []


def _iter_json_nodes(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, list):
        for item in value:
            yield from _iter_json_nodes(item)
    elif isinstance(value, dict):
        yield value
        for nested_key in ("@graph", "mainEntity", "itemListElement"):
            if nested_key in value:
                yield from _iter_json_nodes(value[nested_key])


def _is_product(node: dict[str, Any]) -> bool:
    types = node.get("@type")
    if isinstance(types, str):
        types = [types]
    return any(str(value).lower() == "product" for value in (types or []))


def _first_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return next((item for item in value if isinstance(item, dict)), {})
    return {}


def _text(value: Any) -> str | None:
    if isinstance(value, dict):
        value = value.get("name") or value.get("value")
    text = str(value or "").strip()
    return text or None


def _price(value: Any) -> int | None:
    match = _PRICE_RE.search(str(value or "").replace(" ", ""))
    if not match:
        return None
    return round(float(match.group().replace(",", ".")))


def _availability(value: Any) -> str:
    token = str(value or "").rsplit("/", 1)[-1].lower()
    return {
        "instock": "in_stock",
        "outofstock": "out_of_stock",
        "preorder": "preorder",
        "limitedavailability": "limited",
    }.get(token, token or "unknown")


def _product_from_json_ld(parser: _ProductHtmlParser, warnings: list[str]) -> dict[str, Any] | None:
    products: list[dict[str, Any]] = []
    for index, source in enumerate(parser.json_ld, start=1):
        try:
            payload = json.loads(source)
        except json.JSONDecodeError as exc:
            warnings.append(f"JSON-LD block {index} is invalid: {exc.msg}")
            continue
        products.extend(node for node in _iter_json_nodes(payload) if _is_product(node))
    if not products:
        return None

    product = products[0]
    offer = _first_mapping(product.get("offers"))
    brand = _text(product.get("brand"))
    specs: dict[str, Any] = {}
    properties = product.get("additionalProperty") or []
    if isinstance(properties, dict):
        properties = [properties]
    for prop in properties:
        if isinstance(prop, dict) and _text(prop.get("name")) and _text(prop.get("value")):
            specs[_text(prop["name"])] = _text(prop["value"])

    gtin = next(
        (_text(product.get(key)) for key in ("gtin14", "gtin13", "gtin12", "gtin8", "gtin") if product.get(key)),
        None,
    )
    return {
        "title": _text(product.get("name")),
        "url": _text(offer.get("url")) or _text(product.get("url")),
        "price_int": _price(offer.get("price") or offer.get("lowPrice")),
        "currency": _text(offer.get("priceCurrency")) or "RUB",
        "availability": _availability(offer.get("availability")),
        "brand": brand,
        "model": _text(product.get("model")),
        "mpn": _text(product.get("mpn") or product.get("sku")),
        "gtin": gtin,
        "specs": specs,
        "parse_method": "json-ld",
    }


def parse_dns_product_html(html: str) -> tuple[dict[str, Any], list[str]]:
    parser = _ProductHtmlParser()
    parser.feed(html)
    warnings: list[str] = []
    item = _product_from_json_ld(parser, warnings)
    if item is None:
        warnings.append("Product JSON-LD was not found; HTML fallback was used")
        item = {
            "title": next(iter(parser.h1_parts or parser.title_parts), None)
            or parser.meta.get("og:title"),
            "price_int": _price(
                parser.meta.get("price") or parser.meta.get("product:price:amount")
            ),
            "currency": parser.meta.get("pricecurrency")
            or parser.meta.get("product:price:currency")
            or "RUB",
            "availability": _availability(
                parser.meta.get("availability") or parser.meta.get("product:availability")
            ),
            "specs": dict(parser.specs),
            "parse_method": "html-fallback",
        }
    if not item.get("title"):
        warnings.append("product title was not found")
    if item.get("price_int") is None:
        warnings.append("product price was not found")
    return item, warnings


def _safe_snapshot_file(root: Path, relative_name: Any) -> Path:
    relative = PurePosixPath(str(relative_name or ""))
    if not relative.name or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe snapshot file path: {relative_name!r}")
    path = (root / Path(*relative.parts)).resolve()
    if root.resolve() not in path.parents or path.suffix.lower() not in {".html", ".htm"}:
        raise ValueError(f"unsupported snapshot file path: {relative_name!r}")
    return path


def build_catalog_from_dns_snapshot(snapshot_dir: Path) -> dict[str, Any]:
    root = Path(snapshot_dir).resolve()
    manifest_path = root / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("snapshot manifest schema_version must be 1")
    entries = manifest.get("items")
    if not isinstance(entries, list) or not entries:
        raise ValueError("snapshot manifest must contain a non-empty items list")

    snapshot: dict[str, list[dict[str, Any]]] = {}
    warnings: list[str] = []
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            warnings.append(f"manifest item {index} is not an object")
            continue
        source_path = _safe_snapshot_file(root, entry.get("file"))
        item, item_warnings = parse_dns_product_html(source_path.read_text(encoding="utf-8"))
        item.update(
            {
                "url": entry.get("url") or item.get("url"),
                "type": entry.get("category") or "components",
                "observed_at": entry.get("observed_at") or manifest.get("observed_at"),
                "region": entry.get("region") or manifest.get("region") or "",
                "raw_snapshot": entry.get("file"),
                "parse_warnings": item_warnings,
            }
        )
        snapshot.setdefault(str(item["type"]), []).append(item)
        warnings.extend(f"{entry.get('file')}: {warning}" for warning in item_warnings)

    normalized = normalize_dns_snapshot(snapshot, snapshot_name=manifest_path.name)
    deduplicated = deduplicate_items(normalized)
    return build_catalog_payload(
        items=deduplicated,
        sources=[
            CatalogSourceInfo(
                source="dns",
                snapshot_name=root.name,
                mode="offline-html-jsonld",
                items_before_dedup=len(normalized),
                items_after_dedup=len(deduplicated),
                warnings=warnings,
            )
        ],
        generated_by="tools.catalog_parser.sources.dns_snapshot",
    )
