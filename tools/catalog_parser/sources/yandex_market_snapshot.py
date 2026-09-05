from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from urllib.parse import urlparse

from ..catalog_builder import (
    build_catalog_payload,
    deduplicate_items,
    normalize_yandex_market_snapshot,
)
from ..catalog_schema import CatalogSourceInfo

MANIFEST_NAME = "snapshot_manifest.json"
_PRICE_RE = re.compile(r"(?:^|\D)(\d[\d\s\u00a0\u2009\u202f]*(?:[.,]\d+)?)\s*(?:₽|руб)", re.I)
_CARD_ID_RE = re.compile(r"/(?:card/[^/?#]+|product--[^/?#]+)/(?P<id>\d+)")
_KNOWN_SPEC_LABELS = {
    "артикул маркета",
    "бренд",
    "модель",
    "процессор",
    "количество ядер процессора",
    "частота процессора",
    "оперативная память",
    "тип памяти",
    "видеокарта",
    "объем накопителя",
    "конфигурация накопителей",
    "мощность блока питания",
    "максимальная потребляемая мощность",
    "тип устройства",
    "количество lan-портов",
    "количество wan-портов",
    "базовая скорость передачи данных",
    "скорость ethernet",
    "макс. скорость беспроводного соединения",
    "стандарт wi-fi 802.11",
    "частоты wi-fi",
    "тип управления коммутатора",
    "сетевые стандарты",
    "поддержка ipv6",
}


class _MarketHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.json_ld: list[str] = []
        self.embedded_json: list[str] = []
        self.meta: dict[str, str] = {}
        self.canonical_url = ""
        self.h1_parts: list[str] = []
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self._capture: str | None = None
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {str(key).lower(): str(value or "") for key, value in attrs}
        if tag == "script":
            script_type = attributes.get("type", "").lower()
            if script_type == "application/ld+json":
                self._capture, self._parts = "json_ld", []
            elif script_type in {"application/json", "application/x-json"}:
                self._capture, self._parts = "embedded_json", []
        elif tag in {"h1", "title"}:
            self._capture, self._parts = tag, []
        elif tag == "meta":
            key = attributes.get("itemprop") or attributes.get("property") or attributes.get("name")
            value = attributes.get("content")
            if key and value:
                self.meta[key.lower()] = value
        elif tag == "link" and "canonical" in attributes.get("rel", "").lower().split():
            self.canonical_url = attributes.get("href", "")

    def handle_data(self, data: str) -> None:
        value = " ".join(data.split())
        if value:
            self.text_parts.append(value)
        if self._capture:
            self._parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        expected = "json_ld" if tag == "script" and self._capture == "json_ld" else tag
        if tag == "script" and self._capture == "embedded_json":
            expected = "embedded_json"
        if self._capture != expected:
            return
        value = "".join(self._parts).strip()
        if self._capture == "json_ld" and value:
            self.json_ld.append(value)
        elif self._capture == "embedded_json" and value:
            self.embedded_json.append(value)
        elif self._capture == "h1" and value:
            self.h1_parts.append(" ".join(value.split()))
        elif self._capture == "title" and value:
            self.title_parts.append(" ".join(value.split()))
        self._capture, self._parts = None, []


def _iter_json_nodes(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from _iter_json_nodes(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _iter_json_nodes(nested)


def _text(value: Any) -> str | None:
    if isinstance(value, dict):
        value = value.get("name") or value.get("value") or value.get("raw")
    if isinstance(value, list):
        values = [str(item).strip() for item in value if str(item).strip()]
        return ", ".join(values) or None
    result = str(value or "").strip()
    return result or None


def _integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return round(value)
    normalized = re.sub(r"[\s\u00a0\u2009\u202f]", "", str(value or ""))
    match = re.search(r"\d+(?:[.,]\d+)?", normalized)
    if not match:
        return None
    try:
        return round(float(match.group().replace(",", ".")))
    except ValueError:
        return None


def _availability(value: Any) -> str:
    token = str(value or "").rsplit("/", 1)[-1].lower()
    return {
        "instock": "in_stock",
        "outofstock": "out_of_stock",
        "preorder": "preorder",
        "limitedavailability": "limited",
    }.get(token, token or "unknown")


def _first_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return next((item for item in value if isinstance(item, dict)), {})
    return {}


def _specs_from_value(value: Any) -> dict[str, str]:
    specs: dict[str, str] = {}
    if isinstance(value, dict):
        direct_scalars = all(not isinstance(item, (dict, list)) for item in value.values())
        if direct_scalars:
            for key, item in value.items():
                label = str(key).strip()
                item_text = _text(item)
                if label and item_text:
                    specs[label] = item_text
        for nested in value.values():
            specs.update(_specs_from_value(nested))
    elif isinstance(value, list):
        for row in value:
            if isinstance(row, dict):
                label = _text(row.get("name") or row.get("title") or row.get("label"))
                item_text = _text(
                    row.get("value")
                    or row.get("values")
                    or row.get("text")
                    or row.get("description")
                )
                if label and item_text and label.lower() != item_text.lower():
                    specs[label] = item_text
                specs.update(_specs_from_value(row.get("items") or row.get("features") or []))
    return specs


def _json_ld_product(parser: _MarketHtmlParser, warnings: list[str]) -> dict[str, Any] | None:
    products: list[dict[str, Any]] = []
    for index, source in enumerate(parser.json_ld, start=1):
        try:
            payload = json.loads(source)
        except json.JSONDecodeError as exc:
            warnings.append(f"JSON-LD block {index} is invalid: {exc.msg}")
            continue
        for node in _iter_json_nodes(payload):
            node_type = node.get("@type")
            types = [node_type] if isinstance(node_type, str) else node_type or []
            if any(str(value).lower() == "product" for value in types):
                products.append(node)
    if not products:
        return None
    product = products[0]
    offer = _first_mapping(product.get("offers"))
    specs = _specs_from_value(product.get("additionalProperty") or [])
    gtin = next(
        (
            _text(product.get(key))
            for key in ("gtin14", "gtin13", "gtin12", "gtin8", "gtin")
            if product.get(key)
        ),
        None,
    )
    return {
        "title": _text(product.get("name")),
        "url": _text(offer.get("url")) or _text(product.get("url")),
        "price_int": _integer(offer.get("price") or offer.get("lowPrice")),
        "currency": _text(offer.get("priceCurrency")) or "RUB",
        "availability": _availability(offer.get("availability")),
        "brand": _text(product.get("brand")),
        "model": _text(product.get("model")),
        "mpn": _text(product.get("mpn")),
        "gtin": gtin,
        "source_product_id": _text(product.get("sku") or product.get("productID")),
        "specs": specs,
        "parse_method": "json-ld",
    }


def _candidate_score(node: dict[str, Any], product_id: str | None) -> int:
    title = _text(node.get("title") or node.get("name") or node.get("raw"))
    if not title or len(title) < 4:
        return -1
    score = 1
    identifiers = {
        str(node.get(key) or "")
        for key in ("id", "modelId", "productId", "wareId", "sku")
    }
    if product_id and product_id in identifiers:
        score += 10
    if any(key in node for key in ("price", "prices", "offers", "offer")):
        score += 3
    if any(key in node for key in ("specs", "characteristics", "parameters", "features")):
        score += 4
    if any(key in node for key in ("modelId", "productId", "wareId")):
        score += 2
    return score


def _find_price(value: Any) -> tuple[int | None, str]:
    preferred_keys = (
        "value",
        "current",
        "currentPrice",
        "discountedPrice",
        "min",
        "price",
    )
    if isinstance(value, dict):
        currency = str(value.get("currency") or value.get("currencyId") or "RUB").upper()
        for key in preferred_keys:
            if key in value:
                price = _integer(value[key])
                if price is not None:
                    return price, currency
        for key in ("price", "prices", "offers", "offer"):
            if key in value:
                price, nested_currency = _find_price(value[key])
                if price is not None:
                    return price, nested_currency
    elif isinstance(value, list):
        for item in value:
            price, currency = _find_price(item)
            if price is not None:
                return price, currency
    else:
        return _integer(value), "RUB"
    return None, "RUB"


def _embedded_product(parser: _MarketHtmlParser, product_id: str | None) -> dict[str, Any] | None:
    best: tuple[int, dict[str, Any]] | None = None
    for source in parser.embedded_json:
        try:
            payload = json.loads(source)
        except json.JSONDecodeError:
            continue
        for node in _iter_json_nodes(payload):
            score = _candidate_score(node, product_id)
            if score >= 5 and (best is None or score > best[0]):
                best = (score, node)
    if best is None:
        return None
    node = best[1]
    price, currency = _find_price(node)
    specs: dict[str, str] = {}
    for key in ("specs", "characteristics", "parameters", "features"):
        specs.update(_specs_from_value(node.get(key)))
    return {
        "title": _text(node.get("title") or node.get("name") or node.get("raw")),
        "price_int": price,
        "currency": currency,
        "brand": _text(node.get("brand") or node.get("vendor")),
        "model": _text(node.get("model")),
        "source_product_id": _text(
            node.get("modelId") or node.get("productId") or node.get("wareId") or node.get("id")
        ),
        "specs": specs,
        "parse_method": "embedded-json-best-effort",
    }


def _visible_specs(parts: list[str]) -> dict[str, str]:
    specs: dict[str, str] = {}
    for index, part in enumerate(parts[:-1]):
        label = part.strip().rstrip(":")
        if label.lower() not in _KNOWN_SPEC_LABELS:
            continue
        value = parts[index + 1].strip()
        if value and value.lower() not in _KNOWN_SPEC_LABELS and len(value) <= 500:
            specs[label] = value
    return specs


def _merge_missing(target: dict[str, Any], fallback: dict[str, Any] | None) -> None:
    if not fallback:
        return
    target_specs = target.setdefault("specs", {})
    for key, value in dict(fallback.get("specs") or {}).items():
        target_specs.setdefault(key, value)
    for key, value in fallback.items():
        if key == "specs":
            continue
        if target.get(key) in (None, "", [], {}):
            target[key] = value


def parse_yandex_market_product_html(
    html: str,
    *,
    page_url: str = "",
) -> tuple[dict[str, Any], list[str]]:
    parser = _MarketHtmlParser()
    parser.feed(html)
    parser.close()
    warnings: list[str] = []
    canonical_url = parser.canonical_url or page_url
    match = _CARD_ID_RE.search(urlparse(canonical_url).path)
    product_id = match.group("id") if match else None
    item = _json_ld_product(parser, warnings) or {}
    if not item:
        warnings.append("Product JSON-LD was not found; Yandex Market fallbacks were used")
    _merge_missing(item, _embedded_product(parser, product_id))
    visible_specs = _visible_specs(parser.text_parts)
    item.setdefault("specs", {}).update(
        {key: value for key, value in visible_specs.items() if key not in item["specs"]}
    )
    title = (
        next(iter(parser.h1_parts), None)
        or parser.meta.get("og:title")
        or next(iter(parser.title_parts), None)
    )
    if title and not item.get("title"):
        item["title"] = title.split(" — купить", 1)[0].strip()
    item["url"] = canonical_url or item.get("url")
    item["source_product_id"] = item.get("source_product_id") or product_id
    if item.get("price_int") is None:
        meta_price = parser.meta.get("product:price:amount") or parser.meta.get("price")
        item["price_int"] = _integer(meta_price)
    if item.get("price_int") is None:
        match = _PRICE_RE.search(" ".join(parser.text_parts))
        item["price_int"] = _integer(match.group(1)) if match else None
    item["currency"] = str(
        item.get("currency")
        or parser.meta.get("product:price:currency")
        or parser.meta.get("pricecurrency")
        or "RUB"
    ).upper()
    item.setdefault("availability", "unknown")
    item.setdefault("parse_method", "html-visible-best-effort")
    if not item.get("title"):
        warnings.append("product title was not found")
    if item.get("price_int") is None:
        warnings.append("product price was not found")
    if not item.get("specs"):
        warnings.append("product characteristics were not found")
    return item, warnings


_GENERIC_MARKET_TITLES = {
    "маркет",
    "яндекс маркет",
    "яндекс маркет для бизнеса",
    "маркет для бизнеса",
}


def yandex_market_product_quality_warnings(item: dict[str, Any]) -> list[str]:
    """Return reasons why a parsed page does not look like a product card.

    A browser can successfully load a generic Market shell, redirect or
    protection page with HTTP 200.  The HTML parser is intentionally tolerant,
    so the live pipeline needs a small quality gate before such pages are
    promoted into catalog items.
    """

    title = " ".join(str(item.get("title") or "").split())
    normalized_title = title.lower().replace("ё", "е").strip(" -—|·")
    warnings: list[str] = []
    if not title or normalized_title in _GENERIC_MARKET_TITLES:
        warnings.append("page does not contain a meaningful product title")

    has_product_payload = any(
        (
            item.get("price_int") is not None,
            bool(item.get("specs")),
            bool(item.get("brand")),
            bool(item.get("model")),
            bool(item.get("mpn")),
            bool(item.get("gtin")),
        )
    )
    if not has_product_payload:
        warnings.append("page does not contain price, characteristics, or product identity")
    return warnings


def _safe_snapshot_file(root: Path, relative_name: Any) -> Path:
    relative = PurePosixPath(str(relative_name or ""))
    if not relative.name or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe snapshot file path: {relative_name!r}")
    path = (root / Path(*relative.parts)).resolve()
    if root.resolve() not in path.parents or path.suffix.lower() not in {".html", ".htm"}:
        raise ValueError(f"unsupported snapshot file path: {relative_name!r}")
    return path


def build_catalog_from_yandex_market_snapshot(snapshot_dir: Path) -> dict[str, Any]:
    root = Path(snapshot_dir).resolve()
    manifest_path = root / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("snapshot manifest schema_version must be 1")
    if manifest.get("source") not in {"yandex_market", "market.yandex.ru"}:
        raise ValueError("snapshot manifest source must be yandex_market")
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
        item, item_warnings = parse_yandex_market_product_html(
            source_path.read_text(encoding="utf-8"),
            page_url=str(entry.get("url") or ""),
        )
        quality_warnings = yandex_market_product_quality_warnings(item)
        if quality_warnings:
            warnings.extend(
                f"{entry.get('file')}: skipped low-quality product page: {warning}"
                for warning in quality_warnings
            )
            continue
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

    normalized = normalize_yandex_market_snapshot(snapshot, snapshot_name=manifest_path.name)
    deduplicated = deduplicate_items(normalized)
    return build_catalog_payload(
        items=deduplicated,
        sources=[
            CatalogSourceInfo(
                source="yandex_market",
                snapshot_name=root.name,
                mode="offline-html-jsonld+embedded-json",
                items_before_dedup=len(normalized),
                items_after_dedup=len(deduplicated),
                warnings=warnings,
            )
        ],
        generated_by="tools.catalog_parser.sources.yandex_market_snapshot",
    )
