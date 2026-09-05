"""Single-product capture from an ordinary browser without browser automation (P6)."""

from __future__ import annotations

import hashlib
import json
import re
import urllib.parse
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Mapping


class BrowserCaptureError(ValueError):
    pass


@dataclass(frozen=True)
class BrowserCaptureResult:
    item: dict[str, Any]
    source_context: dict[str, Any]
    warnings: tuple[str, ...]


class _StructuredPageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.meta: dict[str, str] = {}
        self.json_ld: list[str] = []
        self.title_parts: list[str] = []
        self._in_title = False
        self._in_ld = False
        self._script_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {str(key).lower(): str(value or "") for key, value in attrs}
        if tag.lower() == "meta":
            key = (
                attributes.get("property")
                or attributes.get("name")
                or attributes.get("itemprop")
            ).strip().lower()
            value = attributes.get("content", "").strip()
            if key and value:
                self.meta.setdefault(key, value)
        elif tag.lower() == "title":
            self._in_title = True
        elif tag.lower() == "script":
            script_type = attributes.get("type", "").lower()
            if "ld+json" in script_type:
                self._in_ld = True
                self._script_parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self._in_title = False
        elif tag.lower() == "script" and self._in_ld:
            self.json_ld.append("".join(self._script_parts).strip())
            self._in_ld = False
            self._script_parts = []

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)
        if self._in_ld:
            self._script_parts.append(data)


def capture_browser_content(
    content: str,
    *,
    source_url: str = "",
    region: str = "",
    category_override: str = "",
) -> BrowserCaptureResult:
    """Extract one Product from saved HTML or copied JSON-LD.

    The function never opens a URL.  The user visits the page in their normal
    browser and supplies saved HTML/page source/JSON-LD through the GUI or CLI.
    """

    text = str(content or "").strip()
    if not text:
        raise BrowserCaptureError("Буфер или файл пуст.")

    parser = _StructuredPageParser()
    product: dict[str, Any] = {}
    warnings: list[str] = []

    direct_json = _try_json(text)
    if direct_json is not None:
        product = _find_product(direct_json) or (
            dict(direct_json) if isinstance(direct_json, Mapping) else {}
        )
    else:
        try:
            parser.feed(text)
        except Exception as exc:
            raise BrowserCaptureError(f"HTML не удалось разобрать: {exc}") from exc
        for raw in parser.json_ld:
            payload = _try_json(_clean_json_ld(raw))
            if payload is None:
                continue
            product = _find_product(payload)
            if product:
                break

    source_url = _first_text(
        source_url,
        _value(product, "url"),
        parser.meta.get("og:url"),
    )
    title = _first_text(
        _value(product, "name"),
        parser.meta.get("og:title"),
        parser.meta.get("twitter:title"),
        " ".join(parser.title_parts).strip(),
    )
    if not title:
        raise BrowserCaptureError("Не найдено название Product/og:title.")

    offer = _extract_offer(product)
    price = _number(
        offer.get("price"),
        parser.meta.get("product:price:amount"),
        parser.meta.get("og:price:amount"),
        parser.meta.get("price"),
    )
    if price is None or price <= 0:
        raise BrowserCaptureError("Не найдена положительная цена Product/offer/meta.")
    currency = _first_text(
        offer.get("priceCurrency"),
        parser.meta.get("product:price:currency"),
        parser.meta.get("og:price:currency"),
        "RUB",
    ).upper()
    if currency in {"RUR", "₽", "РУБ", "РУБ."}:
        currency = "RUB"

    identity = _extract_identity(product)
    category = str(category_override or _value(product, "category") or "").strip()
    attributes = _extract_additional_properties(product)
    availability = _normalize_availability(_first_text(offer.get("availability"), "unknown"))
    observed_at = datetime.now(UTC).isoformat()
    host = urllib.parse.urlparse(source_url).netloc.lower().removeprefix("www.")
    stable_key = source_url or f"{title}|{identity.get('mpn') or identity.get('sku') or ''}"
    digest = hashlib.sha256(stable_key.encode("utf-8")).hexdigest()[:12]
    source_id = f"browser-capture-{_slug(host or 'page')}-{digest}"
    source_name = host or "Browser capture"

    if not product:
        warnings.append("JSON-LD Product не найден; использованы meta/title поля HTML.")
    if not identity:
        warnings.append("Идентификаторы товара не найдены; проверьте модель вручную.")
    if not source_url:
        warnings.append("URL страницы не указан; provenance ограничен локальным захватом.")

    item = {
        "title": title,
        "category": category,
        "source_product_id": _first_text(
            _value(product, "sku"),
            _value(product, "productID"),
            _value(product, "mpn"),
            digest,
        ),
        "identity": identity,
        "offer": {
            "price": price,
            "currency": currency,
            "availability": availability,
            "url": source_url or None,
            "observed_at": observed_at,
            "price_kind": "retail_offer",
        },
        "attributes": attributes,
        "parser_metadata": {
            "parse_source": "browser-jsonld" if product else "browser-meta",
            "confidence": 0.92 if product and identity else 0.72 if product else 0.55,
            "parse_warnings": warnings,
        },
        "review": {"warnings": warnings},
    }
    context = {
        "id": source_id,
        "name": source_name,
        "location": source_url or "local-browser-capture",
        "format": "browser_capture",
        "region": str(region or "").strip(),
        "price_kind": "retail_offer",
        "observed_at": observed_at,
        "source_type": "browser_capture",
        "capture_method": "ordinary_browser",
    }
    return BrowserCaptureResult(item=item, source_context=context, warnings=tuple(warnings))


def capture_browser_file(
    path: str | Path,
    *,
    source_url: str = "",
    region: str = "",
    category_override: str = "",
) -> BrowserCaptureResult:
    source = Path(path)
    if not source.is_file():
        raise BrowserCaptureError(f"Файл захвата не найден: {source}")
    raw = source.read_bytes()
    text = ""
    for encoding in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if not text:
        text = raw.decode("utf-8", errors="replace")
    return capture_browser_content(
        text,
        source_url=source_url,
        region=region,
        category_override=category_override,
    )


def _find_product(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        item_type = value.get("@type")
        types = item_type if isinstance(item_type, list) else [item_type]
        if any(str(kind or "").lower().endswith("product") for kind in types):
            return dict(value)
        graph = value.get("@graph")
        found = _find_product(graph)
        if found:
            return found
        for child in value.values():
            found = _find_product(child)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_product(child)
            if found:
                return found
    return {}


def _extract_offer(product: Mapping[str, Any]) -> dict[str, Any]:
    offers = product.get("offers")
    if isinstance(offers, Mapping):
        return dict(offers)
    if isinstance(offers, list):
        for offer in offers:
            if isinstance(offer, Mapping) and _number(offer.get("price"), offer.get("lowPrice")):
                return dict(offer)
    aggregate = product.get("aggregateOffer")
    return dict(aggregate) if isinstance(aggregate, Mapping) else {}


def _extract_identity(product: Mapping[str, Any]) -> dict[str, str]:
    brand = product.get("brand")
    if isinstance(brand, Mapping):
        brand = brand.get("name")
    result = {
        "brand": _first_text(brand),
        "model": _first_text(_value(product, "model")),
        "mpn": _first_text(_value(product, "mpn"), _value(product, "sku")),
        "gtin": _first_text(
            _value(product, "gtin14"),
            _value(product, "gtin13"),
            _value(product, "gtin12"),
            _value(product, "gtin8"),
            _value(product, "gtin"),
        ),
    }
    return {key: value for key, value in result.items() if value}


def _extract_additional_properties(product: Mapping[str, Any]) -> dict[str, Any]:
    properties = product.get("additionalProperty")
    if not isinstance(properties, list):
        return {}
    raw: dict[str, str] = {}
    for entry in properties:
        if not isinstance(entry, Mapping):
            continue
        name = _first_text(entry.get("name"), entry.get("propertyID")).lower()
        value = _first_text(entry.get("value"))
        if name and value:
            raw[name] = value
    result: dict[str, Any] = {}
    for name, value in raw.items():
        number = _number(value)
        if number is None:
            continue
        if "ram" in name or "оператив" in name:
            result.setdefault("ram_gb", number)
        elif "core" in name or "ядер" in name:
            result.setdefault("cpu_cores", number)
        elif "storage" in name or "накоп" in name or "ssd" in name:
            result.setdefault("storage_gb", number)
        elif "power" in name or "мощност" in name:
            result.setdefault("max_power_watts", number)
    return result


def _normalize_availability(value: str) -> str:
    text = str(value or "").lower()
    if any(marker in text for marker in ("instock", "in_stock", "available", "налич")):
        return "in_stock"
    if any(marker in text for marker in ("outofstock", "out_of_stock", "soldout")):
        return "out_of_stock"
    return "unknown"


def _clean_json_ld(value: str) -> str:
    return value.strip().removeprefix("<!--").removesuffix("-->").strip()


def _try_json(value: str) -> Any | None:
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None


def _value(mapping: Mapping[str, Any], key: str) -> Any:
    return mapping.get(key) if isinstance(mapping, Mapping) else None


def _number(*values: Any) -> float | None:
    for value in values:
        if value is None or isinstance(value, bool):
            continue
        text = str(value).strip().replace("\xa0", " ")
        text = re.sub(r"[^0-9,.-]", "", text).replace(",", ".")
        if not text:
            continue
        try:
            number = float(text)
        except ValueError:
            continue
        if number > 0:
            return number
    return None


def _first_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _slug(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9а-яА-Я]+", "-", str(value or "").lower()).strip("-")
    return re.sub(r"-+", "-", text) or "page"


__all__ = [
    "BrowserCaptureError",
    "BrowserCaptureResult",
    "capture_browser_content",
    "capture_browser_file",
]
