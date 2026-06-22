from __future__ import annotations

import base64
import json
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

from ..catalog_builder import build_catalog_payload, deduplicate_items, normalize_dns_snapshot
from ..catalog_schema import CatalogSourceInfo

DNS_BASE_URL = "https://www.dns-shop.ru"
MAX_CAPTURE_BYTES = 256 * 1024 * 1024
MAX_RESPONSE_BODY_BYTES = 16 * 1024 * 1024
_CATEGORY_PATHS = {
    "/catalog/17a8aa1c16404e77/wi-fi-routery/": "routers",
    "/catalog/17a9dc3716404e77/kommutatory/": "switches",
    "/catalog/17a8932c16404e77/personalnye-komputery/": "prebuilt_pcs",
    "/catalog/17a8939816404e77/servery/": "servers",
}


class DnsCaptureError(ValueError):
    """A local DNS capture cannot be converted into a safe catalog."""


def _product_key(value: Any) -> str:
    return "".join(character for character in str(value or "").lower() if character.isalnum())


def _category_from_title(title: str) -> str | None:
    normalized = title.lower().replace("ё", "е")
    if "коммутатор" in normalized:
        return "switches"
    if "роутер" in normalized or "маршрутизатор" in normalized:
        return "routers"
    if normalized.startswith("сервер") or " сервер " in f" {normalized} ":
        return "servers"
    if normalized.startswith(("пк ", "мини пк ")) or "компьютер" in normalized:
        return "prebuilt_pcs"
    return None


def _category_from_path(path: str) -> str | None:
    return next((category for prefix, category in _CATEGORY_PATHS.items() if path.startswith(prefix)), None)


def _availability(value: Any) -> str:
    token = str(value or "").rsplit("/", 1)[-1].lower()
    return {
        "instock": "in_stock",
        "outofstock": "out_of_stock",
        "preorder": "preorder",
        "limitedavailability": "limited",
    }.get(token, token or "unknown")


def _integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return round(value)
    try:
        return round(float(str(value).replace(" ", "").replace(",", ".")))
    except (TypeError, ValueError):
        return None


def _merge_product(products: dict[str, dict[str, Any]], key: str, values: dict[str, Any]) -> None:
    if not key:
        return
    product = products.setdefault(key, {"specs": {}, "parse_warnings": []})
    incoming_specs = dict(values.pop("specs", {}) or {})
    product.setdefault("specs", {}).update(incoming_specs)
    incoming_warnings = list(values.pop("parse_warnings", []) or [])
    product.setdefault("parse_warnings", []).extend(incoming_warnings)
    for field, value in values.items():
        if value not in (None, "", [], {}):
            product[field] = value


class _DnsCaptureHtmlParser(HTMLParser):
    _VOID_TAGS = {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.canonical_url = ""
        self.category_cards: list[dict[str, Any]] = []
        self.product_card: dict[str, Any] | None = None
        self._category_card: dict[str, Any] | None = None
        self._category_card_depth: int | None = None
        self._category_name_depth: int | None = None
        self._category_name_parts: list[str] = []
        self._product_card_depth: int | None = None
        self._product_title_depth: int | None = None
        self._product_title_parts: list[str] = []
        self._product_code_depth: int | None = None
        self._product_code_parts: list[str] = []
        self._spec: dict[str, list[str]] | None = None
        self._spec_depth: int | None = None
        self._spec_label_depth: int | None = None
        self._spec_value_depth: int | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.depth += 1
        values = {str(key).lower(): str(value or "") for key, value in attrs}
        classes = set(values.get("class", "").split())
        if tag == "link" and "canonical" in values.get("rel", "").split():
            self.canonical_url = values.get("href", "")
        if "catalog-product" in classes and values.get("data-product"):
            self._category_card = {
                "key": _product_key(values["data-product"]),
                "title": "",
                "url": "",
                "specs": {},
            }
            self._category_card_depth = self.depth
        if "product-card" in classes and values.get("data-product-card"):
            self.product_card = {
                "key": _product_key(values["data-product-card"]),
                "title": "",
                "code": "",
                "specs": {},
            }
            self._product_card_depth = self.depth
        if self._category_card is not None and tag == "a" and "catalog-product__name" in classes:
            self._category_card["url"] = urljoin(DNS_BASE_URL, values.get("href", ""))
            full_title = " ".join(values.get("title", "").split())
            self._category_card["title"] = full_title.split("[", 1)[0].strip()
            self._category_name_depth = self.depth
            self._category_name_parts = []
        if self.product_card is not None and (
            "data-product-title" in values or "product-card-top__title" in classes
        ):
            self._product_title_depth = self.depth
            self._product_title_parts = []
        if self.product_card is not None and "product-card-top__code" in classes:
            self._product_code_depth = self.depth
            self._product_code_parts = []
        if self._category_card is not None and "catalog-product__spec" in classes:
            self._spec = {"label": [], "value": []}
            self._spec_depth = self.depth
        elif self.product_card is not None and "product-card-top__specs-item" in classes:
            self._spec = {"label": [], "value": []}
            self._spec_depth = self.depth
        if self._spec is not None and classes.intersection(
            {"catalog-product__spec-label", "product-card-top__specs-item-title"}
        ):
            self._spec_label_depth = self.depth
        if self._spec is not None and classes.intersection(
            {"catalog-product__spec-value", "product-card-top__specs-item-content"}
        ):
            self._spec_value_depth = self.depth
        if tag in self._VOID_TAGS:
            self.depth = max(0, self.depth - 1)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)

    def handle_data(self, data: str) -> None:
        if self._category_name_depth is not None:
            self._category_name_parts.append(data)
        if self._product_title_depth is not None:
            self._product_title_parts.append(data)
        if self._product_code_depth is not None:
            self._product_code_parts.append(data)
        if self._spec_label_depth is not None and self._spec is not None:
            self._spec["label"].append(data)
        if self._spec_value_depth is not None and self._spec is not None:
            self._spec["value"].append(data)

    def handle_endtag(self, _tag: str) -> None:
        if self.depth == self._category_name_depth:
            if self._category_card is not None and not self._category_card["title"]:
                full_title = " ".join("".join(self._category_name_parts).split())
                self._category_card["title"] = full_title.split("[", 1)[0].strip()
            self._category_name_depth = None
        if self.depth == self._product_title_depth:
            if self.product_card is not None:
                self.product_card["title"] = " ".join(
                    "".join(self._product_title_parts).split()
                )
            self._product_title_depth = None
        if self.depth == self._product_code_depth:
            if self.product_card is not None:
                self.product_card["code"] = "".join(
                    character
                    for character in "".join(self._product_code_parts)
                    if character.isdigit()
                )
            self._product_code_depth = None
        if self.depth == self._spec_label_depth:
            self._spec_label_depth = None
        if self.depth == self._spec_value_depth:
            self._spec_value_depth = None
        if self.depth == self._spec_depth and self._spec is not None:
            label = " ".join("".join(self._spec["label"]).split()).rstrip(":")
            value = " ".join("".join(self._spec["value"]).split())
            target = self._category_card or self.product_card
            if target is not None and label and value:
                target["specs"][label] = value
            self._spec = None
            self._spec_depth = None
        if self.depth == self._category_card_depth and self._category_card is not None:
            self.category_cards.append(self._category_card)
            self._category_card = None
            self._category_card_depth = None
        if self.depth == self._product_card_depth:
            self._product_card_depth = None
        self.depth = max(0, self.depth - 1)


def _parse_html(html: str) -> _DnsCaptureHtmlParser:
    parser = _DnsCaptureHtmlParser()
    parser.feed(html)
    parser.close()
    return parser


def _parse_category_html(
    html: str,
    *,
    category: str | None,
    observed_at: str,
    products: dict[str, dict[str, Any]],
) -> None:
    parser = _parse_html(html)
    for card in parser.category_cards:
        key = str(card.get("key") or "")
        if not key or not card.get("url"):
            continue
        item_category = category or _category_from_title(str(card.get("title") or ""))
        _merge_product(
            products,
            key,
            {
                "title": card.get("title"),
                "url": card.get("url"),
                "type": item_category,
                "specs": card.get("specs"),
                "observed_at": observed_at,
                "parse_method": "har-category-html+ajax-state",
            },
        )


def _parse_product_html(
    html: str,
    *,
    url: str,
    observed_at: str,
    products: dict[str, dict[str, Any]],
) -> None:
    parser = _parse_html(html)
    card = parser.product_card
    if not card:
        return
    key = str(card.get("key") or "")
    title = str(card.get("title") or "")
    code = str(card.get("code") or "")
    _merge_product(
        products,
        key,
        {
            "title": title,
            "url": url,
            "type": _category_from_title(title),
            "source_product_id": code or None,
            "specs": card.get("specs"),
            "observed_at": observed_at,
            "parse_method": "har-product-html",
        },
    )


def _parse_product_buy(payload: dict[str, Any], products: dict[str, dict[str, Any]]) -> None:
    states = ((payload.get("data") or {}).get("states") or [])
    for state in states:
        data = state.get("data") if isinstance(state, dict) else None
        if not isinstance(data, dict):
            continue
        price = data.get("price") if isinstance(data.get("price"), dict) else {}
        title = str(data.get("name") or "").strip()
        _merge_product(
            products,
            _product_key(data.get("id")),
            {
                "title": title,
                "type": _category_from_title(title),
                "price_int": _integer(price.get("current")),
                "availability": "out_of_stock" if data.get("notAvail") else "in_stock",
                "currency": "RUB",
            },
        )


def _parse_microdata(
    payload: dict[str, Any],
    *,
    key: str,
    observed_at: str,
    products: dict[str, dict[str, Any]],
) -> None:
    data = payload.get("data")
    if not isinstance(data, dict) or str(data.get("@type") or "").lower() != "product":
        return
    offer = data.get("offers") if isinstance(data.get("offers"), dict) else {}
    brand = data.get("brand") if isinstance(data.get("brand"), dict) else {}
    title = str(data.get("name") or "").strip()
    _merge_product(
        products,
        key,
        {
            "title": title,
            "url": offer.get("url"),
            "type": _category_from_title(title),
            "source_product_id": data.get("sku"),
            "price_int": _integer(offer.get("price")),
            "currency": offer.get("priceCurrency") or "RUB",
            "availability": _availability(offer.get("availability")),
            "brand": brand.get("name"),
            "observed_at": observed_at,
            "parse_method": "har-microdata+pwa",
        },
    )


def _flatten_characteristics(value: Any) -> dict[str, str]:
    specs: dict[str, str] = {}
    if not isinstance(value, dict):
        return specs
    for group, rows in value.items():
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            title = str(row.get("title") or "").strip()
            item_value = str(row.get("value") or "").strip()
            if title and item_value:
                if title not in specs:
                    specs[title] = item_value
                elif specs[title] != item_value:
                    specs[f"{group}: {title}"] = item_value
    return specs


def _parse_pwa_product(
    payload: dict[str, Any],
    *,
    observed_at: str,
    products: dict[str, dict[str, Any]],
) -> None:
    data = payload.get("data")
    if not isinstance(data, dict):
        return
    title = str(data.get("name") or "").strip()
    _merge_product(
        products,
        _product_key(data.get("guid")),
        {
            "title": title,
            "type": _category_from_title(title),
            "source_product_id": data.get("code"),
            "price_int": _integer(data.get("price")),
            "currency": "RUB",
            "specs": _flatten_characteristics(data.get("characteristics")),
            "observed_at": observed_at,
            "parse_method": "har-microdata+pwa",
        },
    )


def _response_text(entry: dict[str, Any]) -> str | None:
    response = entry.get("response")
    if not isinstance(response, dict) or not 200 <= int(response.get("status") or 0) < 300:
        return None
    content = response.get("content")
    if not isinstance(content, dict) or not isinstance(content.get("text"), str):
        return None
    text = content["text"]
    if content.get("encoding") == "base64":
        try:
            raw = base64.b64decode(text, validate=True)
        except (ValueError, TypeError) as exc:
            raise DnsCaptureError("HAR содержит некорректный base64 response body.") from exc
        if len(raw) > MAX_RESPONSE_BODY_BYTES:
            return None
        return raw.decode("utf-8", errors="replace")
    if len(text.encode("utf-8")) > MAX_RESPONSE_BODY_BYTES:
        return None
    return text


def _build_payload(
    products: dict[str, dict[str, Any]],
    *,
    source_name: str,
    mode: str,
    region: str,
    warnings: list[str],
) -> dict[str, Any]:
    snapshot: dict[str, list[dict[str, Any]]] = {}
    for product in products.values():
        title = str(product.get("title") or "").strip()
        category = str(product.get("type") or _category_from_title(title) or "")
        if not title or category not in {"routers", "switches", "prebuilt_pcs", "servers"}:
            continue
        product["type"] = category
        product["region"] = region
        product["raw_snapshot"] = source_name
        if product.get("price_int") is None:
            product.setdefault("parse_warnings", []).append(
                "Цена отсутствует в локальном capture"
            )
        if not product.get("specs"):
            product.setdefault("parse_warnings", []).append(
                "Характеристики отсутствуют в локальном capture"
            )
        snapshot.setdefault(category, []).append(product)
    normalized = normalize_dns_snapshot(snapshot, snapshot_name=source_name)
    deduplicated = deduplicate_items(normalized)
    if not deduplicated:
        raise DnsCaptureError(
            "В capture не найдены поддерживаемые карточки DNS с response body."
        )
    return build_catalog_payload(
        items=deduplicated,
        sources=[
            CatalogSourceInfo(
                source="dns",
                snapshot_name=source_name,
                mode=mode,
                items_before_dedup=len(normalized),
                items_after_dedup=len(deduplicated),
                warnings=warnings,
            )
        ],
        generated_by="tools.catalog_parser.sources.dns_capture",
    )


def build_catalog_from_dns_har(path: Path, *, region: str = "") -> dict[str, Any]:
    source = Path(path)
    if not source.is_file() or source.stat().st_size > MAX_CAPTURE_BYTES:
        raise DnsCaptureError("HAR не найден или превышает лимит 256 МБ.")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DnsCaptureError("Не удалось прочитать HAR как UTF-8 JSON.") from exc
    entries = ((payload.get("log") or {}).get("entries") or []) if isinstance(payload, dict) else []
    if not isinstance(entries, list):
        raise DnsCaptureError("HAR не содержит log.entries.")

    products: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    accepted_bodies = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        request = entry.get("request")
        if not isinstance(request, dict):
            continue
        parsed = urlparse(str(request.get("url") or ""))
        if parsed.scheme != "https" or parsed.hostname != "www.dns-shop.ru":
            continue
        text = _response_text(entry)
        if text is None:
            continue
        observed_at = str(entry.get("startedDateTime") or datetime.now(UTC).isoformat())
        path_value = parsed.path
        if path_value in _CATEGORY_PATHS or path_value == "/search/":
            _parse_category_html(
                text,
                category=_CATEGORY_PATHS.get(path_value),
                observed_at=observed_at,
                products=products,
            )
            accepted_bodies += 1
        elif path_value.startswith("/product/") and not path_value.startswith("/product/microdata/"):
            _parse_product_html(text, url=parsed.geturl(), observed_at=observed_at, products=products)
            accepted_bodies += 1
        elif path_value == "/ajax-state/product-buy/":
            try:
                _parse_product_buy(json.loads(text), products)
                accepted_bodies += 1
            except json.JSONDecodeError:
                warnings.append("Один product-buy response содержит некорректный JSON")
        elif path_value.startswith("/product/microdata/"):
            try:
                key = _product_key(path_value.removeprefix("/product/microdata/").split("/", 1)[0])
                _parse_microdata(
                    json.loads(text), key=key, observed_at=observed_at, products=products
                )
                accepted_bodies += 1
            except json.JSONDecodeError:
                warnings.append("Один microdata response содержит некорректный JSON")
        elif path_value == "/pwa/pwa/get-product/":
            try:
                _parse_pwa_product(json.loads(text), observed_at=observed_at, products=products)
                accepted_bodies += 1
            except json.JSONDecodeError:
                warnings.append("Один PWA product response содержит некорректный JSON")
    warnings.append(
        f"HAR обработан локально: entries={len(entries)}, "
        f"разрешённых response bodies={accepted_bodies}; "
        "headers, cookies, postData и отзывы проигнорированы"
    )
    return _build_payload(
        products,
        source_name=source.name,
        mode="browser-har-offline",
        region=region,
        warnings=warnings,
    )


def build_catalog_from_dns_html(path: Path, *, region: str = "") -> dict[str, Any]:
    source = Path(path)
    if not source.is_file() or source.stat().st_size > MAX_RESPONSE_BODY_BYTES:
        raise DnsCaptureError("HTML не найден или превышает лимит 16 МБ.")
    try:
        html = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise DnsCaptureError("Не удалось прочитать HTML как UTF-8.") from exc
    products: dict[str, dict[str, Any]] = {}
    observed_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    parser = _parse_html(html)
    url = parser.canonical_url
    _parse_product_html(html, url=url, observed_at=observed_at, products=products)
    category = _category_from_path(urlparse(url).path)
    if category:
        _parse_category_html(
            html,
            category=category,
            observed_at=observed_at,
            products=products,
        )
    return _build_payload(
        products,
        source_name=source.name,
        mode="browser-saved-html-offline",
        region=region,
        warnings=[
            "HTML обработан локально; динамические цена и характеристики могут отсутствовать"
        ],
    )
