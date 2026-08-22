from __future__ import annotations

import base64
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ..catalog_builder import (
    build_catalog_payload,
    deduplicate_items,
    normalize_yandex_market_snapshot,
)
from ..catalog_schema import CatalogSourceInfo
from .yandex_market_live import (
    YANDEX_MARKET_CATEGORY_URLS,
    parse_yandex_market_listing_html,
)
from .yandex_market_snapshot import parse_yandex_market_product_html

MAX_CAPTURE_BYTES = 256 * 1024 * 1024
MAX_RESPONSE_BODY_BYTES = 24 * 1024 * 1024
_CATEGORY_PATHS = {
    urlparse(url).path: category for category, url in YANDEX_MARKET_CATEGORY_URLS.items()
}


class YandexMarketCaptureError(ValueError):
    """A local Yandex Market capture cannot be converted into a safe catalog."""


def _category_from_title(title: str) -> str | None:
    normalized = title.lower().replace("ё", "е")
    if "коммутатор" in normalized or "switch" in normalized:
        return "switches"
    if "роутер" in normalized or "маршрутизатор" in normalized:
        return "routers"
    if normalized.startswith("сервер") or " сервер " in f" {normalized} ":
        return "servers"
    if any(token in normalized for token in ("компьютер", "системный блок", "готовый пк", "мини-пк")):
        return "prebuilt_pcs"
    return None


def _category_from_path(path: str) -> str | None:
    return next(
        (category for prefix, category in _CATEGORY_PATHS.items() if path.startswith(prefix)),
        None,
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
            raise YandexMarketCaptureError(
                "HAR содержит некорректный base64 response body."
            ) from exc
        if len(raw) > MAX_RESPONSE_BODY_BYTES:
            return None
        return raw.decode("utf-8", errors="replace")
    if len(text.encode("utf-8")) > MAX_RESPONSE_BODY_BYTES:
        return None
    return text


def _product_key(item: dict[str, Any]) -> str:
    return str(item.get("source_product_id") or item.get("url") or item.get("title") or "").strip()


def _merge_product(products: dict[str, dict[str, Any]], item: dict[str, Any]) -> None:
    key = _product_key(item)
    if not key:
        return
    product = products.setdefault(key, {"specs": {}, "parse_warnings": []})
    for name, value in dict(item.get("specs") or {}).items():
        product.setdefault("specs", {}).setdefault(name, value)
    product.setdefault("parse_warnings", []).extend(item.get("parse_warnings") or [])
    for field, value in item.items():
        if field not in {"specs", "parse_warnings"} and value not in (None, "", [], {}):
            product[field] = value


def _listing_items(
    html: str,
    *,
    category: str | None,
    observed_at: str,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for card in parse_yandex_market_listing_html(html, limit=50):
        title = str(card.get("title") or "").strip()
        if not title:
            continue
        result.append(
            {
                "title": title,
                "url": card["url"],
                "type": category or _category_from_title(title),
                "observed_at": observed_at,
                "parse_method": "har-listing-html",
                "parse_warnings": [
                    "Из listing HTML получена частичная карточка; цена и характеристики могут отсутствовать"
                ],
            }
        )
    return result


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
            product.setdefault("parse_warnings", []).append("Цена отсутствует в локальном capture")
        if not product.get("specs"):
            product.setdefault("parse_warnings", []).append(
                "Характеристики отсутствуют в локальном capture"
            )
        snapshot.setdefault(category, []).append(product)
    normalized = normalize_yandex_market_snapshot(snapshot, snapshot_name=source_name)
    deduplicated = deduplicate_items(normalized)
    if not deduplicated:
        raise YandexMarketCaptureError(
            "В capture не найдены поддерживаемые карточки Яндекс Маркета с response body."
        )
    return build_catalog_payload(
        items=deduplicated,
        sources=[
            CatalogSourceInfo(
                source="yandex_market",
                snapshot_name=source_name,
                mode=mode,
                items_before_dedup=len(normalized),
                items_after_dedup=len(deduplicated),
                warnings=warnings,
            )
        ],
        generated_by="tools.catalog_parser.sources.yandex_market_capture",
    )


def build_catalog_from_yandex_market_har(path: Path, *, region: str = "") -> dict[str, Any]:
    source = Path(path)
    if not source.is_file() or source.stat().st_size > MAX_CAPTURE_BYTES:
        raise YandexMarketCaptureError("HAR не найден или превышает лимит 256 МБ.")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise YandexMarketCaptureError("Не удалось прочитать HAR как UTF-8 JSON.") from exc
    entries = ((payload.get("log") or {}).get("entries") or []) if isinstance(payload, dict) else []
    if not isinstance(entries, list):
        raise YandexMarketCaptureError("HAR не содержит log.entries.")

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
        hostname = str(parsed.hostname or "").lower()
        if parsed.scheme != "https" or hostname != "market.yandex.ru":
            continue
        text = _response_text(entry)
        if text is None:
            continue
        response = entry.get("response") or {}
        content = response.get("content") if isinstance(response, dict) else {}
        mime = str((content or {}).get("mimeType") or "").lower()
        if "html" not in mime and not text.lstrip().lower().startswith(("<!doctype", "<html", "<div")):
            continue
        observed_at = str(entry.get("startedDateTime") or datetime.now(UTC).isoformat())
        category = _category_from_path(parsed.path)
        if parsed.path.startswith("/card/") or parsed.path.startswith("/product--"):
            item, item_warnings = parse_yandex_market_product_html(text, page_url=parsed.geturl())
            item.update(
                {
                    "type": _category_from_title(str(item.get("title") or "")),
                    "observed_at": observed_at,
                    "parse_warnings": item_warnings,
                }
            )
            _merge_product(products, item)
            accepted_bodies += 1
        elif category or parsed.path.startswith("/search"):
            for item in _listing_items(
                text,
                category=category,
                observed_at=observed_at,
            ):
                _merge_product(products, item)
            accepted_bodies += 1
    warnings.append(
        f"HAR обработан локально: entries={len(entries)}, "
        f"разрешённых HTML response bodies={accepted_bodies}; "
        "headers, cookies, postData, изображения и отзывы проигнорированы"
    )
    return _build_payload(
        products,
        source_name=source.name,
        mode="browser-har-offline",
        region=region,
        warnings=warnings,
    )


def build_catalog_from_yandex_market_html(path: Path, *, region: str = "") -> dict[str, Any]:
    source = Path(path)
    if not source.is_file() or source.stat().st_size > MAX_RESPONSE_BODY_BYTES:
        raise YandexMarketCaptureError("HTML не найден или превышает лимит 24 МБ.")
    try:
        html = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise YandexMarketCaptureError("Не удалось прочитать HTML как UTF-8.") from exc

    observed_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    item, item_warnings = parse_yandex_market_product_html(html)
    products: dict[str, dict[str, Any]] = {}
    if item.get("title"):
        item.update(
            {
                "type": _category_from_title(str(item.get("title") or "")),
                "observed_at": observed_at,
                "parse_warnings": item_warnings,
            }
        )
        _merge_product(products, item)
    for listing_item in _listing_items(html, category=None, observed_at=observed_at):
        _merge_product(products, listing_item)
    return _build_payload(
        products,
        source_name=source.name,
        mode="browser-saved-html-offline",
        region=region,
        warnings=[
            "HTML обработан локально; динамические цена и характеристики могут отсутствовать"
        ],
    )
