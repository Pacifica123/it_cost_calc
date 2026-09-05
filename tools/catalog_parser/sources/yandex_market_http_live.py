from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha1
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable
from urllib.parse import urljoin, urlparse, urlunparse

from ..http_session import CatalogHttpRequestError, CatalogHttpResponse, CatalogHttpSession
YANDEX_MARKET_BASE_URL = "https://market.yandex.ru"
YANDEX_MARKET_CATEGORIES = {
    "routers": "Роутеры",
    "switches": "Коммутаторы",
    "prebuilt_pcs": "Готовые компьютеры",
    "servers": "Серверные компьютеры",
}
YANDEX_MARKET_CATEGORY_URLS = {
    "routers": "https://market.yandex.ru/category/routery",
    "switches": "https://market.yandex.ru/category/kommutatory",
    "prebuilt_pcs": "https://market.yandex.ru/category/gotovyye-kompyutery",
    "servers": "https://market.yandex.ru/category/servernyye-kompyutery",
}
from .yandex_market_snapshot import build_catalog_from_yandex_market_snapshot



class _MarketListingParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.items: list[dict[str, str]] = []
        self._href: str | None = None
        self._title = ""
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        attributes = {str(key).lower(): str(value or "") for key, value in attrs}
        href = attributes.get("href", "")
        path = urlparse(urljoin(YANDEX_MARKET_BASE_URL, href)).path
        if path.startswith("/card/") or path.startswith("/product--"):
            self._href = href
            self._title = attributes.get("title", "") or attributes.get("aria-label", "")
            self._parts = []

    def handle_data(self, data: str) -> None:
        if self._href:
            self._parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or not self._href:
            return
        title = " ".join((self._title or "".join(self._parts)).split())
        self.items.append({"url": self._href, "title": title})
        self._href = None
        self._title = ""
        self._parts = []


def _canonical_market_url(value: str) -> str | None:
    parsed = urlparse(urljoin(YANDEX_MARKET_BASE_URL, value))
    hostname = str(parsed.hostname or "").lower()
    if parsed.scheme != "https" or not (
        hostname == "market.yandex.ru" or hostname.endswith(".market.yandex.ru")
    ):
        return None
    if not (parsed.path.startswith("/card/") or parsed.path.startswith("/product--")):
        return None
    return urlunparse(("https", "market.yandex.ru", parsed.path.rstrip("/"), "", "", ""))


def parse_yandex_market_listing_html(html: str, *, limit: int) -> list[dict[str, str]]:
    parser = _MarketListingParser()
    parser.feed(html)
    parser.close()
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in parser.items:
        url = _canonical_market_url(item["url"])
        if not url or url in seen:
            continue
        seen.add(url)
        result.append({"url": url, "title": item["title"]})
        if len(result) >= limit:
            break
    return result


_CHALLENGE_MARKERS = (
    "showcaptcha",
    "smartcaptcha",
    "captcha.yandex",
    "подтвердите, что запросы отправляли вы",
)


class YandexMarketHttpCollectionError(RuntimeError):
    exit_code = 4

    def __init__(self, message: str, *, manifest_path: Path) -> None:
        super().__init__(message)
        self.manifest_path = manifest_path


class YandexMarketHttpAccessDeniedError(YandexMarketHttpCollectionError):
    exit_code = 3


@dataclass(frozen=True, slots=True)
class YandexMarketHttpLiveOptions:
    snapshot_dir: Path
    categories: tuple[str, ...] = ("routers", "switches", "prebuilt_pcs", "servers")
    per_category_limit: int = 10
    time_limit_seconds: int = 300
    request_delay_seconds: float = 0.7
    region: str = ""

    def validate(self) -> None:
        unknown = sorted(set(self.categories) - set(YANDEX_MARKET_CATEGORIES))
        if unknown:
            raise ValueError(f"Неизвестные категории Яндекс Маркета: {', '.join(unknown)}")
        if not self.categories:
            raise ValueError("Нужно выбрать хотя бы одну категорию Яндекс Маркета")
        if not 1 <= self.per_category_limit <= 50:
            raise ValueError("Лимит карточек на категорию должен быть от 1 до 50")
        if not 30 <= self.time_limit_seconds <= 1800:
            raise ValueError("Общий таймаут должен быть от 30 до 1800 секунд")


def _access_failure(response: CatalogHttpResponse) -> dict[str, object] | None:
    normalized = response.text.lower()
    final_url = response.final_url.lower()
    captcha = any(marker in normalized or marker in final_url for marker in _CHALLENGE_MARKERS)
    status = response.status_code if response.status_code in {401, 403, 429} else None
    if not captcha and status is None:
        return None
    if captcha:
        message = "Яндекс Маркет вернул CAPTCHA для HTTP-сессии."
        stage = "challenge"
    elif status == 429:
        message = "Яндекс Маркет ограничил HTTP-запросы (429)."
        stage = "rate_limited"
    else:
        message = f"Яндекс Маркет отклонил HTTP-сессию (HTTP {status})."
        stage = "access_denied"
    return {
        "kind": "access_denied",
        "status_code": status,
        "stage": stage,
        "requested_url": response.requested_url,
        "final_url": response.final_url,
        "message": message,
    }


def _product_filename(category: str, index: int, url: str) -> str:
    digest = sha1(url.encode("utf-8")).hexdigest()[:10]
    return f"products/{category}_{index:03d}_{digest}.html"


def build_catalog_from_http_yandex_market(
    options: YandexMarketHttpLiveOptions,
    *,
    progress: Callable[[str], None] = print,
    session_factory: Callable[..., CatalogHttpSession] = CatalogHttpSession,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict:
    options.validate()
    root = Path(options.snapshot_dir)
    listings_dir = root / "listings"
    products_dir = root / "products"
    listings_dir.mkdir(parents=True, exist_ok=True)
    products_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "snapshot_manifest.json"
    observed_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    started = monotonic()
    warnings: list[str] = []
    manifest_items: list[dict[str, str]] = []
    failure: dict[str, object] | None = None

    def write_manifest(status: str) -> None:
        manifest = {
            "schema_version": 1,
            "source": "yandex_market",
            "region": options.region,
            "observed_at": observed_at,
            "capture": {
                "mode": "user-initiated-http",
                "status": status,
                "transport": "requests-session",
                "categories": list(options.categories),
                "per_category_limit": options.per_category_limit,
                "warnings": warnings,
                **({"failure": failure} if failure else {}),
            },
            "items": manifest_items,
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    try:
        with session_factory(timeout_seconds=min(30.0, float(options.time_limit_seconds))) as session:
            progress("HTTP: подготавливаю сессию Яндекс Маркета.")
            warm = session.get(f"{YANDEX_MARKET_BASE_URL}/")
            (root / "warmup.html").write_text(warm.text, encoding="utf-8")
            access = _access_failure(warm)
            if access:
                failure = access
                write_manifest("failed")
                raise YandexMarketHttpAccessDeniedError(
                    str(access["message"]), manifest_path=manifest_path
                )

            for category in options.categories:
                if monotonic() - started >= options.time_limit_seconds:
                    warnings.append("Общий таймаут достигнут до обработки всех категорий")
                    break
                category_url = YANDEX_MARKET_CATEGORY_URLS[category]
                progress(f"Маркет HTTP: {YANDEX_MARKET_CATEGORIES[category]}")
                listing = session.get(category_url, referer=f"{YANDEX_MARKET_BASE_URL}/")
                (listings_dir / f"{category}.html").write_text(listing.text, encoding="utf-8")
                access = _access_failure(listing)
                if access:
                    failure = access
                    write_manifest("partial" if manifest_items else "failed")
                    if manifest_items:
                        warnings.append(str(access["message"]))
                        break
                    raise YandexMarketHttpAccessDeniedError(
                        str(access["message"]), manifest_path=manifest_path
                    )

                products = parse_yandex_market_listing_html(
                    listing.text, limit=options.per_category_limit
                )
                progress(f"{category}: найдено ссылок {len(products)}")
                if not products:
                    warnings.append(f"{category}: ссылки на карточки не найдены")
                    continue
                for index, product in enumerate(products, start=1):
                    if monotonic() - started >= options.time_limit_seconds:
                        warnings.append(f"{category}: остановлено по таймауту")
                        break
                    url = product["url"]
                    page = session.get(url, referer=category_url)
                    relative_file = _product_filename(category, index, url)
                    (root / relative_file).write_text(page.text, encoding="utf-8")
                    access = _access_failure(page)
                    if access:
                        failure = access
                        write_manifest("partial" if manifest_items else "failed")
                        if manifest_items:
                            warnings.append(str(access["message"]))
                            break
                        raise YandexMarketHttpAccessDeniedError(
                            str(access["message"]), manifest_path=manifest_path
                        )
                    manifest_items.append(
                        {
                            "file": relative_file,
                            "category": category,
                            "url": page.final_url or url,
                            "observed_at": observed_at,
                        }
                    )
                    progress(f"{category}: карточка {index}/{len(products)}")
                    if options.request_delay_seconds:
                        sleep(options.request_delay_seconds)
                if failure:
                    break
    except CatalogHttpRequestError as exc:
        failure = {
            "kind": "network_error",
            "status_code": None,
            "stage": "http_request",
            "message": str(exc),
        }
        write_manifest("partial" if manifest_items else "failed")
        if not manifest_items:
            raise YandexMarketHttpCollectionError(str(exc), manifest_path=manifest_path) from exc
        warnings.append(str(exc))

    if not manifest_items:
        if failure is None:
            failure = {
                "kind": "no_products",
                "status_code": None,
                "stage": "listing_parse",
                "message": "Яндекс Маркет ответил, но ссылки на товары не найдены.",
            }
        write_manifest("failed")
        raise YandexMarketHttpCollectionError(
            str(failure["message"]), manifest_path=manifest_path
        )

    write_manifest("partial" if failure else "completed")
    payload = build_catalog_from_yandex_market_snapshot(root)
    if int((payload.get("stats") or {}).get("items_total") or 0) <= 0:
        raise YandexMarketHttpCollectionError(
            "HTTP-снимок Маркета сохранён, но пригодные товары не распознаны.",
            manifest_path=manifest_path,
        )
    payload["generated_by"] = "tools.catalog_parser.sources.yandex_market_http_live"
    for source in payload.get("sources", []):
        source["mode"] = "http-get+offline-html-jsonld+embedded-json"
        source["warnings"] = list(source.get("warnings") or []) + warnings
    progress(f"Маркет HTTP: готово товаров {(payload.get('stats') or {}).get('items_total', 0)}")
    progress(f"Диагностика: {manifest_path}")
    return payload
