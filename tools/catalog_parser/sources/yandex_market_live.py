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

from ..market_browser import YandexMarketBrowserPage, YandexMarketBrowserSession
from .yandex_market_snapshot import build_catalog_from_yandex_market_snapshot

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


class YandexMarketLiveCollectionError(RuntimeError):
    exit_code = 4

    def __init__(self, message: str, *, manifest_path: Path) -> None:
        super().__init__(message)
        self.manifest_path = manifest_path


class YandexMarketAccessDeniedError(YandexMarketLiveCollectionError):
    exit_code = 3


@dataclass(frozen=True, slots=True)
class _CapturedPage:
    requested_url: str
    final_url: str
    status_code: int | None
    title: str
    html: str


@dataclass(frozen=True, slots=True)
class YandexMarketLiveOptions:
    snapshot_dir: Path
    profile_dir: Path
    categories: tuple[str, ...] = ("routers", "switches", "prebuilt_pcs", "servers")
    browser_engine: str = "firefox"
    per_category_limit: int = 10
    time_limit_seconds: int = 300
    headless: bool = False
    first_page_wait_seconds: float = 8.0
    request_delay_seconds: float = 1.0
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
        if self.browser_engine not in {"firefox", "chromium"}:
            raise ValueError("Движок браузера должен быть firefox или chromium")


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


def _captured_page(
    value: str | YandexMarketBrowserPage,
    *,
    requested_url: str,
) -> _CapturedPage:
    if isinstance(value, YandexMarketBrowserPage):
        return _CapturedPage(
            requested_url=value.requested_url,
            final_url=value.final_url,
            status_code=value.status_code,
            title=value.title,
            html=value.html,
        )
    return _CapturedPage(
        requested_url=requested_url,
        final_url=requested_url,
        status_code=None,
        title="",
        html=str(value),
    )


def _access_failure(page: _CapturedPage) -> dict[str, object] | None:
    normalized = page.html.lower()
    final_url = page.final_url.lower()
    captcha = any(
        marker in normalized or marker in final_url
        for marker in (
            "showcaptcha",
            "smartcaptcha",
            "captcha.yandex",
            "подтвердите, что запросы отправляли вы",
        )
    )
    status = page.status_code if page.status_code in {401, 403, 429} else None
    if not captcha and status is None:
        return None
    if captcha:
        stage = "challenge_not_completed"
        message = (
            "Яндекс Маркет показал CAPTCHA или защитную проверку. Завершите её в видимом "
            "браузере и повторите сбор либо используйте локальный HAR/HTML."
        )
    elif status == 429:
        stage = "rate_limited"
        message = "Яндекс Маркет ограничил частоту запросов (HTTP 429). Повторите сбор позже."
    else:
        stage = "access_denied"
        message = (
            f"Яндекс Маркет вернул HTTP {status} для текущей сессии. "
            "Сохранённый HTML оставлен для диагностики."
        )
    return {
        "kind": "access_denied",
        "status_code": status,
        "stage": stage,
        "requested_url": page.requested_url,
        "final_url": page.final_url,
        "page_title": page.title,
        "message": message,
    }


def _snapshot_filename(category: str, index: int, url: str) -> str:
    digest = sha1(url.encode("utf-8")).hexdigest()[:10]
    return f"products/{category}_{index:03d}_{digest}.html"


def capture_yandex_market_snapshot(
    options: YandexMarketLiveOptions,
    *,
    fetch: Callable[[str], str | YandexMarketBrowserPage],
    progress: Callable[[str], None] = print,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> Path:
    options.validate()
    root = Path(options.snapshot_dir)
    products_dir = root / "products"
    listings_dir = root / "listings"
    products_dir.mkdir(parents=True, exist_ok=True)
    listings_dir.mkdir(parents=True, exist_ok=True)
    started = monotonic()
    observed_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    manifest_items: list[dict[str, str]] = []
    warnings: list[str] = []
    manifest_path = root / "snapshot_manifest.json"

    def write_manifest(*, status: str, failure: dict[str, object] | None = None) -> Path:
        capture: dict[str, object] = {
            "mode": "user-initiated-playwright",
            "status": status,
            "browser_engine": options.browser_engine,
            "categories": list(options.categories),
            "per_category_limit": options.per_category_limit,
            "category_sources": {
                category: {
                    "mode": "category-url",
                    "url": YANDEX_MARKET_CATEGORY_URLS[category],
                }
                for category in options.categories
            },
            "warnings": warnings,
        }
        if failure:
            capture["failure"] = failure
        manifest = {
            "schema_version": 1,
            "source": "yandex_market",
            "region": options.region,
            "observed_at": observed_at,
            "capture": capture,
            "items": manifest_items,
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return manifest_path

    for category in options.categories:
        if monotonic() - started >= options.time_limit_seconds:
            warnings.append("Общий таймаут достигнут до обработки всех категорий")
            break
        source_url = YANDEX_MARKET_CATEGORY_URLS[category]
        progress(f"Категория {category}: открываю '{YANDEX_MARKET_CATEGORIES[category]}'")
        listing_page = _captured_page(fetch(source_url), requested_url=source_url)
        (listings_dir / f"{category}.html").write_text(listing_page.html, encoding="utf-8")
        failure = _access_failure(listing_page)
        if failure:
            warnings.append(f"{category}: доступ отклонён")
            status = "partial" if manifest_items else "failed"
            write_manifest(status=status, failure=failure)
            if manifest_items:
                progress(str(failure["message"]))
                return manifest_path
            raise YandexMarketAccessDeniedError(str(failure["message"]), manifest_path=manifest_path)

        products = parse_yandex_market_listing_html(
            listing_page.html,
            limit=options.per_category_limit,
        )
        progress(f"Категория {category}: найдено карточек {len(products)}")
        if not products:
            warnings.append(f"{category}: ссылки на карточки не найдены")
            continue
        for index, product in enumerate(products, start=1):
            if monotonic() - started >= options.time_limit_seconds:
                warnings.append(f"{category}: обработка остановлена по общему таймауту")
                break
            url = product["url"]
            progress(f"{category}: карточка {index}/{len(products)}")
            product_page = _captured_page(fetch(url), requested_url=url)
            relative_file = _snapshot_filename(category, index, url)
            (root / relative_file).write_text(product_page.html, encoding="utf-8")
            failure = _access_failure(product_page)
            if failure:
                warnings.append(f"{category}: доступ отклонён на карточке {index}")
                status = "partial" if manifest_items else "failed"
                write_manifest(status=status, failure=failure)
                if manifest_items:
                    progress(str(failure["message"]))
                    return manifest_path
                raise YandexMarketAccessDeniedError(
                    str(failure["message"]), manifest_path=manifest_path
                )
            manifest_items.append(
                {
                    "file": relative_file,
                    "category": category,
                    "url": product_page.final_url or url,
                    "observed_at": observed_at,
                }
            )
            if options.request_delay_seconds:
                sleep(options.request_delay_seconds)

    failure = None
    if not manifest_items:
        failure = {
            "kind": "no_products",
            "status_code": None,
            "stage": "listing_parse",
            "message": "Яндекс Маркет ответил, но ссылки на карточки товаров не найдены.",
        }
    write_manifest(status="completed" if manifest_items else "failed", failure=failure)
    progress(f"HTML-снимок Яндекс Маркета сохранён: {manifest_path}")
    if not manifest_items:
        raise YandexMarketLiveCollectionError(
            "Карточки Яндекс Маркета не найдены. Проверьте сохранённые listing HTML: "
            "возможны CAPTCHA или изменение разметки.",
            manifest_path=manifest_path,
        )
    return manifest_path


def build_catalog_from_live_yandex_market(
    options: YandexMarketLiveOptions,
    *,
    progress: Callable[[str], None] = print,
) -> dict:
    options.validate()
    with YandexMarketBrowserSession(
        profile_dir=options.profile_dir,
        engine=options.browser_engine,
        headless=options.headless,
        first_page_wait_seconds=options.first_page_wait_seconds,
        progress=progress,
    ) as browser:
        progress("Подготавливаю сессию Яндекс Маркета через главную страницу.")
        browser.fetch(f"{YANDEX_MARKET_BASE_URL}/")
        capture_yandex_market_snapshot(options, fetch=browser.fetch, progress=progress)
    payload = build_catalog_from_yandex_market_snapshot(options.snapshot_dir)
    if int((payload.get("stats") or {}).get("items_total") or 0) <= 0:
        raise YandexMarketLiveCollectionError(
            "Яндекс Маркет сохранил HTML-карточки, но в них не найдено пригодных "
            "данных товара (название + цена/характеристики). Проверьте snapshot: "
            "возможны защитная страница, редирект или изменение разметки.",
            manifest_path=Path(options.snapshot_dir) / "snapshot_manifest.json",
        )
    payload["generated_by"] = "tools.catalog_parser.sources.yandex_market_live"
    manifest = json.loads(
        (Path(options.snapshot_dir) / "snapshot_manifest.json").read_text(encoding="utf-8")
    )
    capture_warnings = list((manifest.get("capture") or {}).get("warnings") or [])
    for source in payload.get("sources", []):
        source["mode"] = "browser-capture+offline-html-jsonld+embedded-json"
        source["warnings"] = list(source.get("warnings") or []) + capture_warnings
    return payload
