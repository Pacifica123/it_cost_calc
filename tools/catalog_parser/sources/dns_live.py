from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha1
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable
from urllib.parse import urlencode, urljoin, urlparse

from ..dns_browser import DnsBrowserPage, DnsBrowserSession
from .dns_snapshot import build_catalog_from_dns_snapshot

DNS_BASE_URL = "https://www.dns-shop.ru"
DNS_LIVE_CATEGORIES = {
    "routers": "Маршрутизаторы",
    "switches": "Коммутаторы",
    "prebuilt_pcs": "Готовые сборки ПК",
    "servers": "Сервер",
}
DNS_CATALOG_URLS = {
    "routers": "https://www.dns-shop.ru/catalog/17a8aa1c16404e77/wi-fi-routery/",
    "switches": "https://www.dns-shop.ru/catalog/17a9dc3716404e77/kommutatory/",
}


class DnsLiveCollectionError(RuntimeError):
    exit_code = 4

    def __init__(self, message: str, *, manifest_path: Path) -> None:
        super().__init__(message)
        self.manifest_path = manifest_path


class DnsAccessDeniedError(DnsLiveCollectionError):
    exit_code = 3


@dataclass(frozen=True, slots=True)
class _CapturedPage:
    requested_url: str
    final_url: str
    status_code: int | None
    title: str
    html: str


@dataclass(frozen=True, slots=True)
class DnsLiveOptions:
    snapshot_dir: Path
    profile_dir: Path
    categories: tuple[str, ...] = ("routers", "switches", "prebuilt_pcs", "servers")
    browser_engine: str = "firefox"
    per_category_limit: int = 10
    time_limit_seconds: int = 300
    headless: bool = False
    first_page_wait_seconds: float = 8.0
    request_delay_seconds: float = 0.8
    region: str = ""

    def validate(self) -> None:
        unknown = sorted(set(self.categories) - set(DNS_LIVE_CATEGORIES))
        if unknown:
            raise ValueError(f"Неизвестные DNS-категории: {', '.join(unknown)}")
        if not self.categories:
            raise ValueError("Нужно выбрать хотя бы одну DNS-категорию")
        if not 1 <= self.per_category_limit <= 50:
            raise ValueError("Лимит карточек на категорию должен быть от 1 до 50")
        if not 30 <= self.time_limit_seconds <= 1800:
            raise ValueError("Общий таймаут должен быть от 30 до 1800 секунд")
        if self.browser_engine not in {"firefox", "chromium"}:
            raise ValueError("Движок браузера должен быть firefox или chromium")


class _DnsSearchParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.items: list[dict[str, str]] = []
        self._href: str | None = None
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        attributes = {str(key).lower(): str(value or "") for key, value in attrs}
        href = attributes.get("href", "")
        if "/product/" in href:
            self._href = href
            self._parts = []

    def handle_data(self, data: str) -> None:
        if self._href:
            self._parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or not self._href:
            return
        title = " ".join("".join(self._parts).split())
        self.items.append({"url": urljoin(DNS_BASE_URL, self._href), "title": title})
        self._href = None
        self._parts = []


def parse_dns_search_html(html: str, *, limit: int) -> list[dict[str, str]]:
    parser = _DnsSearchParser()
    parser.feed(html)
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in parser.items:
        url = item["url"]
        parsed = urlparse(url)
        hostname = str(parsed.hostname or "").lower()
        if parsed.scheme != "https" or not (
            hostname == "dns-shop.ru" or hostname.endswith(".dns-shop.ru")
        ):
            continue
        if url in seen:
            continue
        seen.add(url)
        result.append(item)
        if len(result) >= limit:
            break
    return result


def _snapshot_filename(category: str, index: int, url: str) -> str:
    digest = sha1(url.encode("utf-8")).hexdigest()[:10]
    return f"products/{category}_{index:03d}_{digest}.html"


def _captured_page(value: str | DnsBrowserPage, *, requested_url: str) -> _CapturedPage:
    if isinstance(value, DnsBrowserPage):
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
    html_status = next(
        (
            status
            for status, markers in (
                (
                    403,
                    (
                        "<title>http 403</title>",
                        "403 error",
                        "forbidden",
                        "доступ к сайту www.dns-shop.ru запрещен",
                    ),
                ),
                (429, ("too many requests", "<title>http 429</title>")),
                (401, ("<title>http 401</title>",)),
            )
            if any(marker in normalized for marker in markers)
        ),
        None,
    )
    qrator_challenge = any(
        marker in normalized
        for marker in ("/__qrator/qauth_", "qauth_handle_validate_success")
    )
    qrator_hard_block = "guru meditation" in normalized and html_status == 403
    denied = page.status_code in {401, 403, 429} or html_status is not None or qrator_challenge
    if not denied:
        return None
    status = html_status or page.status_code
    if qrator_challenge:
        stage = "challenge_not_completed"
    elif status == 429:
        stage = "rate_limited"
    elif html_status == 403:
        stage = "hard_block"
    else:
        stage = "access_denied"
    if stage == "hard_block":
        message = (
            "DNS вернул окончательную страницу HTTP 403 для автоматизированной сессии. "
            "Ожидание защитной проверки уже завершилось; смена Playwright-движка или селекторов "
            "не устранит этот отказ. Откройте DNS в обычном браузере либо используйте заранее "
            "сохранённый офлайн-снимок."
        )
    elif stage == "challenge_not_completed":
        message = (
            "Защитная JavaScript-проверка DNS не завершилась после ожидания и повторной загрузки. "
            "Сессия сохранена для диагностики; можно повторить позже или открыть DNS в обычном браузере."
        )
    elif stage == "rate_limited":
        message = "DNS ограничил частоту запросов (HTTP 429). Повторите сбор позже."
    else:
        message = (
            f"DNS вернул HTTP {status} и запретил доступ для текущего подключения. "
            "Проверьте доступ в обычном браузере; сохранённый HTML оставлен для диагностики."
        )
    return {
        "kind": "access_denied",
        "status_code": status,
        "response_status_code": page.status_code,
        "html_status_code": html_status,
        "stage": stage,
        "protection_provider": "qrator" if qrator_challenge or qrator_hard_block else None,
        "requested_url": page.requested_url,
        "final_url": page.final_url,
        "page_title": page.title,
        "message": message,
    }


def capture_dns_snapshot(
    options: DnsLiveOptions,
    *,
    fetch: Callable[[str], str | DnsBrowserPage],
    progress: Callable[[str], None] = print,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> Path:
    """Capture deterministic local HTML snapshot using an injected browser fetcher."""

    options.validate()
    root = Path(options.snapshot_dir)
    products_dir = root / "products"
    search_dir = root / "search"
    products_dir.mkdir(parents=True, exist_ok=True)
    search_dir.mkdir(parents=True, exist_ok=True)
    started = monotonic()
    observed_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    manifest_items: list[dict[str, str]] = []
    warnings: list[str] = []
    manifest_path = root / "snapshot_manifest.json"

    def write_manifest(
        *,
        status: str,
        failure: dict[str, object] | None = None,
    ) -> Path:
        capture: dict[str, object] = {
            "mode": "user-initiated-playwright",
            "status": status,
            "browser_engine": options.browser_engine,
            "categories": list(options.categories),
            "per_category_limit": options.per_category_limit,
            "category_sources": {
                category: {
                    "mode": "catalog-url" if category in DNS_CATALOG_URLS else "search-query",
                    "url": DNS_CATALOG_URLS.get(category)
                    or f"{DNS_BASE_URL}/search/?{urlencode({'q': DNS_LIVE_CATEGORIES[category]})}",
                }
                for category in options.categories
            },
            "warnings": warnings,
        }
        if failure:
            capture["failure"] = failure
        manifest = {
            "schema_version": 1,
            "source": "dns",
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
        query = DNS_LIVE_CATEGORIES[category]
        source_url = DNS_CATALOG_URLS.get(category)
        if source_url:
            progress(f"Категория {category}: открываю каталог '{query}'")
        else:
            progress(f"Категория {category}: поиск '{query}'")
            source_url = f"{DNS_BASE_URL}/search/?{urlencode({'q': query})}"
        search_page = _captured_page(fetch(source_url), requested_url=source_url)
        (search_dir / f"{category}.html").write_text(search_page.html, encoding="utf-8")
        failure = _access_failure(search_page)
        if failure:
            warnings.append(f"{category}: DNS access denied, HTTP {failure['status_code']}")
            status = "partial" if manifest_items else "failed"
            write_manifest(status=status, failure=failure)
            progress(str(failure["message"]))
            progress(f"Диагностика сохранена: {manifest_path}")
            if manifest_items:
                progress(
                    f"Сохраняю частичный результат: собрано карточек {len(manifest_items)}."
                )
                return manifest_path
            raise DnsAccessDeniedError(str(failure["message"]), manifest_path=manifest_path)
        products = parse_dns_search_html(search_page.html, limit=options.per_category_limit)
        progress(f"Категория {category}: найдено карточек {len(products)}")
        if not products:
            warnings.append(f"{category}: ссылки на товары не найдены")
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
                warnings.append(f"{category}: DNS access denied, HTTP {failure['status_code']}")
                status = "partial" if manifest_items else "failed"
                write_manifest(status=status, failure=failure)
                progress(str(failure["message"]))
                progress(f"Диагностика сохранена: {manifest_path}")
                if manifest_items:
                    progress(
                        f"Сохраняю частичный результат: собрано карточек {len(manifest_items)}."
                    )
                    return manifest_path
                raise DnsAccessDeniedError(str(failure["message"]), manifest_path=manifest_path)
            manifest_items.append(
                {
                    "file": relative_file,
                    "category": category,
                    "url": url,
                    "observed_at": observed_at,
                }
            )
            if options.request_delay_seconds:
                sleep(options.request_delay_seconds)

    no_products_failure = None
    if not manifest_items:
        no_products_failure = {
            "kind": "no_products",
            "status_code": None,
            "requested_url": None,
            "final_url": None,
            "page_title": None,
            "message": "DNS ответил, но ссылки на карточки товаров не найдены.",
        }
    write_manifest(
        status="completed" if manifest_items else "failed",
        failure=no_products_failure,
    )
    progress(f"HTML-снимок сохранён: {manifest_path}")
    if not manifest_items:
        raise DnsLiveCollectionError(
            "DNS ответил без HTTP-запрета, но карточки товаров не найдены. "
            "Проверьте сохранённые search HTML: это может быть challenge или изменение разметки.",
            manifest_path=manifest_path,
        )
    return manifest_path


def build_catalog_from_live_dns(
    options: DnsLiveOptions,
    *,
    progress: Callable[[str], None] = print,
) -> dict:
    options.validate()
    with DnsBrowserSession(
        profile_dir=options.profile_dir,
        engine=options.browser_engine,
        headless=options.headless,
        first_page_wait_seconds=options.first_page_wait_seconds,
        progress=progress,
    ) as browser:
        progress("Подготавливаю DNS-сессию через главную страницу.")
        browser.fetch(f"{DNS_BASE_URL}/")
        capture_dns_snapshot(options, fetch=browser.fetch, progress=progress)
    payload = build_catalog_from_dns_snapshot(options.snapshot_dir)
    payload["generated_by"] = "tools.catalog_parser.sources.dns_live"
    manifest = json.loads(
        (Path(options.snapshot_dir) / "snapshot_manifest.json").read_text(encoding="utf-8")
    )
    capture_warnings = list((manifest.get("capture") or {}).get("warnings") or [])
    for source in payload.get("sources", []):
        source["mode"] = "browser-capture+offline-html-jsonld"
        source["warnings"] = list(source.get("warnings") or []) + capture_warnings
    return payload


def build_catalog_from_legacy_dns() -> dict:
    """Keep the old experimental mode callable outside the new GUI workflow."""

    import os
    import tempfile

    from ..catalog_builder import build_catalog_payload, deduplicate_items, normalize_dns_snapshot
    from ..catalog_schema import CatalogSourceInfo
    from ..legacy import main as run_legacy_dns_parser

    with tempfile.TemporaryDirectory(prefix="dns_legacy_live_") as temp_dir:
        temp_path = Path(temp_dir)
        previous_cwd = Path.cwd()
        try:
            os.chdir(temp_path)
            run_legacy_dns_parser()
            raw_path = temp_path / "dns_data_playwright.json"
            snapshot = json.loads(raw_path.read_text(encoding="utf-8"))
        finally:
            os.chdir(previous_cwd)

    normalized = normalize_dns_snapshot(snapshot, snapshot_name="dns_data_playwright.json")
    deduplicated = deduplicate_items(normalized)
    return build_catalog_payload(
        items=deduplicated,
        sources=[
            CatalogSourceInfo(
                source="dns",
                snapshot_name="dns_data_playwright.json",
                mode="legacy-live",
                items_before_dedup=len(normalized),
                items_after_dedup=len(deduplicated),
            )
        ],
        generated_by="tools.catalog_parser.sources.dns_live.build_catalog_from_legacy_dns",
    )
