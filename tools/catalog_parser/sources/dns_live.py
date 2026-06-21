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

from ..dns_browser import DnsBrowserSession
from .dns_snapshot import build_catalog_from_dns_snapshot

DNS_BASE_URL = "https://www.dns-shop.ru"
DNS_LIVE_CATEGORIES = {
    "routers": "Маршрутизаторы",
    "prebuilt_pcs": "Готовые сборки ПК",
    "servers": "Сервер",
}


@dataclass(frozen=True, slots=True)
class DnsLiveOptions:
    snapshot_dir: Path
    profile_dir: Path
    categories: tuple[str, ...] = ("routers", "prebuilt_pcs", "servers")
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


def capture_dns_snapshot(
    options: DnsLiveOptions,
    *,
    fetch: Callable[[str], str],
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

    for category in options.categories:
        if monotonic() - started >= options.time_limit_seconds:
            warnings.append("Общий таймаут достигнут до обработки всех категорий")
            break
        query = DNS_LIVE_CATEGORIES[category]
        progress(f"Категория {category}: поиск '{query}'")
        search_url = f"{DNS_BASE_URL}/search/?{urlencode({'q': query})}"
        search_html = fetch(search_url)
        (search_dir / f"{category}.html").write_text(search_html, encoding="utf-8")
        products = parse_dns_search_html(search_html, limit=options.per_category_limit)
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
            html = fetch(url)
            relative_file = _snapshot_filename(category, index, url)
            (root / relative_file).write_text(html, encoding="utf-8")
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

    manifest = {
        "schema_version": 1,
        "source": "dns",
        "region": options.region,
        "observed_at": observed_at,
        "capture": {
            "mode": "user-initiated-playwright",
            "categories": list(options.categories),
            "per_category_limit": options.per_category_limit,
            "warnings": warnings,
        },
        "items": manifest_items,
    }
    manifest_path = root / "snapshot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    progress(f"HTML-снимок сохранён: {manifest_path}")
    if not manifest_items:
        raise RuntimeError(
            "DNS не вернул карточки товаров. Проверьте регион/challenge в браузере и сохранённые search HTML."
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
        headless=options.headless,
        first_page_wait_seconds=options.first_page_wait_seconds,
        progress=progress,
    ) as browser:
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
