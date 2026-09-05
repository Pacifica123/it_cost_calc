from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urljoin, urlparse

from ..catalog_builder import build_catalog_payload, deduplicate_items, normalize_dns_snapshot
from ..catalog_schema import CatalogSourceInfo
from ..http_session import CatalogHttpRequestError, CatalogHttpResponse, CatalogHttpSession

DNS_BASE_URL = "https://www.dns-shop.ru"
DNS_CATALOG_MARKDOWN_URL = f"{DNS_BASE_URL}/catalog/markdown/"
DNS_PRODUCT_BUY_URL = f"{DNS_BASE_URL}/ajax-state/product-buy/"
DNS_HTTP_CATEGORIES = {
    "routers": "Маршрутизаторы",
    "switches": "Коммутаторы",
    "prebuilt_pcs": "Готовые ПК",
    "servers": "Серверы",
}
DNS_HTTP_CATEGORY_URLS = {
    "routers": "https://www.dns-shop.ru/catalog/17a8aa1c16404e77/wi-fi-routery/",
    "switches": "https://www.dns-shop.ru/catalog/17a9dc3716404e77/kommutatory/",
    "prebuilt_pcs": "https://www.dns-shop.ru/catalog/17a8932c16404e77/personalnye-komputery/",
    "servers": "https://www.dns-shop.ru/catalog/17a8939816404e77/servery/",
}

_UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
_PRODUCT_UUID_ESCAPED_RE = re.compile(
    rf'\\\"id\\\":\\\"({_UUID})\\\",\\\"type\\\":4',
    re.IGNORECASE,
)
_PRODUCT_UUID_PLAIN_RE = re.compile(
    rf'"id"\s*:\s*"({_UUID})"\s*,\s*"type"\s*:\s*4',
    re.IGNORECASE,
)
_PRODUCT_BUY_HASH_RE = re.compile(
    r'(?:\\\"|\")hash(?:\\\"|\")\s*:\s*(?:\\\"|\")([0-9a-f]{40,})(?:\\\"|\")',
    re.IGNORECASE,
)
_CSRF_PATTERNS = (
    re.compile(r'<meta[^>]+name=["\']csrf-token["\'][^>]+content=["\']([^"\']+)', re.I),
    re.compile(r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']csrf-token["\']', re.I),
    re.compile(r'"csrfToken"\s*:\s*"([^"]+)"', re.I),
)
_CHALLENGE_MARKERS = (
    "/__qrator/qauth_",
    "qauth_handle_validate_success",
    "<title>http 403</title>",
    "403 error",
    "доступ к сайту www.dns-shop.ru запрещен",
    "проверка безопасности",
)

CatalogBatch = tuple[str, list[dict[str, Any]]]


class DnsHttpCollectionError(RuntimeError):
    exit_code = 4

    def __init__(self, message: str, *, manifest_path: Path) -> None:
        super().__init__(message)
        self.manifest_path = manifest_path


class DnsHttpAccessDeniedError(DnsHttpCollectionError):
    exit_code = 3


@dataclass(frozen=True, slots=True)
class DnsHttpLiveOptions:
    snapshot_dir: Path
    categories: tuple[str, ...] = ("routers", "switches", "prebuilt_pcs", "servers")
    per_category_limit: int = 10
    time_limit_seconds: int = 300
    request_delay_seconds: float = 0.45
    region: str = ""

    def validate(self) -> None:
        unknown = sorted(set(self.categories) - set(DNS_HTTP_CATEGORIES))
        if unknown:
            raise ValueError(f"Неизвестные DNS-категории: {', '.join(unknown)}")
        if not self.categories:
            raise ValueError("Нужно выбрать хотя бы одну DNS-категорию")
        if not 1 <= self.per_category_limit <= 50:
            raise ValueError("Лимит карточек на категорию должен быть от 1 до 50")
        if not 30 <= self.time_limit_seconds <= 1800:
            raise ValueError("Общий таймаут должен быть от 30 до 1800 секунд")


def _filter_product_containers(containers: Iterable[Any]) -> tuple[list[str], list[dict[str, Any]]]:
    uuids: list[str] = []
    valid: list[dict[str, Any]] = []
    for container in containers:
        if not isinstance(container, dict):
            continue
        inner = container.get("data")
        if not isinstance(inner, dict) or inner.get("type") != 4:
            continue
        uuid = str(inner.get("id") or "").strip().lower()
        if not re.fullmatch(_UUID, uuid, re.IGNORECASE):
            continue
        uuids.append(uuid)
        valid.append(container)
    return uuids, valid


def _batches_from_registered_value(value: Any) -> list[CatalogBatch]:
    results: list[CatalogBatch] = []
    candidates = value if isinstance(value, list) else []
    if len(candidates) >= 2 and isinstance(candidates[0], dict):
        candidates = [candidates]
    for batch in candidates:
        if not isinstance(batch, list) or len(batch) < 2:
            continue
        config = batch[0] if isinstance(batch[0], dict) else {}
        if config.get("type") != "product-buy":
            continue
        product_hash = str(config.get("hash") or "").strip()
        containers = batch[1] if isinstance(batch[1], list) else []
        _uuids, valid = _filter_product_containers(containers)
        if product_hash and valid:
            results.append((product_hash, valid))
    return results


def parse_dns_product_buy_batches(raw: str) -> list[CatalogBatch]:
    """Extract the product-buy hash + original containers from DNS JSON/inline JS.

    This is the useful part of the user supplied ``simple_dns_parser.py`` ported
    into the project's catalog boundary.  Original server container ids remain
    paired with the hash that produced them.
    """

    batches: list[CatalogBatch] = []
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        payload = None

    if isinstance(payload, list):
        batches.extend(_batches_from_registered_value(payload))
    elif isinstance(payload, dict):
        inline = (payload.get("assets") or {}).get("inlineJs") if isinstance(payload.get("assets"), dict) else None
        if isinstance(inline, dict):
            for source in inline.values():
                batches.extend(_parse_ajax_state_text(str(source)))
        for key in ("html", "data"):
            source = payload.get(key)
            if isinstance(source, str):
                batches.extend(_parse_ajax_state_text(source))

    if not batches:
        batches.extend(_parse_ajax_state_text(raw))
    if batches:
        return _dedupe_batches(batches)

    uuids = [match.group(1).lower() for match in _PRODUCT_UUID_ESCAPED_RE.finditer(raw)]
    if not uuids:
        uuids = [match.group(1).lower() for match in _PRODUCT_UUID_PLAIN_RE.finditer(raw)]
    hash_match = _PRODUCT_BUY_HASH_RE.search(raw)
    if uuids and hash_match:
        containers = [
            {"id": f"http-{index:03d}", "data": {"id": uuid, "type": 4, "params": {"hideButtons": True}}}
            for index, uuid in enumerate(dict.fromkeys(uuids), start=1)
        ]
        return [(hash_match.group(1), containers)]
    return []


def _registered_json_arguments(source: str) -> Iterable[str]:
    """Yield balanced JSON array arguments passed to ``AjaxState.register``.

    A non-greedy bracket regex is unsafe here because product containers contain
    nested arrays/objects.  This small scanner understands JSON strings and nesting
    without executing any JavaScript.
    """

    marker = "AjaxState.register("
    search_from = 0
    while True:
        marker_at = source.find(marker, search_from)
        if marker_at < 0:
            return
        start = marker_at + len(marker)
        while start < len(source) and source[start].isspace():
            start += 1
        if start >= len(source) or source[start] != "[":
            search_from = start
            continue

        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(source)):
            char = source[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char in "[{":
                depth += 1
            elif char in "]}":
                depth -= 1
                if depth == 0:
                    yield source[start : index + 1]
                    search_from = index + 1
                    break
        else:
            return


def _parse_ajax_state_text(source: str) -> list[CatalogBatch]:
    batches: list[CatalogBatch] = []
    for raw_value in _registered_json_arguments(source):
        try:
            value = json.loads(raw_value)
        except (json.JSONDecodeError, ValueError):
            continue
        batches.extend(_batches_from_registered_value(value))
    return batches


def _dedupe_batches(batches: Iterable[CatalogBatch]) -> list[CatalogBatch]:
    seen: set[tuple[str, tuple[str, ...]]] = set()
    result: list[CatalogBatch] = []
    for product_hash, containers in batches:
        ids = tuple(str((container.get("data") or {}).get("id") or "") for container in containers)
        key = (product_hash, ids)
        if key in seen:
            continue
        seen.add(key)
        result.append((product_hash, containers))
    return result


def _csrf_token(raw: str) -> str | None:
    for pattern in _CSRF_PATTERNS:
        match = pattern.search(raw)
        if match:
            return match.group(1)
    return None


def _access_failure(response: CatalogHttpResponse) -> dict[str, object] | None:
    normalized = response.text.lower()
    status = response.status_code if response.status_code in {401, 403, 429} else None
    challenge = any(marker in normalized for marker in _CHALLENGE_MARKERS)
    if status is None and not challenge:
        return None
    if status == 429:
        message = "DNS ограничил частоту HTTP-запросов (429). Повторите позже."
        stage = "rate_limited"
    elif status in {401, 403}:
        message = f"DNS отклонил HTTP-сессию (HTTP {status})."
        stage = "access_denied"
    else:
        message = "DNS вернул защитную страницу вместо каталога."
        stage = "challenge"
    return {
        "kind": "access_denied",
        "status_code": status,
        "stage": stage,
        "requested_url": response.requested_url,
        "final_url": response.final_url,
        "message": message,
    }


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _state_to_raw_item(
    state: Any,
    *,
    container_map: dict[str, str],
    category: str,
    region: str,
    observed_at: str,
) -> dict[str, Any] | None:
    if not isinstance(state, dict):
        return None
    inner = state.get("data")
    if not isinstance(inner, dict):
        return None
    container_id = str(state.get("id") or "")
    uuid = str(inner.get("id") or container_map.get(container_id) or "").strip().lower()
    title = str(inner.get("name") or inner.get("title") or "").strip()
    price_obj = inner.get("price") if isinstance(inner.get("price"), dict) else {}
    try:
        price = int(float(price_obj.get("current") or inner.get("priceValue") or 0))
    except (TypeError, ValueError):
        price = 0
    if not uuid or not title or price <= 0:
        return None

    item_url = str(inner.get("url") or "").strip()
    if item_url:
        item_url = urljoin(DNS_BASE_URL, item_url)
    else:
        item_url = f"{DNS_CATALOG_MARKDOWN_URL}{uuid}/"
    parsed = urlparse(item_url)
    if parsed.scheme != "https" or not str(parsed.hostname or "").endswith("dns-shop.ru"):
        item_url = f"{DNS_CATALOG_MARKDOWN_URL}{uuid}/"

    specs: dict[str, Any] = {}
    previous = price_obj.get("previous")
    if previous not in (None, "", 0, "0"):
        specs["previous_price_rub"] = previous
    for key in ("brand", "model", "code", "article"):
        value = inner.get(key)
        if value not in (None, ""):
            specs[key] = value

    return {
        "title": title,
        "price_int": price,
        "currency": "RUB",
        "availability": "unknown",
        "url": item_url,
        "type": category,
        "source_product_id": uuid,
        "specs": specs,
        "observed_at": observed_at,
        "region": region,
        "raw_snapshot": f"http-json:{category}",
        "parse_method": "dns-http-product-buy",
        "parse_warnings": [
            "HTTP JSON collector получает цену и название из product-buy; технические характеристики могут потребовать ручной проверки."
        ],
    }


def build_catalog_from_http_dns(
    options: DnsHttpLiveOptions,
    *,
    progress: Callable[[str], None] = print,
    session_factory: Callable[..., CatalogHttpSession] = CatalogHttpSession,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    options.validate()
    root = Path(options.snapshot_dir)
    raw_dir = root / "raw"
    product_buy_dir = root / "product_buy"
    raw_dir.mkdir(parents=True, exist_ok=True)
    product_buy_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "snapshot_manifest.json"
    observed_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    started = monotonic()
    warnings: list[str] = []
    snapshot: dict[str, list[dict[str, Any]]] = {}
    failure: dict[str, object] | None = None

    def write_manifest(status: str) -> None:
        manifest = {
            "schema_version": 1,
            "source": "dns",
            "region": options.region,
            "observed_at": observed_at,
            "capture": {
                "mode": "user-initiated-http-json",
                "status": status,
                "transport": "requests-session",
                "categories": list(options.categories),
                "per_category_limit": options.per_category_limit,
                "warnings": warnings,
                **({"failure": failure} if failure else {}),
            },
            "items": [
                {
                    "category": category,
                    "source_product_id": item.get("source_product_id"),
                    "url": item.get("url"),
                    "observed_at": observed_at,
                }
                for category, items in snapshot.items()
                for item in items
            ],
        }
        manifest_path.write_text(_safe_json(manifest), encoding="utf-8")

    try:
        with session_factory(timeout_seconds=min(30.0, float(options.time_limit_seconds))) as session:
            progress("HTTP: подготавливаю DNS-сессию.")
            warm = session.get(f"{DNS_BASE_URL}/")
            (raw_dir / "warmup.html").write_text(warm.text, encoding="utf-8")
            warm_failure = _access_failure(warm)
            if warm_failure:
                failure = warm_failure
                write_manifest("failed")
                raise DnsHttpAccessDeniedError(str(warm_failure["message"]), manifest_path=manifest_path)

            # The supplied parser intentionally touches /catalog/markdown/ first so DNS
            # can establish current_path/city cookies.  Keep this as best-effort warmup.
            try:
                markdown_warm = session.get(DNS_CATALOG_MARKDOWN_URL, xhr=True, referer=f"{DNS_BASE_URL}/")
                (raw_dir / "catalog_markdown_warmup.txt").write_text(markdown_warm.text, encoding="utf-8")
            except CatalogHttpRequestError as exc:
                warnings.append(f"catalog markdown warmup: {exc}")

            for category in options.categories:
                if monotonic() - started >= options.time_limit_seconds:
                    warnings.append("Общий таймаут достигнут до обработки всех категорий")
                    break
                category_url = DNS_HTTP_CATEGORY_URLS[category]
                progress(f"DNS HTTP: {DNS_HTTP_CATEGORIES[category]}")
                response = session.get(category_url, xhr=True, referer=f"{DNS_BASE_URL}/")
                (raw_dir / f"{category}.txt").write_text(response.text, encoding="utf-8")
                access = _access_failure(response)
                if access:
                    failure = access
                    write_manifest("partial" if snapshot else "failed")
                    if snapshot:
                        warnings.append(str(access["message"]))
                        break
                    raise DnsHttpAccessDeniedError(str(access["message"]), manifest_path=manifest_path)

                batches = parse_dns_product_buy_batches(response.text)
                if not batches:
                    warnings.append(f"{category}: product-buy batch не найден")
                    continue
                csrf = _csrf_token(response.text)
                collected = 0
                for batch_index, (product_hash, containers) in enumerate(batches, start=1):
                    if collected >= options.per_category_limit:
                        break
                    remaining = options.per_category_limit - collected
                    selected_containers = containers[:remaining]
                    container_map = {
                        str(container.get("id") or ""): str((container.get("data") or {}).get("id") or "").lower()
                        for container in selected_containers
                    }
                    payload = {
                        "type": "product-buy",
                        "hash": product_hash,
                        "containers": selected_containers,
                    }
                    result = session.post_form(
                        DNS_PRODUCT_BUY_URL,
                        data="data=" + json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                        referer=category_url,
                        csrf_token=csrf,
                    )
                    (product_buy_dir / f"{category}_{batch_index:02d}.json").write_text(
                        result.text, encoding="utf-8"
                    )
                    access = _access_failure(result)
                    if access:
                        failure = access
                        write_manifest("partial" if snapshot else "failed")
                        if snapshot:
                            warnings.append(str(access["message"]))
                            break
                        raise DnsHttpAccessDeniedError(str(access["message"]), manifest_path=manifest_path)
                    try:
                        decoded = json.loads(result.text)
                    except json.JSONDecodeError:
                        warnings.append(f"{category}: product-buy batch {batch_index} вернул не JSON")
                        continue
                    states = ((decoded.get("data") or {}).get("states") or []) if isinstance(decoded, dict) else []
                    for state in states:
                        item = _state_to_raw_item(
                            state,
                            container_map=container_map,
                            category=category,
                            region=options.region,
                            observed_at=observed_at,
                        )
                        if item is not None:
                            snapshot.setdefault(category, []).append(item)
                            collected += 1
                            if collected >= options.per_category_limit:
                                break
                    progress(f"{category}: получено {collected}")
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
        write_manifest("partial" if snapshot else "failed")
        if not snapshot:
            raise DnsHttpCollectionError(str(exc), manifest_path=manifest_path) from exc
        warnings.append(str(exc))

    if not snapshot:
        if failure is None:
            failure = {
                "kind": "no_products",
                "status_code": None,
                "stage": "catalog_json",
                "message": "DNS ответил, но product-buy данные товаров не найдены.",
            }
        write_manifest("failed")
        raise DnsHttpCollectionError(str(failure["message"]), manifest_path=manifest_path)

    write_manifest("partial" if failure else "completed")
    normalized = normalize_dns_snapshot(snapshot, snapshot_name=manifest_path.name)
    deduplicated = deduplicate_items(normalized)
    payload = build_catalog_payload(
        items=deduplicated,
        sources=[
            CatalogSourceInfo(
                source="dns",
                snapshot_name=root.name,
                mode="http-json+ajax-state-product-buy",
                items_before_dedup=len(normalized),
                items_after_dedup=len(deduplicated),
                warnings=warnings,
            )
        ],
        generated_by="tools.catalog_parser.sources.dns_http_live",
    )
    progress(f"DNS HTTP: готово товаров {len(deduplicated)}")
    progress(f"Диагностика: {manifest_path}")
    return payload
