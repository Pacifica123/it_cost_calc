"""Download or copy structured catalog feeds without browser automation."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

_MAX_BYTES = 150 * 1024 * 1024
_TIMEOUT = (10, 90)
_USER_AGENT = "ITCostCalc-CatalogFeed/1.0"
_YANDEX_PUBLIC_DOWNLOAD_API = "https://cloud-api.yandex.net/v1/disk/public/resources/download"


class CatalogFeedFetchError(RuntimeError):
    pass


@dataclass(frozen=True)
class CatalogFeedFetchResult:
    source_id: str
    source_name: str
    requested_location: str
    resolved_location: str
    format: str
    region: str
    price_kind: str
    observed_at: str
    output_path: str
    sha256: str
    size_bytes: int
    content_type: str
    download_strategy: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fetch_catalog_feed(
    *,
    location: str,
    output_path: str | Path,
    source_id: str,
    source_name: str,
    feed_format: str = "auto",
    region: str = "",
    price_kind: str = "supplier_price",
    download_strategy: str = "direct",
    manifest_path: str | Path | None = None,
) -> CatalogFeedFetchResult:
    """Fetch one structured feed and persist a provenance manifest."""

    requested = str(location or "").strip()
    if not requested:
        raise CatalogFeedFetchError("Не задан URL или локальный путь источника.")
    normalized_format = str(feed_format or "auto").strip().lower()
    if normalized_format not in {"auto", "xlsx", "csv", "xml", "yml"}:
        raise CatalogFeedFetchError(f"Неподдерживаемый формат feed: {normalized_format}")
    strategy = str(download_strategy or "direct").strip().lower()
    if strategy not in {"direct", "yandex_disk_public"}:
        raise CatalogFeedFetchError(f"Неподдерживаемая стратегия загрузки: {strategy}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    observed_at = datetime.now(UTC).isoformat()

    parsed = urlparse(requested)
    content_type = ""
    resolved = requested
    if parsed.scheme in {"http", "https"}:
        if strategy == "yandex_disk_public":
            resolved = _resolve_yandex_public_download(requested)
        body, resolved, content_type = _download_http(resolved)
        actual_format = _detect_format(body, resolved, content_type, requested=normalized_format)
        output.write_bytes(body)
    else:
        source_path = Path(requested).expanduser().resolve()
        if not source_path.is_file():
            raise CatalogFeedFetchError(f"Локальный feed не найден: {source_path}")
        if source_path.stat().st_size > _MAX_BYTES:
            raise CatalogFeedFetchError("Файл feed превышает допустимый размер 150 МБ.")
        body = source_path.read_bytes()
        actual_format = _detect_format(body, source_path.name, "", requested=normalized_format)
        shutil.copyfile(source_path, output)
        resolved = str(source_path)

    digest = hashlib.sha256(body).hexdigest()
    result = CatalogFeedFetchResult(
        source_id=str(source_id or "imported-feed").strip() or "imported-feed",
        source_name=str(source_name or source_id or "Импортированный feed").strip(),
        requested_location=requested,
        resolved_location=resolved,
        format=actual_format,
        region=str(region or "").strip(),
        price_kind=str(price_kind or "supplier_price").strip() or "supplier_price",
        observed_at=observed_at,
        output_path=str(output.resolve()),
        sha256=digest,
        size_bytes=len(body),
        content_type=content_type,
        download_strategy=strategy,
    )
    target_manifest = Path(manifest_path) if manifest_path is not None else output.with_suffix(output.suffix + ".source.json")
    target_manifest.parent.mkdir(parents=True, exist_ok=True)
    target_manifest.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _resolve_yandex_public_download(public_url: str) -> str:
    try:
        response = requests.get(
            _YANDEX_PUBLIC_DOWNLOAD_API,
            params={"public_key": public_url},
            headers={"User-Agent": _USER_AGENT, "Accept": "application/json"},
            timeout=_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise CatalogFeedFetchError(f"Не удалось получить ссылку скачивания Яндекс Диска: {exc}") from exc
    href = str(payload.get("href") or "").strip() if isinstance(payload, dict) else ""
    if not href:
        raise CatalogFeedFetchError("Яндекс Диск не вернул ссылку для публичного файла.")
    return href


def _download_http(url: str) -> tuple[bytes, str, str]:
    try:
        response = requests.get(
            url,
            headers={
                "User-Agent": _USER_AGENT,
                "Accept": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,text/csv,application/xml,text/xml,*/*;q=0.5",
            },
            timeout=_TIMEOUT,
            allow_redirects=True,
            stream=True,
        )
        response.raise_for_status()
        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_content(chunk_size=1024 * 128):
            if not chunk:
                continue
            total += len(chunk)
            if total > _MAX_BYTES:
                raise CatalogFeedFetchError("Удалённый feed превышает допустимый размер 150 МБ.")
            chunks.append(chunk)
        body = b"".join(chunks)
        if not body:
            raise CatalogFeedFetchError("Источник вернул пустой файл.")
        return body, str(response.url), str(response.headers.get("Content-Type") or "")
    except CatalogFeedFetchError:
        raise
    except requests.RequestException as exc:
        raise CatalogFeedFetchError(f"Не удалось скачать feed: {exc}") from exc


def _detect_format(body: bytes, location: str, content_type: str, *, requested: str) -> str:
    if requested != "auto":
        _validate_expected_format(body, requested)
        return requested
    suffix = Path(urlparse(location).path).suffix.lower()
    if body.startswith(b"PK\x03\x04"):
        return "xlsx"
    prefix = body[:2048].lstrip().lower()
    if _looks_like_html(prefix):
        raise CatalogFeedFetchError("Источник вернул HTML вместо структурированного feed.")
    if prefix.startswith(b"<?xml") or prefix.startswith(b"<yml_catalog") or prefix.startswith(b"<shop"):
        return "yml" if b"<yml_catalog" in prefix else "xml"
    if suffix in {".xlsx", ".csv", ".xml", ".yml"}:
        return suffix.lstrip(".")
    lowered_type = str(content_type or "").lower()
    if "spreadsheetml" in lowered_type:
        return "xlsx"
    if "csv" in lowered_type:
        return "csv"
    if "xml" in lowered_type:
        return "xml"
    try:
        body[:8192].decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    else:
        return "csv"
    raise CatalogFeedFetchError("Не удалось определить формат feed. Укажите его явно.")


def _validate_expected_format(body: bytes, expected: str) -> None:
    prefix = body[:4096].lstrip().lower()
    if _looks_like_html(prefix):
        raise CatalogFeedFetchError("Источник вернул HTML вместо структурированного feed.")
    if expected == "xlsx" and not body.startswith(b"PK\x03\x04"):
        sample = prefix[:120].decode("utf-8", errors="ignore")
        raise CatalogFeedFetchError(f"Вместо XLSX получен другой ответ: {sample!r}")
    if expected in {"xml", "yml"} and not prefix.startswith(b"<"):
        raise CatalogFeedFetchError("Источник не вернул XML/YML.")
    if expected == "yml" and b"<yml_catalog" not in prefix:
        raise CatalogFeedFetchError("Источник не вернул YML-каталог.")
    if expected == "csv" and body.startswith(b"PK\x03\x04"):
        raise CatalogFeedFetchError("Источник вернул XLSX вместо CSV.")


def _looks_like_html(prefix: bytes) -> bool:
    return prefix.startswith((b"<!doctype html", b"<html", b"<head", b"<body"))


__all__ = [
    "CatalogFeedFetchError",
    "CatalogFeedFetchResult",
    "fetch_catalog_feed",
]
