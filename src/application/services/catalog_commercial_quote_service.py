"""Commercial-quote import as the highest-trust catalog price observation (P5)."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from application.services.catalog_staging_service import CatalogStagingService


@dataclass(frozen=True)
class CommercialQuoteImportSummary:
    source_id: str
    supplier_name: str
    quote_number: str
    observed_at: str
    records_total: int


def import_commercial_quote(
    staging: CatalogStagingService,
    path: str | Path,
    *,
    supplier_name: str,
    quote_number: str = "",
    quote_date: str = "",
    region: str = "",
    assume_available: bool = True,
) -> CommercialQuoteImportSummary:
    source_path = Path(path)
    if not source_path.is_file():
        raise ValueError(f"Файл коммерческого предложения не найден: {source_path}")
    supplier = str(supplier_name or "").strip()
    if not supplier:
        raise ValueError("Укажите поставщика коммерческого предложения.")
    observed_at = _normalize_date(quote_date)
    if not observed_at:
        observed_at = datetime.now(UTC).isoformat()
    quote = str(quote_number or "").strip()
    source_id = commercial_quote_source_id(supplier, quote or source_path.stem)
    context: dict[str, Any] = {
        "id": source_id,
        "name": supplier,
        "location": str(source_path),
        "format": source_path.suffix.lower().lstrip("."),
        "region": str(region or "").strip(),
        "price_kind": "commercial_quote",
        "observed_at": observed_at,
        "source_type": "commercial_quote",
        "supplier_name": supplier,
        "quote_number": quote or None,
        "quote_date": observed_at,
        "default_availability": "quoted" if assume_available else "unknown",
        "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
    }
    records = staging.stage_file(source_path, source_context=context)
    source_count = 0
    for record in records:
        item = dict(record.get("source_catalog_item") or {})
        for observation in item.get("source_observations", []) or [item]:
            if isinstance(observation, dict) and observation.get("source") == source_id:
                source_count += 1
    return CommercialQuoteImportSummary(
        source_id=source_id,
        supplier_name=supplier,
        quote_number=quote,
        observed_at=observed_at,
        records_total=source_count,
    )


def commercial_quote_source_id(supplier_name: str, quote_key: str) -> str:
    supplier = _slug(supplier_name) or "supplier"
    key = _slug(quote_key) or "quote"
    return f"commercial-quote-{supplier}-{key}"[:120]


def _slug(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9а-яА-Я]+", "-", str(value or "").strip().lower()).strip("-")
    return re.sub(r"-+", "-", text)


def _normalize_date(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return f"{text}T00:00:00+00:00"
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Дата КП должна быть YYYY-MM-DD или ISO-8601.") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).isoformat()


__all__ = [
    "CommercialQuoteImportSummary",
    "commercial_quote_source_id",
    "import_commercial_quote",
]
