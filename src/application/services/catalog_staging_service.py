"""Reviewable import boundary between external catalogs and runtime ТО rows."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import posixpath
import re
import zipfile
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping
from xml.etree import ElementTree

from infrastructure.storage import CatalogStagingReadModel, JsonFileStorage

from application.services.catalog_federation_service import (
    federate_catalog_items,
    observation_keys,
    source_observations,
)
from application.services.catalog_enrichment_service import carry_specification_sources
from application.services.catalog_local_enrichment_service import (
    infer_explicit_metrics,
    infer_price_candidate,
)

logger = logging.getLogger(__name__)

CATALOG_STAGING_SCHEMA_VERSION = 3
CATALOG_SOURCE_SCHEMA_VERSION = 2

STAGING_PENDING = "pending"
STAGING_APPROVED = "approved"
STAGING_REJECTED = "rejected"
STAGING_BLOCKED = "blocked"
STAGING_IMPORTED = "imported"

_SUPPORTED_EXTENSIONS = {".json", ".csv", ".xlsx", ".xml", ".yml"}
_METRIC_FIELDS = (
    "ram_gb",
    "cpu_cores",
    "storage_gb",
    "max_power_watts",
    "lan_ports",
    "lan_speed_mbps",
    "wifi_total_mbps",
    "ipv6_support",
)
_TARGET_BY_CATEGORY = {
    "server": ("server", "server"),
    "rack_server": ("server", "server"),
    "tower_server": ("server", "server"),
    "prebuilt_pc": ("client", "workstation"),
    "workstation": ("client", "workstation"),
    "desktop": ("client", "workstation"),
    "laptop": ("client", "workstation"),
    "peripheral": ("client", "peripheral"),
    "printer": ("client", "peripheral"),
    "monitor": ("client", "peripheral"),
    "router": ("network", "network_device"),
    "switch": ("network", "network_device"),
    "access_point": ("network", "network_device"),
    "network_device": ("network", "network_device"),
}
_ALLOWED_TARGET_TYPES = {
    "server": ("server",),
    "client": ("workstation", "peripheral"),
    "network": ("network_device",),
}


def supported_catalog_extensions() -> tuple[str, ...]:
    return tuple(sorted(_SUPPORTED_EXTENSIONS))


def target_for_catalog_category(category: str) -> tuple[str, str] | None:
    return _TARGET_BY_CATEGORY.get(str(category or "").strip().lower())


def catalog_target_options() -> dict[str, tuple[str, ...]]:
    return dict(_ALLOWED_TARGET_TYPES)


def catalog_metric_fields() -> tuple[str, ...]:
    return _METRIC_FIELDS


def iter_catalog_rows(
    path: str | Path,
    *,
    max_rows: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Iterate catalog rows without materialising XLSX/CSV sheets in memory.

    ``max_rows`` limits *input alternatives* for an intentionally scoped demo
    import. ``None`` or ``0`` means all rows.  The main staging path consumes
    this iterator, while :func:`load_catalog_rows` remains a compatibility
    helper for tests and small callers.
    """

    source_path = Path(path)
    suffix = source_path.suffix.lower()
    if suffix not in _SUPPORTED_EXTENSIONS:
        raise ValueError(f"Неподдерживаемый формат каталога: {suffix or 'без расширения'}")
    limit = None if not max_rows or int(max_rows) <= 0 else int(max_rows)

    if suffix == ".json":
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            rows = payload.get("items", [])
            schema_version = int(payload.get("schema_version") or 1)
        else:
            rows = payload
            schema_version = 1
        if not isinstance(rows, list):
            raise ValueError("JSON-каталог должен содержать список items")
        emitted = 0
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            item = dict(row)
            item.setdefault("_catalog_schema_version", schema_version)
            yield item
            emitted += 1
            if limit is not None and emitted >= limit:
                return
        return
    if suffix == ".csv":
        yield from _iter_csv_rows(source_path, max_rows=limit)
        return
    if suffix == ".xlsx":
        yield from _iter_xlsx_rows(source_path, max_rows=limit)
        return
    for index, row in enumerate(_load_yml_rows(source_path)):
        if limit is not None and index >= limit:
            break
        yield row


def load_catalog_rows(path: str | Path) -> list[dict[str, Any]]:
    """Compatibility helper returning all rows as a list."""

    return list(iter_catalog_rows(path))


def normalize_catalog_item(
    raw: Mapping[str, Any],
    *,
    source_path: str | Path,
    row_index: int,
    source_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize schema v1/v2 and flat feed rows to one staging item."""

    item = dict(deepcopy(raw))
    context = _mapping(source_context)
    identity = _mapping(item.get("identity"))
    offer = _mapping(item.get("offer"))
    attributes = _mapping_or_json(item.get("attributes"))
    parsed_metrics = _mapping(attributes.get("parsed_metrics"))
    metrics = {**parsed_metrics, **attributes}
    parser_metadata = {
        key: deepcopy(attributes[key])
        for key in ("parsed_metrics", "parse_warnings", "confidence", "parse_source")
        if key in attributes
    }

    title = _text(
        _first_alias(
            item,
            "title",
            "name",
            "product_name",
            "наименование",
            "наименование товара",
            "название",
            "товар",
            "номенклатура",
            "наименование номенклатуры",
            "описание товара",
        )
    )
    raw_category = _text(
        _first_alias(
            item,
            "category",
            "type",
            "product_type",
            "категория",
            "категория товара",
            "раздел",
            "группа",
            "товарная группа",
            "товарная категория",
            "категория номенклатуры",
            "раздел номенклатуры",
        )
    )
    category = _normalize_catalog_category(raw_category, title)
    source = _text(
        item.get("source")
        or context.get("id")
        or context.get("source_id")
        or Path(source_path).stem
        or "imported"
    )
    url = _text(
        offer.get("url")
        or _first_alias(item, "url", "link", "ссылка", "ссылка на товар", "url товара")
    )
    price = _number(
        _first_alias(
            offer,
            "price",
            "price_rub",
            "cost",
            "цена",
            "цена руб",
            "цена, руб",
            "стоимость",
            "розничная цена",
        )
    )
    if price is None:
        price = _number(
            _first_alias(
                item,
                "price_rub",
                "price",
                "cost",
                "цена",
                "цена руб",
                "цена, руб",
                "цена с ндс",
                "цена с ндс, руб",
                "цена партнера",
                "цена партнёра",
                "оптовая цена",
                "ррц",
                "стоимость",
                "розничная цена",
            )
        )
    if price is None:
        # Supplier XLSX feeds rarely agree on one exact price header.  Use a
        # scored current-price detector instead of silently creating 31k zeroes.
        price, _price_header = infer_price_candidate(offer, item)
    else:
        _price_header = ""
    currency = _text(
        offer.get("currency")
        or _first_alias(item, "currency", "валюта", "currencyid")
        or "RUB"
    ).upper()
    availability = _text(
        offer.get("availability")
        or _first_alias(item, "availability", "available", "наличие", "остаток")
        or context.get("default_availability")
        or "unknown"
    )
    observed_at = _text(
        offer.get("observed_at")
        or item.get("observed_at")
        or context.get("observed_at")
    )

    normalized_identity = {
        key: value
        for key, value in {
            "brand": _text(
                identity.get("brand")
                or _first_alias(item, "brand", "vendor", "бренд", "производитель")
            ),
            "model": _text(
                identity.get("model")
                or _first_alias(item, "model", "модель")
            ),
            "mpn": _text(
                identity.get("mpn")
                or _first_alias(
                    item,
                    "mpn",
                    "pn",
                    "part number",
                    "part_number",
                    "vendorcode",
                    "артикул производителя",
                    "парт номер",
                    "парт-номер",
                )
            ),
            "gtin": _text(
                identity.get("gtin")
                or _first_alias(item, "gtin", "ean", "barcode", "штрихкод")
            ),
        }.items()
        if value
    }
    normalized_metrics = {
        field: _metric_field_value(field, metrics.get(field, item.get(field)))
        for field in _METRIC_FIELDS
        if _metric_field_value(field, metrics.get(field, item.get(field))) is not None
    }
    explicit_metrics, explicit_evidence = infer_explicit_metrics(
        item,
        title=title,
        category=category,
    )
    for field, value in explicit_metrics.items():
        normalized_metrics.setdefault(field, value)
    source_product_id = _text(
        item.get("source_product_id")
        or _first_alias(
            item,
            "product_id",
            "vendor_id",
            "sku",
            "article",
            "артикул",
            "код товара",
            "код",
        )
    )
    catalog_item_id = _text(item.get("item_id") or item.get("id"))
    if not catalog_item_id:
        catalog_item_id = _stable_id(
            source,
            source_product_id,
            normalized_identity,
            url,
            title,
            row_index,
        )

    context_location = _text(
        context.get("resolved_location")
        or context.get("location")
        or context.get("requested_location")
    )
    source_name = _text(context.get("name") or context.get("source_name") or source)
    feed_format = _text(context.get("format") or Path(source_path).suffix.lstrip(".")).lower()
    price_kind = _text(
        offer.get("price_kind") or item.get("price_kind") or context.get("price_kind")
    ) or "retail_offer"
    method = f"feed:{feed_format or 'file'}"

    field_provenance = _mapping(item.get("field_provenance"))
    if context:
        field_provenance.setdefault(
            "feed",
            {
                "source": source,
                "source_name": source_name,
                "source_url": context_location or None,
                "format": feed_format or None,
                "region": _text(context.get("region")) or None,
                "price_kind": price_kind,
                "observed_at": observed_at or None,
                "sha256": _text(context.get("sha256")) or None,
                "source_type": _text(context.get("source_type")) or None,
                "supplier_name": _text(context.get("supplier_name")) or None,
                "quote_number": _text(context.get("quote_number")) or None,
                "quote_date": _text(context.get("quote_date")) or None,
                "capture_method": _text(context.get("capture_method")) or None,
            },
        )
        for field, present in (
            ("title", bool(title)),
            ("category", bool(raw_category or category)),
            ("price", price is not None),
            ("identity", bool(normalized_identity)),
        ):
            if present:
                field_provenance.setdefault(
                    field,
                    {
                        "source": source,
                        "method": method,
                        "observed_at": observed_at or None,
                    },
                )

    if _price_header:
        field_provenance.setdefault(
            "price",
            {
                "source": source,
                "method": "feed:fuzzy-price-column",
                "column": _price_header,
                "observed_at": observed_at or None,
            },
        )
    if explicit_evidence:
        specs = _mapping(field_provenance.get("local_explicit"))
        for field, evidence in explicit_evidence.items():
            if field in normalized_metrics:
                specs.setdefault(field, deepcopy(evidence))
        if specs:
            field_provenance["local_explicit"] = specs

    return {
        "item_id": catalog_item_id,
        "title": title,
        "category": category,
        "source_category": raw_category or None,
        "source": source,
        "source_product_id": source_product_id or None,
        "identity": normalized_identity,
        "offer": {
            "price": price,
            "currency": currency,
            "availability": availability,
            "url": url or None,
            "region": _text(offer.get("region") or item.get("region") or context.get("region")),
            "observed_at": observed_at or None,
            "price_kind": price_kind,
            "source_url": context_location or None,
        },
        "attributes": normalized_metrics,
        "field_provenance": field_provenance,
        "review": _mapping(item.get("review")),
        "parser_metadata": parser_metadata,
        "source_schema_version": int(
            item.get("_catalog_schema_version") or item.get("schema_version") or 1
        ),
        "source_row": row_index + 1,
    }


def validate_staging_item(item: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    title = _text(item.get("title"))
    category = _text(item.get("category")).lower()
    offer = _mapping(item.get("offer"))
    identity = _mapping(item.get("identity"))
    metrics = _mapping(item.get("attributes"))
    review = _mapping(item.get("review"))
    parser_metadata = _mapping(item.get("parser_metadata"))

    if not title:
        errors.append("Не заполнено название товара.")
    price = _number(offer.get("price"))
    if price is None or price <= 0:
        errors.append("Цена должна быть положительным числом.")
    if _text(offer.get("currency") or "RUB").upper() != "RUB":
        errors.append("Для импорта в расчёт нужна цена в RUB.")
    if not offer.get("observed_at"):
        warnings.append("Не указано время получения цены.")
    if not any(identity.get(key) for key in ("gtin", "mpn", "model")):
        warnings.append("Нет GTIN, MPN или модели для надёжного объединения источников.")
    federation = _mapping(item.get("federation"))
    price_summary = _mapping(item.get("price_summary"))
    if federation.get("identity_conflicts"):
        warnings.append("Источники расходятся по идентификаторам товара.")
    if federation.get("category_conflicts"):
        warnings.append("Источники расходятся по категории товара.")
    if price_summary.get("freshness") == "stale":
        warnings.append("Эффективная цена старше 90 дней.")
    for warning in list(review.get("warnings") or []) + list(
        parser_metadata.get("parse_warnings") or []
    ):
        if warning:
            warnings.append(f"Диагностика источника: {warning}")

    target = _TARGET_BY_CATEGORY.get(category)
    if target is None:
        errors.append(
            f"Категория '{category or 'не указана'}' пока не преобразуется в готовое устройство ТО."
        )
    component_type = target[1] if target else ""
    if component_type in {"server", "workstation"}:
        for field in ("ram_gb", "cpu_cores", "storage_gb", "max_power_watts"):
            if metrics.get(field) in (None, ""):
                warnings.append(f"Не заполнена вычислительная метрика {field}.")
    if component_type == "network_device":
        if metrics.get("max_power_watts") in (None, ""):
            warnings.append("Не заполнена мощность сетевого устройства.")
        if not any(
            metrics.get(field) not in (None, "")
            for field in ("lan_ports", "lan_speed_mbps", "wifi_total_mbps")
        ):
            warnings.append("Не заполнены сетевые порты или скорости.")
    return errors, warnings


def validate_staging_record(record: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """Validate the effective item, explicit runtime target and quantity inputs."""

    item = _mapping(record.get("catalog_item"))
    errors, warnings = validate_staging_item(item)
    category_error = next(
        (
            message
            for message in errors
            if message.startswith("Категория '") and "готовое устройство ТО" in message
        ),
        None,
    )
    target_category = _text(record.get("target_category"))
    component_type = _text(record.get("target_component_type"))
    allowed_types = _ALLOWED_TARGET_TYPES.get(target_category, ())
    if target_category and component_type in allowed_types:
        if category_error:
            errors.remove(category_error)
            warnings.append(
                "Назначение ТО задано вручную и не совпадает с исходной категорией."
            )
    else:
        if category_error is None:
            errors.append("Не выбрано допустимое назначение записи в ТО.")

    runtime_inputs = _mapping(record.get("runtime_inputs"))
    quantity = _number(runtime_inputs.get("quantity"))
    if quantity is None or quantity <= 0:
        errors.append("Количество должно быть положительным числом.")
    elif not float(quantity).is_integer():
        errors.append("Количество оборудования должно быть целым числом.")
    if component_type == "workstation":
        client_seats = _number(runtime_inputs.get("client_seats"))
        if client_seats is None or client_seats < 0:
            errors.append("Количество рабочих мест должно быть неотрицательным числом.")
        elif not float(client_seats).is_integer():
            errors.append("Количество рабочих мест должно быть целым числом.")

    # Re-evaluate profile warnings against an explicitly corrected target.
    metrics = _mapping(item.get("attributes"))
    for field in _METRIC_FIELDS:
        value = metrics.get(field)
        if value in (None, ""):
            continue
        if field == "ipv6_support":
            if not isinstance(value, bool):
                errors.append("IPv6 должен быть указан как да/нет.")
        elif _number(value) is None:
            errors.append(f"Метрика {field} должна быть числом.")
    warnings = [
        message
        for message in warnings
        if not message.startswith("Не заполнена вычислительная метрика")
        and message not in {
            "Не заполнена мощность сетевого устройства.",
            "Не заполнены сетевые порты или скорости.",
        }
    ]
    if component_type in {"server", "workstation"}:
        for field in ("ram_gb", "cpu_cores", "storage_gb", "max_power_watts"):
            if metrics.get(field) in (None, ""):
                warnings.append(f"Не заполнена вычислительная метрика {field}.")
    if component_type == "network_device":
        if metrics.get("max_power_watts") in (None, ""):
            warnings.append("Не заполнена мощность сетевого устройства.")
        if not any(
            metrics.get(field) not in (None, "")
            for field in ("lan_ports", "lan_speed_mbps", "wifi_total_mbps")
        ):
            warnings.append("Не заполнены сетевые порты или скорости.")
    if record.get("source_changed_since_review"):
        warnings.append(
            "Источник изменился после импорта; данные ТО не обновлены."
            if record.get("status") == STAGING_IMPORTED
            else "Источник изменился; требуется повторное подтверждение."
        )
    return _unique(errors), _unique(warnings)


def staging_record_readiness(record: Mapping[str, Any]) -> str:
    if record.get("validation_errors"):
        return "blocked"
    if record.get("status") == STAGING_IMPORTED:
        return "stale" if record.get("source_changed_since_review") else "ga_ready"
    if record.get("status") == STAGING_APPROVED:
        return "import_ready"
    return "review"


def catalog_item_to_runtime_row(record: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Convert one approved staging record to a runtime technical alternative."""

    if record.get("status") != STAGING_APPROVED:
        raise ValueError("Импортировать можно только подтверждённую запись.")
    if record.get("validation_errors"):
        raise ValueError("Запись содержит блокирующие ошибки.")

    item = _mapping(record.get("catalog_item"))
    target_category = _text(record.get("target_category"))
    component_type = _text(record.get("target_component_type"))
    if component_type not in _ALLOWED_TARGET_TYPES.get(target_category, ()):
        raise ValueError("Запись не имеет допустимого назначения в ТО.")
    offer = _mapping(item.get("offer"))
    metrics = _mapping(item.get("attributes"))
    runtime_inputs = _mapping(record.get("runtime_inputs"))
    warnings = list(record.get("validation_warnings") or [])

    row: dict[str, Any] = {
        "name": _text(item.get("title")),
        "quantity": float(runtime_inputs.get("quantity") or 1.0),
        "price": float(offer.get("price") or 0.0),
        "scope": "technical",
        "component_type": component_type,
        "origin": "catalog",
        "catalog_item_id": _text(item.get("item_id")),
        "catalog_metadata": {
            "source": item.get("source"),
            "source_product_id": item.get("source_product_id"),
            "identity": deepcopy(_mapping(item.get("identity"))),
            "offer": deepcopy(offer),
            "offers": deepcopy(item.get("offers") if isinstance(item.get("offers"), list) else []),
            "price_summary": deepcopy(_mapping(item.get("price_summary"))),
            "federation": deepcopy(_mapping(item.get("federation"))),
            "specification_sources": deepcopy(
                item.get("specification_sources")
                if isinstance(item.get("specification_sources"), list)
                else []
            ),
            "specification_summary": deepcopy(_mapping(item.get("specification_summary"))),
            "procurement_benchmark": deepcopy(_mapping(item.get("procurement_benchmark"))),
            "field_provenance": deepcopy(_mapping(item.get("field_provenance"))),
            "review": deepcopy(_mapping(item.get("review"))),
            "parser_metadata": deepcopy(_mapping(item.get("parser_metadata"))),
            "staging_id": record.get("staging_id"),
            "manual_overrides": deepcopy(_mapping(record.get("manual_overrides"))),
        },
    }
    for field in _METRIC_FIELDS:
        if field in metrics and metrics[field] not in (None, ""):
            row[field] = metrics[field]
    if "max_power_watts" in row:
        row["max_power"] = row["max_power_watts"]
    if component_type == "workstation":
        row["client_seats"] = float(runtime_inputs.get("client_seats") or 0.0)
    elif component_type == "peripheral":
        row["client_seats"] = 0
    if warnings:
        row["metric_warnings"] = _unique(warnings)
    return target_category, row


class CatalogStagingService:
    """Persist imported catalog rows until the user explicitly approves them."""

    def __init__(
        self,
        staging_path: str | Path,
        storage: JsonFileStorage | None = None,
    ) -> None:
        self.path = Path(staging_path)
        self.storage = storage or JsonFileStorage()
        self.read_model = CatalogStagingReadModel(self.path)
        self._records_cache: list[dict[str, Any]] | None = None
        self._record_index: dict[str, dict[str, Any]] = {}
        self._cache_stamp: tuple[int, int] | None = None
        self._source_path_cache = ""

    def _file_stamp(self) -> tuple[int, int] | None:
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return None
        return (stat.st_mtime_ns, stat.st_size)

    def _set_cache(
        self,
        records: list[dict[str, Any]],
        *,
        source_path: str | Path | None = None,
    ) -> list[dict[str, Any]]:
        self._records_cache = records
        self._record_index = {
            _text(record.get("staging_id")): record
            for record in records
            if _text(record.get("staging_id"))
        }
        if source_path is not None:
            self._source_path_cache = str(source_path)
        self._cache_stamp = self._file_stamp()
        return records

    def list_records(self) -> list[dict[str, Any]]:
        stamp = self._file_stamp()
        if self._records_cache is not None and stamp == self._cache_stamp:
            return self._records_cache
        if stamp is None:
            self._source_path_cache = ""
            return self._set_cache([])
        payload = self.storage.read(self.path)
        records = payload.get("records", []) if isinstance(payload, Mapping) else []
        self._source_path_cache = (
            str(payload.get("source_path") or "") if isinstance(payload, Mapping) else ""
        )
        upgraded = [
            _upgrade_staging_record(record)
            for record in records
            if isinstance(record, Mapping)
        ]
        return self._set_cache(upgraded, source_path=self._source_path_cache)

    def _ensure_read_model(self) -> bool:
        """Make the disposable UI projection available when staging exists.

        Normal large imports build the projection inside their worker process.
        A legacy/stale JSON therefore pays at most one migration read; all
        subsequent GUI page/search/summary operations stay inside SQLite.
        """

        if self.read_model.is_fresh():
            return True
        if not self.path.is_file():
            return False
        records = self.list_records()
        compact = [_compact_staging_record(record) for record in records]
        try:
            self.read_model.rebuild(records, compact_records=compact)
        except (OSError, RuntimeError) as exc:
            logger.warning("Не удалось построить UI read-model каталога: %s", exc)
            return False
        return self.read_model.is_fresh()

    def get_record(self, staging_id: str) -> dict[str, Any]:
        stamp = self._file_stamp()
        if self._records_cache is not None and stamp == self._cache_stamp:
            try:
                return self._record_index[str(staging_id)]
            except KeyError as exc:
                raise KeyError(staging_id) from exc
        if self._ensure_read_model():
            raw = self.read_model.compact_record(str(staging_id))
            if isinstance(raw, Mapping):
                return _upgrade_staging_record(raw)
        self.list_records()
        try:
            return self._record_index[str(staging_id)]
        except KeyError as exc:
            raise KeyError(staging_id) from exc

    def page_projection(
        self,
        *,
        status_filter: str = "all",
        query: str = "",
        offset: int = 0,
        limit: int = 250,
    ) -> tuple[list[dict[str, Any]], int]:
        """Return a lightweight page for UI without loading all staging records."""

        if self._ensure_read_model():
            return self.read_model.page(
                status_filter=status_filter,
                query=query,
                offset=offset,
                limit=limit,
            )
        # Safe compatibility fallback for unavailable/corrupt projection.
        records, total = self.page_records(
            status_filter=status_filter,
            query=query,
            offset=offset,
            limit=limit,
        )
        projections: list[dict[str, Any]] = []
        for record in records:
            item = _mapping(record.get("catalog_item"))
            offer = _mapping(item.get("offer"))
            federation = _mapping(item.get("federation"))
            projections.append(
                {
                    "staging_id": _text(record.get("staging_id")),
                    "status": _text(record.get("status")),
                    "readiness": staging_record_readiness(record),
                    "title": _text(item.get("title")),
                    "source": _text(item.get("source")),
                    "source_count": int(federation.get("source_count") or 0),
                    "category": _text(item.get("category")),
                    "target_category": _text(record.get("target_category")),
                    "price": float(offer.get("price") or 0.0),
                    "issues": len(record.get("validation_errors") or [])
                    + len(record.get("validation_warnings") or []),
                }
            )
        return projections, total

    def page_records(
        self,
        *,
        status_filter: str = "all",
        query: str = "",
        offset: int = 0,
        limit: int = 250,
    ) -> tuple[list[dict[str, Any]], int]:
        """Return one UI page while keeping the full computation catalog available."""

        needle = str(query or "").strip().casefold()
        start = max(0, int(offset))
        page_limit = max(1, min(5000, int(limit)))
        page: list[dict[str, Any]] = []
        total = 0
        for record in self.list_records():
            if status_filter != "all" and record.get("status") != status_filter:
                continue
            item = _mapping(record.get("catalog_item"))
            if needle:
                haystack = " ".join(
                    (
                        _text(item.get("title")),
                        _text(item.get("source")),
                        _text(item.get("category")),
                        _text(_mapping(item.get("identity")).get("brand")),
                        _text(_mapping(item.get("identity")).get("model")),
                        _text(_mapping(item.get("identity")).get("mpn")),
                    )
                ).casefold()
                if needle not in haystack:
                    continue
            if total >= start and len(page) < page_limit:
                page.append(record)
            total += 1
        return page, total

    def summary_counts(self) -> dict[str, int]:
        base = {
            "total": 0,
            STAGING_PENDING: 0,
            STAGING_APPROVED: 0,
            STAGING_BLOCKED: 0,
            STAGING_IMPORTED: 0,
            STAGING_REJECTED: 0,
            "ready": 0,
        }
        if self._ensure_read_model():
            base.update(self.read_model.summary_counts())
            return base
        for record in self.list_records():
            base["total"] += 1
            status = _text(record.get("status"))
            if status in base:
                base[status] += 1
            if staging_record_readiness(record) in {"import_ready", "ga_ready"}:
                base["ready"] += 1
        return base

    def stage_file(
        self,
        source_path: str | Path,
        *,
        source_context: Mapping[str, Any] | None = None,
        max_rows: int | None = None,
        progress: Any | None = None,
    ) -> list[dict[str, Any]]:
        """Refresh one source and rebuild the multi-source federated staging catalog.

        Parsing is streaming for CSV/XLSX.  ``max_rows`` is optional and exists
        for quick demonstrations; the default processes every alternative.
        """

        source = Path(source_path)
        emit = progress or (lambda _message: None)
        incoming: list[dict[str, Any]] = []
        for index, raw in enumerate(iter_catalog_rows(source, max_rows=max_rows)):
            incoming.append(
                normalize_catalog_item(
                    raw,
                    source_path=source,
                    row_index=index,
                    source_context=source_context,
                )
            )
            if index and index % 1000 == 0:
                emit(f"Нормализовано: {index + 1}")

        previous_records = self.list_records()
        refresh_sources = {
            _text(item.get("source"))
            for item in incoming
            if _text(item.get("source"))
        }
        context = _mapping(source_context)
        context_source = _text(context.get("id") or context.get("source_id"))
        if context_source:
            refresh_sources.add(context_source)
        if not refresh_sources:
            refresh_sources.add(source.stem or "imported")

        observations: list[dict[str, Any]] = []
        for record in previous_records:
            for observation in source_observations(_mapping(record.get("source_catalog_item"))):
                if _text(observation.get("source")) not in refresh_sources:
                    observations.append(observation)
        observations.extend(incoming)
        emit(f"Федерация: {len(observations)} наблюдений")

        federated_items = federate_catalog_items(observations)
        # The federated result owns everything needed from here on.  Release
        # the source-normalization buffers before constructing review records.
        del observations
        del incoming
        previous_index = _build_previous_record_index(previous_records)
        records: list[dict[str, Any]] = []
        used_previous: set[str] = set()
        for index, item in enumerate(federated_items):
            old = _best_previous_record_indexed(item, previous_index, used_previous)
            if old:
                previous_item = _mapping(old.get("source_catalog_item"))
                item = carry_specification_sources(item, previous_item)
                previous_benchmark = _mapping(previous_item.get("procurement_benchmark"))
                if previous_benchmark and not _mapping(item.get("procurement_benchmark")):
                    carried_benchmark = deepcopy(previous_benchmark)
                    carried_benchmark["needs_refresh"] = True
                    item["procurement_benchmark"] = carried_benchmark
                    provenance = _mapping(item.get("field_provenance"))
                    previous_provenance = _mapping(previous_item.get("field_provenance"))
                    if previous_provenance.get("procurement_benchmark"):
                        provenance["procurement_benchmark"] = deepcopy(
                            previous_provenance["procurement_benchmark"]
                        )
                    item["field_provenance"] = provenance
                if _text(previous_item.get("item_id")):
                    item["item_id"] = _text(previous_item.get("item_id"))
                staging_id = _text(old.get("staging_id")) or _staging_id(item)
                used_previous.add(staging_id)
            else:
                staging_id = _staging_id(item)
            records.append(
                _build_staging_record(
                    item,
                    staging_id=staging_id,
                    previous=old,
                )
            )
            if index and index % 2000 == 0:
                emit(f"Staging: {index + 1}/{len(federated_items)}")
        self._save(records, source_path=source)
        emit(f"Готово: {len(records)} позиций")
        return records

    def set_status(self, staging_id: str, status: str) -> dict[str, Any]:
        if status not in {STAGING_PENDING, STAGING_APPROVED, STAGING_REJECTED}:
            raise ValueError(f"Недопустимый статус staging: {status}")
        records = self.list_records()
        for record in records:
            if record.get("staging_id") != staging_id:
                continue
            if status == STAGING_APPROVED and record.get("validation_errors"):
                raise ValueError("Нельзя подтвердить запись с блокирующими ошибками.")
            if record.get("status") == STAGING_IMPORTED:
                raise ValueError("Импортированную запись нельзя вернуть в staging.")
            record["status"] = status
            self._save(records)
            return dict(deepcopy(record))
        raise KeyError(staging_id)

    def set_status_many(self, staging_ids: Iterable[str], status: str) -> dict[str, int]:
        if status not in {STAGING_PENDING, STAGING_APPROVED, STAGING_REJECTED}:
            raise ValueError(f"Недопустимый статус staging: {status}")
        selected = set(staging_ids)
        records = self.list_records()
        result = {"updated": 0, "blocked": 0, "skipped": 0}
        for record in records:
            if record.get("staging_id") not in selected:
                continue
            if record.get("status") == STAGING_IMPORTED:
                result["skipped"] += 1
                continue
            if status == STAGING_APPROVED and record.get("validation_errors"):
                result["blocked"] += 1
                continue
            record["status"] = status
            result["updated"] += 1
        self._save(records)
        return result

    def update_record(
        self,
        staging_id: str,
        values: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Apply user corrections, recalculate validation and require re-approval."""

        records = self.list_records()
        for record in records:
            if record.get("staging_id") != staging_id:
                continue
            if record.get("status") == STAGING_IMPORTED:
                raise ValueError("Импортированная запись заблокирована для редактирования.")
            source_item = deepcopy(_mapping(record.get("source_catalog_item")))
            item = deepcopy(_mapping(record.get("catalog_item")))
            item["title"] = _text(values.get("title"))
            item["category"] = _text(values.get("category")).lower()

            offer = _mapping(item.get("offer"))
            offer["price"] = _number(values.get("price"))
            offer["currency"] = _text(values.get("currency") or "RUB").upper()
            item["offer"] = offer

            identity = _mapping(item.get("identity"))
            identity_values = _mapping(values.get("identity"))
            for field in ("brand", "model", "mpn", "gtin"):
                if field not in identity_values:
                    continue
                value = _text(identity_values.get(field))
                if value:
                    identity[field] = value
                else:
                    identity.pop(field, None)
            item["identity"] = identity

            attributes = _mapping(item.get("attributes"))
            metric_values = _mapping(values.get("attributes"))
            for field in _METRIC_FIELDS:
                value = _metric_field_value(field, metric_values.get(field))
                if value is None:
                    attributes.pop(field, None)
                else:
                    attributes[field] = value
            item["attributes"] = attributes

            target_category = _text(values.get("target_category"))
            component_type = _text(values.get("target_component_type"))
            runtime_inputs = {
                "quantity": _number(values.get("quantity")),
                "client_seats": _number(values.get("client_seats")),
            }
            record.update(
                {
                    "catalog_item": item,
                    "target_category": target_category,
                    "target_component_type": component_type,
                    "runtime_inputs": runtime_inputs,
                    "manual_overrides": _build_manual_overrides(
                        source_item,
                        item,
                        target_category=target_category,
                        target_component_type=component_type,
                        runtime_inputs=runtime_inputs,
                    ),
                    "manual_updated_at": datetime.now(UTC).isoformat(),
                }
            )
            item_provenance = _mapping(item.get("field_provenance"))
            item_provenance["manual"] = {
                "updated_at": record["manual_updated_at"],
                "fields": sorted(record["manual_overrides"]),
            }
            item["field_provenance"] = item_provenance
            errors, warnings = validate_staging_record(record)
            record["validation_errors"] = errors
            record["validation_warnings"] = warnings
            record["status"] = STAGING_BLOCKED if errors else STAGING_PENDING
            self._save(records)
            return dict(deepcopy(record))
        raise KeyError(staging_id)

    def transform_source_items(
        self,
        transform: Callable[[Mapping[str, Any]], Mapping[str, Any] | None],
        *,
        staging_ids: Iterable[str] | None = None,
    ) -> int:
        """Transform source items in place and persist once.

        Large enrichment jobs must not build a second ``staging_id -> item``
        dictionary containing tens of thousands of deep catalog objects.  The
        callback receives one staging record at a time; returning ``None``
        leaves it unchanged, while a mapping replaces only that source item.
        Review state and manual overrides are reconstructed through the normal
        staging builder.
        """

        selected = {str(value) for value in (staging_ids or []) if str(value)}
        records = self.list_records()
        changed = 0
        for index, record in enumerate(records):
            staging_id = _text(record.get("staging_id"))
            if selected and staging_id not in selected:
                continue
            if record.get("status") == STAGING_IMPORTED:
                continue
            source_item = transform(record)
            if not isinstance(source_item, Mapping):
                continue
            records[index] = _build_staging_record(
                source_item,
                staging_id=staging_id,
                previous=record,
            )
            changed += 1
        if changed:
            self._save(records)
        return changed

    def apply_source_item_updates(
        self,
        updates: Mapping[str, Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Replace enriched source items while preserving review/manual overrides."""

        if not updates:
            return self.list_records()
        records = self.list_records()
        changed = False
        for index, record in enumerate(records):
            staging_id = _text(record.get("staging_id"))
            source_item = updates.get(staging_id)
            if not isinstance(source_item, Mapping):
                continue
            if record.get("status") == STAGING_IMPORTED:
                continue
            records[index] = _build_staging_record(
                source_item,
                staging_id=staging_id,
                previous=record,
            )
            changed = True
        if changed:
            self._save(records)
        return self.list_records()

    def mark_imported(self, staging_ids: Iterable[str]) -> None:
        imported = set(staging_ids)
        records = self.list_records()
        timestamp = datetime.now(UTC).isoformat()
        for record in records:
            if record.get("staging_id") in imported:
                record["status"] = STAGING_IMPORTED
                record["imported_at"] = timestamp
        self._save(records)

    def approved_records(self) -> list[dict[str, Any]]:
        return [
            record
            for record in self.list_records()
            if record.get("status") == STAGING_APPROVED
        ]

    def _save(
        self,
        records: list[dict[str, Any]],
        *,
        source_path: str | Path | None = None,
    ) -> None:
        resolved_source = str(source_path or self._source_path_cache or "")
        compact_records = [_compact_staging_record(record) for record in records]
        payload = {
            "schema_version": CATALOG_STAGING_SCHEMA_VERSION,
            "updated_at": datetime.now(UTC).isoformat(),
            "source_path": resolved_source,
            # Runtime/validation fields are reproducible and therefore are not
            # duplicated on disk.  This cuts large staging files by multiples.
            "records": compact_records,
        }
        writer = getattr(self.storage, "write_compact", None)
        if callable(writer):
            writer(self.path, payload)
        else:
            self.storage.write(self.path, payload)
        try:
            self.read_model.rebuild(records, compact_records=compact_records)
        except (OSError, RuntimeError) as exc:
            # Canonical JSON is authoritative; a projection failure must not
            # corrupt or roll back imported catalog data.  UI falls back and
            # can rebuild the projection on the next read.
            logger.warning("Не удалось обновить UI read-model каталога: %s", exc)
        self._set_cache(records, source_path=resolved_source)



def _upgrade_staging_record(raw: Mapping[str, Any]) -> dict[str, Any]:
    record = dict(deepcopy(raw))
    stored_item = _mapping(record.get("catalog_item"))
    source_item = _mapping(record.get("source_catalog_item")) or deepcopy(stored_item)
    overrides = _mapping(record.get("manual_overrides"))
    if stored_item:
        item = stored_item
    elif overrides:
        item = _apply_manual_overrides(source_item, overrides)
    else:
        item = source_item
    record["source_catalog_item"] = source_item
    record["catalog_item"] = item
    default_target = target_for_catalog_category(_text(item.get("category")))
    record.setdefault("target_category", default_target[0] if default_target else "")
    record.setdefault("target_component_type", default_target[1] if default_target else "")
    record.setdefault(
        "runtime_inputs",
        _default_runtime_inputs(_text(record.get("target_component_type"))),
    )
    record.setdefault("manual_overrides", overrides)
    if overrides and record.get("manual_updated_at"):
        provenance = _mapping(item.get("field_provenance"))
        provenance.setdefault(
            "manual",
            {
                "updated_at": record.get("manual_updated_at"),
                "fields": sorted(overrides),
            },
        )
        item["field_provenance"] = provenance
        record["catalog_item"] = item
    errors, warnings = validate_staging_record(record)
    record["validation_errors"] = errors
    record["validation_warnings"] = warnings
    if record.get("status") != STAGING_IMPORTED:
        if errors:
            record["status"] = STAGING_BLOCKED
        elif record.get("status") == STAGING_BLOCKED:
            record["status"] = STAGING_PENDING
    return record


def _compact_staging_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only source state + user decisions; derived fields are rebuilt on load."""

    keys = (
        "staging_id",
        "status",
        "source_catalog_item",
        "target_category",
        "target_component_type",
        "runtime_inputs",
        "manual_overrides",
        "manual_updated_at",
        "imported_at",
        "source_changed_since_review",
    )
    return {
        key: record.get(key)
        for key in keys
        if record.get(key) not in (None, {}, [])
    }


def _build_staging_record(
    source_item: Mapping[str, Any],
    *,
    staging_id: str,
    previous: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    old = _upgrade_staging_record(previous or {}) if previous else {}
    overrides = _mapping(old.get("manual_overrides"))
    source_copy = dict(deepcopy(source_item))
    item = _apply_manual_overrides(source_copy, overrides) if overrides else source_copy
    default_target = target_for_catalog_category(_text(item.get("category")))
    target_category = _text(
        overrides.get("target_category")
        or (default_target[0] if default_target else "")
    )
    target_component_type = _text(
        overrides.get("target_component_type")
        or (default_target[1] if default_target else "")
    )
    default_inputs = _default_runtime_inputs(target_component_type)
    runtime_inputs = {
        **default_inputs,
        **_mapping(overrides.get("runtime_inputs")),
    }
    previous_source = _mapping(old.get("source_catalog_item"))
    source_changed = bool(previous_source and previous_source != dict(source_item))
    status = str(old.get("status") or STAGING_PENDING)
    if source_changed and status == STAGING_APPROVED:
        status = STAGING_PENDING
    record = {
        "staging_id": staging_id,
        "status": status,
        "source_catalog_item": source_copy,
        "catalog_item": item,
        "target_category": target_category,
        "target_component_type": target_component_type,
        "runtime_inputs": runtime_inputs,
        "manual_overrides": overrides,
        "manual_updated_at": old.get("manual_updated_at"),
        "imported_at": old.get("imported_at"),
        "source_changed_since_review": source_changed,
    }
    errors, warnings = validate_staging_record(record)
    record["validation_errors"] = errors
    record["validation_warnings"] = warnings
    if record["status"] != STAGING_IMPORTED:
        if errors:
            record["status"] = STAGING_BLOCKED
        elif record["status"] == STAGING_BLOCKED:
            record["status"] = STAGING_PENDING
    return record


def _default_runtime_inputs(component_type: str) -> dict[str, float | None]:
    return {
        "quantity": 1.0,
        "client_seats": 1.0 if component_type == "workstation" else 0.0,
    }


def _apply_manual_overrides(
    source_item: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    item = dict(deepcopy(source_item))
    for field in ("title", "category"):
        if field in overrides:
            item[field] = overrides[field]
    if "offer" in overrides:
        item["offer"] = {**_mapping(item.get("offer")), **_mapping(overrides.get("offer"))}
    if "identity" in overrides:
        identity = _mapping(item.get("identity"))
        for field, value in _mapping(overrides.get("identity")).items():
            if value in (None, ""):
                identity.pop(field, None)
            else:
                identity[field] = value
        item["identity"] = identity
    if "attributes" in overrides:
        attributes = _mapping(item.get("attributes"))
        for field, value in _mapping(overrides.get("attributes")).items():
            if value is None:
                attributes.pop(field, None)
            else:
                attributes[field] = value
        item["attributes"] = attributes
    return item


def _build_manual_overrides(
    source_item: Mapping[str, Any],
    item: Mapping[str, Any],
    *,
    target_category: str,
    target_component_type: str,
    runtime_inputs: Mapping[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in ("title", "category"):
        if item.get(field) != source_item.get(field):
            result[field] = item.get(field)

    source_offer = _mapping(source_item.get("offer"))
    current_offer = _mapping(item.get("offer"))
    offer_diff = {
        field: current_offer.get(field)
        for field in ("price", "currency")
        if current_offer.get(field) != source_offer.get(field)
    }
    if offer_diff:
        result["offer"] = offer_diff

    source_identity = _mapping(source_item.get("identity"))
    current_identity = _mapping(item.get("identity"))
    identity_diff = {
        field: current_identity.get(field)
        for field in ("brand", "model", "mpn", "gtin")
        if current_identity.get(field) != source_identity.get(field)
    }
    if identity_diff:
        result["identity"] = identity_diff

    source_attributes = _mapping(source_item.get("attributes"))
    current_attributes = _mapping(item.get("attributes"))
    attribute_diff = {
        field: current_attributes.get(field)
        for field in _METRIC_FIELDS
        if current_attributes.get(field) != source_attributes.get(field)
    }
    if attribute_diff:
        result["attributes"] = attribute_diff

    default_target = target_for_catalog_category(_text(item.get("category")))
    default_category = default_target[0] if default_target else ""
    default_type = default_target[1] if default_target else ""
    if target_category != default_category:
        result["target_category"] = target_category
    if target_component_type != default_type:
        result["target_component_type"] = target_component_type

    default_inputs = _default_runtime_inputs(target_component_type)
    input_diff = {
        field: runtime_inputs.get(field)
        for field in ("quantity", "client_seats")
        if runtime_inputs.get(field) != default_inputs.get(field)
    }
    if input_diff:
        result["runtime_inputs"] = input_diff
    return result


def _load_csv_rows(path: Path) -> list[dict[str, Any]]:
    return list(_iter_csv_rows(path))


def _iter_csv_rows(
    path: Path,
    *,
    max_rows: int | None = None,
) -> Iterator[dict[str, Any]]:
    with path.open("rb") as raw_file:
        sample_bytes = raw_file.read(65536)
    encoding = None
    sample_text = ""
    for candidate in ("utf-8-sig", "cp1251"):
        try:
            sample_text = sample_bytes.decode(candidate)
            encoding = candidate
            break
        except UnicodeDecodeError:
            continue
    if encoding is None:
        raise ValueError("CSV должен быть в UTF-8 или Windows-1251.")
    try:
        dialect = csv.Sniffer().sniff(sample_text[:4096], delimiters=",;\t")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = ";"
    with path.open("r", encoding=encoding, newline="") as stream:
        matrix = csv.reader(stream, delimiter=delimiter)
        yield from _iter_matrix_rows(matrix, max_rows=max_rows)


def _load_xlsx_rows(path: Path) -> list[dict[str, Any]]:
    """Compatibility wrapper around the streaming OOXML reader."""

    return list(_iter_xlsx_rows(path))


def _iter_xlsx_rows(
    path: Path,
    *,
    max_rows: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Read the first XLSX sheet without expanding the whole XML in memory."""

    main_ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    rel_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    package_rel_ns = "http://schemas.openxmlformats.org/package/2006/relationships"
    with zipfile.ZipFile(path) as archive:
        shared = _read_xlsx_shared_strings(archive, main_ns)
        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        first_sheet = workbook.find(f".//{{{main_ns}}}sheet")
        if first_sheet is None:
            return
        relationship_id = first_sheet.attrib.get(f"{{{rel_ns}}}id")
        rels = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        target = None
        for relation in rels.findall(f"{{{package_rel_ns}}}Relationship"):
            if relation.attrib.get("Id") == relationship_id:
                target = relation.attrib.get("Target")
                break
        if not target:
            raise ValueError("Не удалось найти первый лист XLSX.")
        target_path = str(target).lstrip("/")
        sheet_path = posixpath.normpath(
            target_path if target_path.startswith("xl/") else f"xl/{target_path}"
        )
        if sheet_path not in archive.namelist():
            raise ValueError("Первый лист XLSX отсутствует в архиве.")

        def matrix_rows() -> Iterator[list[Any]]:
            with archive.open(sheet_path) as sheet_stream:
                for _event, row in ElementTree.iterparse(sheet_stream, events=("end",)):
                    if _local_name(row.tag) != "row":
                        continue
                    values: dict[int, Any] = {}
                    for cell in list(row):
                        if _local_name(cell.tag) != "c":
                            continue
                        reference = cell.attrib.get("r", "")
                        column = _xlsx_column_index(reference)
                        if column < 0 or column > 511:
                            continue
                        cell_type = cell.attrib.get("t")
                        value_node = next(
                            (child for child in list(cell) if _local_name(child.tag) == "v"),
                            None,
                        )
                        if cell_type == "inlineStr":
                            value = "".join(
                                node.text or ""
                                for node in cell.iter()
                                if _local_name(node.tag) == "t"
                            )
                        else:
                            raw_value = value_node.text if value_node is not None else ""
                            if cell_type == "s" and raw_value:
                                try:
                                    value = shared[int(raw_value)]
                                except (ValueError, IndexError):
                                    value = ""
                            elif cell_type == "b":
                                value = raw_value == "1"
                            else:
                                value = raw_value
                        values[column] = value
                    if values:
                        yield [values.get(index, "") for index in range(max(values) + 1)]
                    row.clear()

        yield from _iter_matrix_rows(matrix_rows(), max_rows=max_rows)


def _read_xlsx_shared_strings(archive: zipfile.ZipFile, main_ns: str) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    result: list[str] = []
    with archive.open("xl/sharedStrings.xml") as stream:
        for _event, node in ElementTree.iterparse(stream, events=("end",)):
            if _local_name(node.tag) != "si":
                continue
            result.append(
                "".join(
                    child.text or ""
                    for child in node.iter()
                    if _local_name(child.tag) == "t"
                )
            )
            node.clear()
    return result


def _iter_matrix_rows(
    matrix: Iterable[Iterable[Any]],
    *,
    max_rows: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Resolve a supplier header from a tiny buffer, then stream data rows."""

    iterator = iter(matrix)
    buffer: list[list[Any]] = []
    while len(buffer) < 25:
        try:
            row = list(next(iterator))
        except StopIteration:
            break
        if any(_text(value) for value in row):
            buffer.append(row)
    if not buffer:
        return
    header_index = _find_header_row(buffer)
    headers = [_text(value) for value in buffer[header_index]]
    emitted = 0

    def payload_for(row: Iterable[Any]) -> dict[str, Any]:
        values = list(row)
        return {
            header: values[index] if index < len(values) else ""
            for index, header in enumerate(headers)
            if header
        }

    for row in buffer[header_index + 1 :]:
        payload = payload_for(row)
        if any(value not in ("", None) for value in payload.values()):
            yield payload
            emitted += 1
            if max_rows is not None and emitted >= max_rows:
                return
    for raw_row in iterator:
        row = list(raw_row)
        if not any(_text(value) for value in row):
            continue
        payload = payload_for(row)
        if not any(value not in ("", None) for value in payload.values()):
            continue
        yield payload
        emitted += 1
        if max_rows is not None and emitted >= max_rows:
            return


def _load_yml_rows(path: Path) -> list[dict[str, Any]]:
    """Read Yandex Market Language or generic XML offer lists."""

    try:
        root = ElementTree.fromstring(path.read_bytes())
    except ElementTree.ParseError as exc:
        raise ValueError(f"Не удалось разобрать XML/YML: {exc}") from exc

    categories: dict[str, str] = {}
    for node in root.findall(".//category"):
        category_id = str(node.attrib.get("id") or "").strip()
        label = _text(node.text)
        if category_id and label:
            categories[category_id] = label

    offers = root.findall(".//offer")
    if not offers and _local_name(root.tag) == "offer":
        offers = [root]
    rows: list[dict[str, Any]] = []
    for offer in offers:
        row: dict[str, Any] = {
            "source_product_id": str(offer.attrib.get("id") or "").strip(),
            "availability": offer.attrib.get("available", "unknown"),
        }
        params: dict[str, Any] = {}
        for child in list(offer):
            key = _local_name(child.tag)
            value = _text(child.text)
            if key == "param":
                param_name = str(child.attrib.get("name") or "").strip()
                if param_name and value:
                    params[param_name] = value
                continue
            if key == "categoryId":
                row["category"] = categories.get(value, value)
            elif key in {"name", "model"}:
                row[key] = value
            elif key == "vendor":
                row["brand"] = value
            elif key == "vendorCode":
                row["mpn"] = value
            elif key in {"barcode", "gtin"}:
                row["gtin"] = value
            elif key == "currencyId":
                row["currency"] = value
            elif key in {"price", "url"}:
                row[key] = value
            elif value:
                row[key] = value
        if params:
            row["attributes"] = params
        if not row.get("name") and row.get("model"):
            row["name"] = " ".join(
                part for part in (str(row.get("brand") or "").strip(), str(row.get("model") or "").strip()) if part
            )
        if any(value not in ("", None) for value in row.values()):
            rows.append(row)
    return rows


def _matrix_to_rows(matrix: list[list[Any]]) -> list[dict[str, Any]]:
    matrix = [row for row in matrix if any(_text(value) for value in row)]
    if not matrix:
        return []
    header_index = _find_header_row(matrix)
    headers = [_text(value) for value in matrix[header_index]]
    result: list[dict[str, Any]] = []
    for row in matrix[header_index + 1 :]:
        payload = {
            header: row[index] if index < len(row) else ""
            for index, header in enumerate(headers)
            if header
        }
        if any(value not in ("", None) for value in payload.values()):
            result.append(payload)
    return result


def _find_header_row(matrix: list[list[Any]]) -> int:
    """Pick the first plausible feed header, tolerating title/preamble rows."""

    best_index = 0
    best_score = -1
    for index, row in enumerate(matrix[:25]):
        normalized = {_normalize_key(value) for value in row if _text(value)}
        score = sum(
            1
            for group in _HEADER_ALIAS_GROUPS
            if normalized.intersection(group)
        )
        if score > best_score:
            best_index, best_score = index, score
        if score >= 3:
            return index
    return best_index


def _xlsx_column_index(reference: str) -> int:
    letters = re.match(r"[A-Z]+", reference.upper())
    value = 0
    for char in letters.group(0) if letters else "A":
        value = value * 26 + ord(char) - ord("A") + 1
    return value - 1


_HEADER_ALIAS_GROUPS = (
    {
        "title",
        "name",
        "productname",
        "наименование",
        "наименованиетовара",
        "название",
        "товар",
        "номенклатура",
        "наименованиеноменклатуры",
        "описаниетовара",
    },
    {
        "category",
        "type",
        "producttype",
        "категория",
        "категориятовара",
        "раздел",
        "группа",
        "товарнаягруппа",
        "товарнаякатегория",
        "категорияноменклатуры",
        "разделноменклатуры",
    },
    {
        "price",
        "pricerub",
        "cost",
        "цена",
        "ценаруб",
        "стоимость",
        "розничнаяцена",
        "ценасндс",
        "ценасндсруб",
        "ценапартнера",
        "оптоваяцена",
        "ррц",
    },
    {"brand", "vendor", "бренд", "производитель"},
    {
        "mpn",
        "pn",
        "partnumber",
        "vendorcode",
        "артикулпроизводителя",
        "партномер",
    },
    {"sku", "article", "артикул", "кодтовара", "код"},
)


_BLOCKED_CATEGORY_PATTERNS = (
    ("cpu", ("процессор", "cpu")),
    ("gpu", ("видеокарт", "gpu", "графический ускоритель")),
    ("motherboard", ("материнск", "motherboard")),
    ("ram", ("оперативн", "модул памяти", "ram")),
    ("ssd", ("ssd", "твердотельн")),
    ("hdd", ("жестк", "жёстк", "hdd")),
    ("component", ("комплектующ", "аксессуар для сервер", "серверн комплектующ")),
)

_CATEGORY_PATTERNS = (
    ("workstation", ("рабочая станция", "workstation")),
    ("laptop", ("ноутбук", "laptop")),
    ("prebuilt_pc", ("системный блок", "системные блоки", "моноблок", "desktop", "готовый компьютер")),
    ("router", ("маршрутизатор", "роутер", "router")),
    ("switch", ("коммутатор", "switch")),
    ("access_point", ("точка доступа", "точки доступа", "access point")),
    ("server", ("сервер", "server")),
    ("printer", ("принтер", "мфу", "printer")),
    ("monitor", ("монитор", "monitor")),
)


def _normalize_key(value: Any) -> str:
    return re.sub(r"[^0-9a-zа-я]+", "", _text(value).lower())


def _first_alias(payload: Mapping[str, Any], *keys: str) -> Any:
    normalized = {_normalize_key(key): value for key, value in payload.items()}
    for key in keys:
        value = normalized.get(_normalize_key(key))
        if value not in (None, ""):
            return value
    return None


def _normalize_catalog_category(raw_category: str, title: str) -> str:
    value = str(raw_category or "").strip().lower()
    if value in _TARGET_BY_CATEGORY:
        return value

    # Supplier sections such as "Серверная оперативная память" must stay
    # blocked even if the title also contains a supported-device word.
    for category, patterns in _BLOCKED_CATEGORY_PATTERNS:
        if any(pattern in value for pattern in patterns):
            return category

    # Prefer the concrete product title over a broad supplier section.
    # Example: a system block inside "Компьютеры и ноутбуки" is a PC, not a laptop.
    title_value = str(title or "").strip().lower()
    for category, patterns in _CATEGORY_PATTERNS:
        if any(pattern in title_value for pattern in patterns):
            return category
    for category, patterns in _CATEGORY_PATTERNS:
        if any(pattern in value for pattern in patterns):
            return category
    return value


def _local_name(tag: str) -> str:
    return str(tag).rsplit("}", 1)[-1]


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _mapping_or_json(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value.strip().startswith("{"):
        try:
            payload = json.loads(value)
            return dict(payload) if isinstance(payload, Mapping) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _first(payload: Mapping[str, Any], *keys: str) -> Any:
    lowered = {str(key).strip().lower(): value for key, value in payload.items()}
    for key in keys:
        value = lowered.get(key.lower())
        if value not in (None, ""):
            return value
    return None


def _text(value: Any) -> str:
    return str(value or "").strip()


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    normalized = re.sub(r"[^0-9,.-]", "", str(value)).replace(",", ".")
    try:
        return float(normalized)
    except ValueError:
        return None


def _metric_value(value: Any) -> Any:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    text = _text(value).lower()
    if text in {"true", "yes", "да", "есть", "1"}:
        return True
    if text in {"false", "no", "нет", "0"}:
        return False
    number = _number(value)
    return number if number is not None else value


def _metric_field_value(field: str, value: Any) -> Any:
    if value in (None, ""):
        return None
    if field == "ipv6_support":
        return _metric_value(value)
    number = _number(value)
    return number if number is not None else value



def _build_previous_record_index(
    previous_records: Iterable[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for record in previous_records:
        item = _mapping(record.get("source_catalog_item"))
        for key in observation_keys(item):
            index.setdefault(key, []).append(record)
    return index


def _best_previous_record_indexed(
    item: Mapping[str, Any],
    previous_index: Mapping[str, list[dict[str, Any]]],
    used_previous: set[str],
) -> dict[str, Any]:
    """O(keys*candidates) continuity lookup instead of an O(N²) full scan."""

    keys = observation_keys(item)
    status_rank = {
        STAGING_IMPORTED: 5,
        STAGING_APPROVED: 4,
        STAGING_PENDING: 3,
        STAGING_BLOCKED: 2,
        STAGING_REJECTED: 1,
    }
    unique: dict[str, dict[str, Any]] = {}
    for key in keys:
        for record in previous_index.get(key, []):
            staging_id = _text(record.get("staging_id"))
            if staging_id and staging_id not in used_previous:
                unique[staging_id] = record
    candidates: list[tuple[int, int, str, dict[str, Any]]] = []
    for staging_id, record in unique.items():
        overlap = len(keys & observation_keys(_mapping(record.get("source_catalog_item"))))
        if overlap:
            candidates.append(
                (overlap, status_rank.get(_text(record.get("status")), 0), staging_id, record)
            )
    if not candidates:
        return {}
    return max(candidates, key=lambda value: (value[0], value[1], value[2]))[3]


def _best_previous_record(
    item: Mapping[str, Any],
    previous_records: list[dict[str, Any]],
    used_previous: set[str],
) -> dict[str, Any]:
    """Pick one reviewed record whose source observations survive in this group."""

    keys = observation_keys(item)
    status_rank = {
        STAGING_IMPORTED: 5,
        STAGING_APPROVED: 4,
        STAGING_PENDING: 3,
        STAGING_BLOCKED: 2,
        STAGING_REJECTED: 1,
    }
    candidates: list[tuple[int, int, str, dict[str, Any]]] = []
    for record in previous_records:
        staging_id = _text(record.get("staging_id"))
        if not staging_id or staging_id in used_previous:
            continue
        previous_item = _mapping(record.get("source_catalog_item"))
        overlap = len(keys & observation_keys(previous_item))
        if not overlap:
            continue
        candidates.append(
            (
                overlap,
                status_rank.get(_text(record.get("status")), 0),
                staging_id,
                record,
            )
        )
    if not candidates:
        return {}
    return max(candidates, key=lambda value: (value[0], value[1], value[2]))[3]

def _stable_id(
    source: str,
    source_product_id: str,
    identity: Mapping[str, Any],
    url: str,
    title: str,
    row_index: int,
) -> str:
    identity_value = (
        identity.get("gtin")
        or identity.get("mpn")
        or source_product_id
        or url
        or f"{title}:{row_index}"
    )
    digest = hashlib.sha1(f"{source}:{identity_value}".encode("utf-8")).hexdigest()[:12]
    return f"catalog-{digest}"


def _staging_id(item: Mapping[str, Any]) -> str:
    digest = hashlib.sha1(
        f"{item.get('source')}:{item.get('item_id')}".encode("utf-8")
    ).hexdigest()[:16]
    return f"stage-{digest}"


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value)))


__all__ = [
    "CATALOG_SOURCE_SCHEMA_VERSION",
    "CATALOG_STAGING_SCHEMA_VERSION",
    "CatalogStagingService",
    "STAGING_APPROVED",
    "STAGING_BLOCKED",
    "STAGING_IMPORTED",
    "STAGING_PENDING",
    "STAGING_REJECTED",
    "catalog_metric_fields",
    "catalog_target_options",
    "catalog_item_to_runtime_row",
    "load_catalog_rows",
    "normalize_catalog_item",
    "staging_record_readiness",
    "supported_catalog_extensions",
    "target_for_catalog_category",
    "validate_staging_item",
    "validate_staging_record",
]
