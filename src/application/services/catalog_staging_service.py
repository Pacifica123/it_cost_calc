"""Reviewable import boundary between external catalogs and runtime ТО rows."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import posixpath
import re
import zipfile
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping
from xml.etree import ElementTree

from infrastructure.storage import JsonFileStorage

CATALOG_STAGING_SCHEMA_VERSION = 2
CATALOG_SOURCE_SCHEMA_VERSION = 2

STAGING_PENDING = "pending"
STAGING_APPROVED = "approved"
STAGING_REJECTED = "rejected"
STAGING_BLOCKED = "blocked"
STAGING_IMPORTED = "imported"

_SUPPORTED_EXTENSIONS = {".json", ".csv", ".xlsx"}
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


def load_catalog_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load normalized or flat catalog rows from JSON, CSV or simple XLSX."""

    source_path = Path(path)
    suffix = source_path.suffix.lower()
    if suffix not in _SUPPORTED_EXTENSIONS:
        raise ValueError(f"Неподдерживаемый формат каталога: {suffix or 'без расширения'}")
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
        result = []
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            item = dict(row)
            item.setdefault("_catalog_schema_version", schema_version)
            result.append(item)
        return result
    if suffix == ".csv":
        return _load_csv_rows(source_path)
    return _load_xlsx_rows(source_path)


def normalize_catalog_item(
    raw: Mapping[str, Any],
    *,
    source_path: str | Path,
    row_index: int,
) -> dict[str, Any]:
    """Normalize schema v1/v2 and flat spreadsheet rows to one staging item."""

    item = dict(deepcopy(raw))
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

    title = _text(_first(item, "title", "name", "product_name", "наименование"))
    category = _text(_first(item, "category", "type", "product_type", "категория")).lower()
    source = _text(item.get("source") or Path(source_path).stem or "imported")
    url = _text(offer.get("url") or item.get("url"))
    price = _number(
        _first(
            offer,
            "price",
            "price_rub",
            "cost",
            "цена",
        )
    )
    if price is None:
        price = _number(_first(item, "price_rub", "price", "cost", "цена"))
    currency = _text(offer.get("currency") or item.get("currency") or "RUB").upper()
    availability = _text(
        offer.get("availability") or item.get("availability") or "unknown"
    )
    observed_at = _text(offer.get("observed_at") or item.get("observed_at"))

    normalized_identity = {
        key: value
        for key, value in {
            "brand": _text(identity.get("brand") or item.get("brand")),
            "model": _text(identity.get("model") or item.get("model")),
            "mpn": _text(identity.get("mpn") or item.get("mpn")),
            "gtin": _text(
                identity.get("gtin") or item.get("gtin") or item.get("ean")
            ),
        }.items()
        if value
    }
    normalized_metrics = {
        field: _metric_field_value(field, metrics.get(field, item.get(field)))
        for field in _METRIC_FIELDS
        if _metric_field_value(field, metrics.get(field, item.get(field))) is not None
    }
    source_product_id = _text(item.get("source_product_id"))
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

    return {
        "item_id": catalog_item_id,
        "title": title,
        "category": category,
        "source": source,
        "source_product_id": source_product_id or None,
        "identity": normalized_identity,
        "offer": {
            "price": price,
            "currency": currency,
            "availability": availability,
            "url": url or None,
            "region": _text(offer.get("region") or item.get("region")),
            "observed_at": observed_at or None,
        },
        "attributes": normalized_metrics,
        "field_provenance": _mapping(item.get("field_provenance")),
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

    def list_records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        payload = self.storage.read(self.path)
        records = payload.get("records", []) if isinstance(payload, Mapping) else []
        return [
            _upgrade_staging_record(record)
            for record in records
            if isinstance(record, Mapping)
        ]

    def stage_file(self, source_path: str | Path) -> list[dict[str, Any]]:
        source = Path(source_path)
        rows = load_catalog_rows(source)
        previous = {record.get("staging_id"): record for record in self.list_records()}
        records: list[dict[str, Any]] = []
        for index, raw in enumerate(rows):
            item = normalize_catalog_item(raw, source_path=source, row_index=index)
            staging_id = _staging_id(item)
            old = previous.get(staging_id, {})
            records.append(_build_staging_record(item, staging_id=staging_id, previous=old))
        self._save(records, source_path=source)
        return self.list_records()

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
        previous_source = None
        if self.path.exists():
            payload = self.storage.read(self.path)
            if isinstance(payload, Mapping):
                previous_source = payload.get("source_path")
        self.storage.write(
            self.path,
            {
                "schema_version": CATALOG_STAGING_SCHEMA_VERSION,
                "updated_at": datetime.now(UTC).isoformat(),
                "source_path": str(source_path or previous_source or ""),
                "records": records,
            },
        )


def _upgrade_staging_record(raw: Mapping[str, Any]) -> dict[str, Any]:
    record = dict(deepcopy(raw))
    item = _mapping(record.get("catalog_item"))
    source_item = _mapping(record.get("source_catalog_item")) or deepcopy(item)
    record["source_catalog_item"] = source_item
    record["catalog_item"] = item
    default_target = target_for_catalog_category(_text(item.get("category")))
    record.setdefault("target_category", default_target[0] if default_target else "")
    record.setdefault("target_component_type", default_target[1] if default_target else "")
    record.setdefault(
        "runtime_inputs",
        _default_runtime_inputs(_text(record.get("target_component_type"))),
    )
    record.setdefault("manual_overrides", {})
    errors, warnings = validate_staging_record(record)
    record["validation_errors"] = errors
    record["validation_warnings"] = warnings
    if record.get("status") != STAGING_IMPORTED:
        if errors:
            record["status"] = STAGING_BLOCKED
        elif record.get("status") == STAGING_BLOCKED:
            record["status"] = STAGING_PENDING
    return record


def _build_staging_record(
    source_item: Mapping[str, Any],
    *,
    staging_id: str,
    previous: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    old = _upgrade_staging_record(previous or {}) if previous else {}
    overrides = _mapping(old.get("manual_overrides"))
    item = _apply_manual_overrides(source_item, overrides)
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
        "source_catalog_item": dict(deepcopy(source_item)),
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
    raw = path.read_bytes()
    text = None
    for encoding in ("utf-8-sig", "cp1251"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise ValueError("CSV должен быть в UTF-8 или Windows-1251.")
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error:
        reader = csv.DictReader(io.StringIO(text), delimiter=";")
    else:
        reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    return [dict(row) for row in reader]


def _load_xlsx_rows(path: Path) -> list[dict[str, Any]]:
    """Read the first XLSX sheet using OOXML from the standard library."""

    main_ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    rel_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    package_rel_ns = "http://schemas.openxmlformats.org/package/2006/relationships"
    with zipfile.ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = [
                "".join(node.text or "" for node in item.findall(f".//{{{main_ns}}}t"))
                for item in root.findall(f"{{{main_ns}}}si")
            ]
        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        first_sheet = workbook.find(f".//{{{main_ns}}}sheet")
        if first_sheet is None:
            return []
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
        sheet = ElementTree.fromstring(archive.read(sheet_path))

    matrix: list[list[Any]] = []
    for row in sheet.findall(f".//{{{main_ns}}}row"):
        values: dict[int, Any] = {}
        for cell in row.findall(f"{{{main_ns}}}c"):
            reference = cell.attrib.get("r", "")
            column = _xlsx_column_index(reference)
            cell_type = cell.attrib.get("t")
            value_node = cell.find(f"{{{main_ns}}}v")
            if cell_type == "inlineStr":
                value = "".join(
                    node.text or "" for node in cell.findall(f".//{{{main_ns}}}t")
                )
            else:
                raw_value = value_node.text if value_node is not None else ""
                if cell_type == "s" and raw_value:
                    value = shared[int(raw_value)]
                elif cell_type == "b":
                    value = raw_value == "1"
                else:
                    value = raw_value
            values[column] = value
        if values:
            matrix.append([values.get(index, "") for index in range(max(values) + 1)])
    if not matrix:
        return []
    headers = [_text(value) for value in matrix[0]]
    result: list[dict[str, Any]] = []
    for row in matrix[1:]:
        payload = {
            header: row[index] if index < len(row) else ""
            for index, header in enumerate(headers)
            if header
        }
        if any(value not in ("", None) for value in payload.values()):
            result.append(payload)
    return result


def _xlsx_column_index(reference: str) -> int:
    letters = re.match(r"[A-Z]+", reference.upper())
    value = 0
    for char in letters.group(0) if letters else "A":
        value = value * 26 + ord(char) - ord("A") + 1
    return value - 1


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
