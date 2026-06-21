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

CATALOG_STAGING_SCHEMA_VERSION = 1
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


def supported_catalog_extensions() -> tuple[str, ...]:
    return tuple(sorted(_SUPPORTED_EXTENSIONS))


def target_for_catalog_category(category: str) -> tuple[str, str] | None:
    return _TARGET_BY_CATEGORY.get(str(category or "").strip().lower())


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
        field: _metric_value(metrics.get(field, item.get(field)))
        for field in _METRIC_FIELDS
        if _metric_value(metrics.get(field, item.get(field))) is not None
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

    if not title:
        errors.append("Не заполнено название товара.")
    if category not in _TARGET_BY_CATEGORY:
        errors.append(
            f"Категория '{category or 'не указана'}' пока не преобразуется в готовое устройство ТО."
        )
    price = _number(offer.get("price"))
    if price is None or price <= 0:
        errors.append("Цена должна быть положительным числом.")
    if _text(offer.get("currency") or "RUB").upper() != "RUB":
        errors.append("Для импорта в расчёт нужна цена в RUB.")
    if not offer.get("observed_at"):
        warnings.append("Не указано время получения цены.")
    if not any(identity.get(key) for key in ("gtin", "mpn", "model")):
        warnings.append("Нет GTIN, MPN или модели для надёжного объединения источников.")

    target = _TARGET_BY_CATEGORY.get(category)
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


def catalog_item_to_runtime_row(record: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Convert one approved staging record to a runtime technical alternative."""

    if record.get("status") != STAGING_APPROVED:
        raise ValueError("Импортировать можно только подтверждённую запись.")
    if record.get("validation_errors"):
        raise ValueError("Запись содержит блокирующие ошибки.")

    item = _mapping(record.get("catalog_item"))
    category = _text(item.get("category")).lower()
    if category not in _TARGET_BY_CATEGORY:
        raise ValueError(f"Категория {category!r} не поддерживается.")
    target_category, component_type = _TARGET_BY_CATEGORY[category]
    offer = _mapping(item.get("offer"))
    metrics = _mapping(item.get("attributes"))
    warnings = list(record.get("validation_warnings") or [])

    row: dict[str, Any] = {
        "name": _text(item.get("title")),
        "quantity": 1,
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
            "staging_id": record.get("staging_id"),
        },
    }
    for field in _METRIC_FIELDS:
        if field in metrics and metrics[field] not in (None, ""):
            row[field] = metrics[field]
    if "max_power_watts" in row:
        row["max_power"] = row["max_power_watts"]
    if component_type == "workstation":
        row["client_seats"] = 1
        warnings.append("Принято допущение: одна готовая рабочая станция создаёт одно место.")
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
        return [dict(deepcopy(record)) for record in records if isinstance(record, Mapping)]

    def stage_file(self, source_path: str | Path) -> list[dict[str, Any]]:
        source = Path(source_path)
        rows = load_catalog_rows(source)
        previous = {record.get("staging_id"): record for record in self.list_records()}
        records: list[dict[str, Any]] = []
        for index, raw in enumerate(rows):
            item = normalize_catalog_item(raw, source_path=source, row_index=index)
            errors, warnings = validate_staging_item(item)
            staging_id = _staging_id(item)
            old = previous.get(staging_id, {})
            status = str(old.get("status") or STAGING_PENDING)
            if errors:
                status = STAGING_BLOCKED
            elif status == STAGING_BLOCKED:
                status = STAGING_PENDING
            records.append(
                {
                    "staging_id": staging_id,
                    "status": status,
                    "catalog_item": item,
                    "validation_errors": errors,
                    "validation_warnings": warnings,
                    "imported_at": old.get("imported_at"),
                }
            )
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
    "catalog_item_to_runtime_row",
    "load_catalog_rows",
    "normalize_catalog_item",
    "supported_catalog_extensions",
    "target_for_catalog_category",
    "validate_staging_item",
]
