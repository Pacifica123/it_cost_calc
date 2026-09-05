"""Independent procurement-price benchmarks for catalog staging.

P4 deliberately treats EIS/contract data as statistical evidence, not as a
supplier catalog.  Imported contract rows never become ``offers`` and therefore
cannot silently replace the commercial price used by GA/AHP/NPV.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
import statistics
import urllib.parse
import urllib.request
import zipfile
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from xml.etree import ElementTree

from application.services.catalog_staging_service import (
    CatalogStagingService,
    normalize_catalog_item,
)


class ProcurementBenchmarkError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProcurementObservation:
    title: str
    unit_price_rub: float
    quantity: float | None = None
    total_price_rub: float | None = None
    category: str = ""
    okpd2: str = ""
    region: str = ""
    observed_at: str = ""
    contract_number: str = ""
    source_file: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "unit_price_rub": self.unit_price_rub,
            "quantity": self.quantity,
            "total_price_rub": self.total_price_rub,
            "category": self.category,
            "okpd2": self.okpd2 or None,
            "region": self.region or None,
            "observed_at": self.observed_at or None,
            "contract_number": self.contract_number or None,
            "source_file": self.source_file or None,
        }


@dataclass(frozen=True)
class ProcurementBenchmarkSummary:
    observations: int
    catalog_records: int
    matched_records: int
    identity_matches: int
    category_matches: int
    skipped_records: int

    def as_dict(self) -> dict[str, int]:
        return {
            "observations": self.observations,
            "catalog_records": self.catalog_records,
            "matched_records": self.matched_records,
            "identity_matches": self.identity_matches,
            "category_matches": self.category_matches,
            "skipped_records": self.skipped_records,
        }


_TITLE_KEYS = (
    "name",
    "productname",
    "product_name",
    "fullname",
    "description",
    "objectname",
    "subjectname",
    "purchaseobjectname",
    "наименование",
    "наименованиетовара",
)
_UNIT_PRICE_KEYS = (
    "unitprice",
    "unit_price",
    "priceperunit",
    "price",
    "pricerub",
    "цена",
    "ценазаединицу",
)
_TOTAL_PRICE_KEYS = (
    "totalprice",
    "total_price",
    "sum",
    "amount",
    "cost",
    "стоимость",
    "сумма",
)
_QUANTITY_KEYS = ("quantity", "qty", "count", "количество")
_DATE_KEYS = (
    "publisheddate",
    "publishdate",
    "signdate",
    "contractdate",
    "createdate",
    "date",
    "дата",
)
_CONTRACT_KEYS = (
    "regnum",
    "registrynumber",
    "contractregnum",
    "contractnumber",
    "notificationnumber",
    "number",
)
_OKPD_KEYS = ("okpd2code", "okpdcode", "okpd2", "okpd")
_REGION_KEYS = ("regionname", "region", "regioncode", "субъект", "регион")
_ITEM_NODE_NAMES = {
    "product",
    "productinfo",
    "contractsubject",
    "contractsubjectinfo",
    "purchaseobject",
    "object",
    "position",
    "item",
}
_TOKEN_RE = re.compile(r"[a-zа-я0-9][a-zа-я0-9+._/-]*", re.IGNORECASE)
_STOPWORDS = {
    "для",
    "или",
    "and",
    "the",
    "with",
    "без",
    "компьютер",
    "сервер",
    "маршрутизатор",
    "роутер",
    "коммутатор",
    "рабочая",
    "станция",
    "оборудование",
    "шт",
}


def load_procurement_observations(
    location: str | Path,
    *,
    region: str = "",
    max_records: int = 20000,
    progress: Callable[[str], None] | None = None,
) -> list[ProcurementObservation]:
    """Load EIS-like machine-readable exports from XML/ZIP/JSON/CSV.

    ``location`` may be a local file or a direct HTTP(S) URL.  The parser is
    deliberately schema-tolerant because EIS exchange schemas evolve.  It only
    accepts rows that contain a title and a positive unit price (or total/qty
    pair), so unrelated XML nodes do not become benchmark observations.
    """

    if max_records <= 0:
        return []
    name, payload = _read_location(location)
    suffix = Path(urllib.parse.urlparse(name).path).suffix.lower()
    if progress:
        progress(f"ЕИС: читаю {name}")

    observations: list[ProcurementObservation] = []
    if suffix == ".zip" or payload[:4] == b"PK\x03\x04":
        try:
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                for member in archive.infolist():
                    if len(observations) >= max_records or member.is_dir():
                        break
                    member_suffix = Path(member.filename).suffix.lower()
                    if member_suffix not in {".xml", ".json", ".csv"}:
                        continue
                    if progress:
                        progress(f"ЕИС: {member.filename}")
                    observations.extend(
                        _parse_payload(
                            archive.read(member),
                            suffix=member_suffix,
                            source_file=member.filename,
                            region=region,
                            limit=max_records - len(observations),
                        )
                    )
        except zipfile.BadZipFile as exc:
            raise ProcurementBenchmarkError(f"Некорректный ZIP ЕИС: {exc}") from exc
    else:
        observations = _parse_payload(
            payload,
            suffix=suffix,
            source_file=Path(urllib.parse.urlparse(name).path).name or name,
            region=region,
            limit=max_records,
        )

    deduplicated: dict[tuple[str, int, str, str], ProcurementObservation] = {}
    for item in observations:
        key = (
            _normalized_text(item.title),
            int(round(item.unit_price_rub * 100)),
            item.contract_number,
            item.source_file,
        )
        deduplicated.setdefault(key, item)
    result = list(deduplicated.values())[:max_records]
    if progress:
        progress(f"ЕИС: ценовых наблюдений {len(result)}")
    return result


def apply_procurement_benchmarks(
    staging: CatalogStagingService,
    observations: Iterable[ProcurementObservation],
    *,
    staging_ids: Iterable[str] = (),
    source_location: str = "",
    region: str = "",
    manifest_path: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> ProcurementBenchmarkSummary:
    """Attach statistical benchmark evidence to staging records without offers."""

    pool = list(observations)
    selected = {str(value).strip() for value in staging_ids if str(value).strip()}
    records = staging.list_records()
    updates: dict[str, dict[str, Any]] = {}
    matched = identity_matches = category_matches = skipped = 0

    for record in records:
        staging_id = str(record.get("staging_id") or "")
        if selected and staging_id not in selected:
            continue
        if record.get("status") == "imported":
            skipped += 1
            continue
        source_item = deepcopy(dict(record.get("source_catalog_item") or {}))
        benchmark = benchmark_for_catalog_item(source_item, pool, region=region)
        if not benchmark:
            skipped += 1
            continue
        matched += 1
        if benchmark["match_level"] == "identity":
            identity_matches += 1
        else:
            category_matches += 1
        current_price = _number(dict(source_item.get("offer") or {}).get("price"))
        if current_price:
            benchmark["offer_to_median_ratio"] = round(
                current_price / float(benchmark["median_rub"]), 4
            )
            benchmark["offer_delta_rub"] = round(
                current_price - float(benchmark["median_rub"]), 2
            )
        benchmark["source"] = "eis_procurement"
        benchmark["source_location"] = source_location or None
        source_item["procurement_benchmark"] = benchmark
        provenance = dict(source_item.get("field_provenance") or {})
        provenance["procurement_benchmark"] = {
            "source": "eis_procurement",
            "source_location": source_location or None,
            "region": region or None,
            "applied_at": datetime.now(UTC).isoformat(),
            "observation_count": benchmark["observation_count"],
            "match_level": benchmark["match_level"],
        }
        source_item["field_provenance"] = provenance
        updates[staging_id] = source_item
        if progress:
            progress(
                f"ЕИС: {source_item.get('title') or staging_id}: "
                f"median {benchmark['median_rub']:.0f} ₽ ({benchmark['observation_count']})"
            )

    if updates:
        staging.apply_source_item_updates(updates)

    summary = ProcurementBenchmarkSummary(
        observations=len(pool),
        catalog_records=len(records) if not selected else len(selected),
        matched_records=matched,
        identity_matches=identity_matches,
        category_matches=category_matches,
        skipped_records=skipped,
    )
    if manifest_path:
        _write_manifest(
            Path(manifest_path),
            summary=summary,
            source_location=source_location,
            region=region,
            selected=sorted(selected),
        )
    return summary


def benchmark_for_catalog_item(
    item: Mapping[str, Any],
    observations: Iterable[ProcurementObservation],
    *,
    region: str = "",
) -> dict[str, Any]:
    pool = list(observations)
    if region:
        preferred = [obs for obs in pool if not obs.region or _same_text(obs.region, region)]
        if preferred:
            pool = preferred

    identity_pool = [obs for obs in pool if _identity_matches(item, obs)]
    match_level = "identity" if identity_pool else "category"
    candidates = identity_pool or [
        obs for obs in pool if obs.category and obs.category == str(item.get("category") or "")
    ]
    if not candidates:
        return {}

    prices = sorted(obs.unit_price_rub for obs in candidates if obs.unit_price_rub > 0)
    if not prices:
        return {}
    dates = sorted(obs.observed_at for obs in candidates if obs.observed_at)
    examples = []
    for obs in candidates[:5]:
        examples.append(
            {
                "title": obs.title,
                "price_rub": obs.unit_price_rub,
                "contract_number": obs.contract_number or None,
                "observed_at": obs.observed_at or None,
                "okpd2": obs.okpd2 or None,
            }
        )
    return {
        "match_level": match_level,
        "observation_count": len(prices),
        "median_rub": round(float(statistics.median(prices)), 2),
        "p25_rub": round(_percentile(prices, 0.25), 2),
        "p75_rub": round(_percentile(prices, 0.75), 2),
        "min_rub": round(float(prices[0]), 2),
        "max_rub": round(float(prices[-1]), 2),
        "observed_from": dates[0] if dates else None,
        "observed_to": dates[-1] if dates else None,
        "region": region or None,
        "examples": examples,
    }


def _read_location(location: str | Path) -> tuple[str, bytes]:
    value = str(location).strip()
    if not value:
        raise ProcurementBenchmarkError("Не задан файл или URL выгрузки ЕИС.")
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme in {"http", "https"}:
        request = urllib.request.Request(
            value,
            headers={"User-Agent": "ITCostCalc/0.1 procurement-benchmark"},
        )
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                return response.geturl(), response.read()
        except Exception as exc:
            raise ProcurementBenchmarkError(f"Не удалось загрузить выгрузку ЕИС: {exc}") from exc
    path = Path(value).expanduser()
    if not path.is_file():
        raise ProcurementBenchmarkError(f"Файл ЕИС не найден: {path}")
    return str(path), path.read_bytes()


def _parse_payload(
    payload: bytes,
    *,
    suffix: str,
    source_file: str,
    region: str,
    limit: int,
) -> list[ProcurementObservation]:
    if limit <= 0:
        return []
    if suffix == ".json":
        return _parse_json(payload, source_file=source_file, region=region, limit=limit)
    if suffix == ".csv":
        return _parse_csv(payload, source_file=source_file, region=region, limit=limit)
    return _parse_xml(payload, source_file=source_file, region=region, limit=limit)


def _parse_csv(
    payload: bytes, *, source_file: str, region: str, limit: int
) -> list[ProcurementObservation]:
    text = _decode_text(payload)
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    rows = csv.DictReader(io.StringIO(text), dialect=dialect)
    result: list[ProcurementObservation] = []
    for raw in rows:
        item = _observation_from_mapping(raw, source_file=source_file, region=region)
        if item:
            result.append(item)
        if len(result) >= limit:
            break
    return result


def _parse_json(
    payload: bytes, *, source_file: str, region: str, limit: int
) -> list[ProcurementObservation]:
    try:
        root = json.loads(_decode_text(payload))
    except json.JSONDecodeError as exc:
        raise ProcurementBenchmarkError(f"Некорректный JSON ЕИС: {exc}") from exc
    result: list[ProcurementObservation] = []
    for mapping in _walk_mappings(root):
        item = _observation_from_mapping(mapping, source_file=source_file, region=region)
        if item:
            result.append(item)
        if len(result) >= limit:
            break
    return result


def _parse_xml(
    payload: bytes, *, source_file: str, region: str, limit: int
) -> list[ProcurementObservation]:
    try:
        root = ElementTree.fromstring(payload)
    except ElementTree.ParseError as exc:
        raise ProcurementBenchmarkError(f"Некорректный XML ЕИС ({source_file}): {exc}") from exc

    global_values = _leaf_mapping(root)
    contract_number = _first_text(global_values, _CONTRACT_KEYS)
    observed_at = _date_text(_first_text(global_values, _DATE_KEYS))
    document_region = _first_text(global_values, _REGION_KEYS) or region
    result: list[ProcurementObservation] = []
    for element in root.iter():
        if _key(element.tag) not in _ITEM_NODE_NAMES:
            continue
        values = _leaf_mapping(element)
        item = _observation_from_mapping(
            values,
            source_file=source_file,
            region=document_region,
            default_contract=contract_number,
            default_date=observed_at,
        )
        if item:
            result.append(item)
        if len(result) >= limit:
            return result

    if result:
        return result

    # Schema fallback: inspect medium-sized containers rather than assuming one
    # historical EIS namespace or exact XSD path.
    for element in root.iter():
        children = list(element)
        if not children or len(children) > 120:
            continue
        values = _leaf_mapping(element)
        item = _observation_from_mapping(
            values,
            source_file=source_file,
            region=document_region,
            default_contract=contract_number,
            default_date=observed_at,
        )
        if item:
            result.append(item)
        if len(result) >= limit:
            break
    return result


def _observation_from_mapping(
    raw: Mapping[str, Any],
    *,
    source_file: str,
    region: str,
    default_contract: str = "",
    default_date: str = "",
) -> ProcurementObservation | None:
    values = {_key(key): value for key, value in raw.items()}
    title = _first_text(values, _TITLE_KEYS)
    if len(title) < 3:
        return None
    quantity = _number(_first_value(values, _QUANTITY_KEYS))
    unit_price = _number(_first_value(values, _UNIT_PRICE_KEYS))
    total_price = _number(_first_value(values, _TOTAL_PRICE_KEYS))
    if (unit_price is None or unit_price <= 0) and total_price and quantity and quantity > 0:
        unit_price = total_price / quantity
    if unit_price is None or unit_price <= 0 or not math.isfinite(unit_price):
        return None
    if unit_price > 10_000_000_000:
        return None

    normalized = normalize_catalog_item(
        {"title": title, "price": unit_price, "currency": "RUB"},
        source_path=source_file or "eis.xml",
        row_index=0,
        source_context={"id": "eis_procurement", "price_kind": "procurement_benchmark"},
    )
    return ProcurementObservation(
        title=title,
        unit_price_rub=float(unit_price),
        quantity=quantity,
        total_price_rub=total_price,
        category=str(normalized.get("category") or ""),
        okpd2=_first_text(values, _OKPD_KEYS),
        region=_first_text(values, _REGION_KEYS) or region,
        observed_at=_date_text(_first_text(values, _DATE_KEYS)) or default_date,
        contract_number=_first_text(values, _CONTRACT_KEYS) or default_contract,
        source_file=source_file,
    )


def _walk_mappings(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _walk_mappings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_mappings(child)


def _leaf_mapping(element: ElementTree.Element) -> dict[str, str]:
    result: dict[str, str] = {}
    for child in element.iter():
        if list(child):
            continue
        text = (child.text or "").strip()
        if text:
            result.setdefault(_key(child.tag), text)
    return result


def _identity_matches(item: Mapping[str, Any], observation: ProcurementObservation) -> bool:
    identity = dict(item.get("identity") or {})
    haystack = _normalized_text(observation.title)
    gtin = re.sub(r"\D", "", str(identity.get("gtin") or ""))
    if len(gtin) >= 8 and gtin in re.sub(r"\D", "", observation.title):
        return True
    brand = _normalized_text(identity.get("brand"))
    for field in ("mpn", "model"):
        value = _normalized_text(identity.get(field))
        if len(value) < 3:
            continue
        if value in haystack and (not brand or brand in haystack):
            return True
    # Conservative title overlap fallback only when at least two distinctive
    # tokens coincide.  This is still stronger than category-only matching.
    item_tokens = _significant_tokens(str(item.get("title") or ""))
    obs_tokens = _significant_tokens(observation.title)
    return len(item_tokens & obs_tokens) >= 2


def _significant_tokens(value: str) -> set[str]:
    return {
        token.lower()
        for token in _TOKEN_RE.findall(value.lower())
        if len(token) >= 3 and token.lower() not in _STOPWORDS
    }


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    position = (len(values) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(values[lower])
    weight = position - lower
    return float(values[lower] * (1.0 - weight) + values[upper] * weight)


def _write_manifest(
    path: Path,
    *,
    summary: ProcurementBenchmarkSummary,
    source_location: str,
    region: str,
    selected: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": "eis_procurement",
                "source_location": source_location or None,
                "region": region or None,
                "selected_staging_ids": selected,
                "created_at": datetime.now(UTC).isoformat(),
                "summary": summary.as_dict(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _key(value: Any) -> str:
    text = str(value or "")
    if "}" in text:
        text = text.rsplit("}", 1)[-1]
    return re.sub(r"[^a-zа-я0-9]", "", text.lower())


def _first_value(values: Mapping[str, Any], keys: Iterable[str]) -> Any:
    normalized = {_key(key) for key in keys}
    for key in normalized:
        if key in values and values[key] not in (None, ""):
            return values[key]
    return None


def _first_text(values: Mapping[str, Any], keys: Iterable[str]) -> str:
    value = _first_value(values, keys)
    return str(value or "").strip()


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    text = str(value).strip().replace("\xa0", " ")
    text = re.sub(r"[^0-9,.-]", "", text).replace(",", ".")
    if not text:
        return None
    try:
        result = float(text)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def _date_text(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return f"{text}T00:00:00+00:00"
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).isoformat()


def _decode_text(payload: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    return payload.decode("utf-8", errors="replace")


def _normalized_text(value: Any) -> str:
    return " ".join(_TOKEN_RE.findall(str(value or "").lower()))


def _same_text(left: str, right: str) -> bool:
    return _normalized_text(left) == _normalized_text(right)


__all__ = [
    "ProcurementBenchmarkError",
    "ProcurementBenchmarkSummary",
    "ProcurementObservation",
    "apply_procurement_benchmarks",
    "benchmark_for_catalog_item",
    "load_procurement_observations",
]
