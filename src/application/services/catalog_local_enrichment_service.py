"""Account-free local enrichment for catalog staging.

The service intentionally has two confidence tiers:
- explicit evidence parsed from supplier rows/titles (medium/high confidence);
- conservative category defaults and an estimated demo price (low confidence).

It never replaces a real supplier value.  Every synthetic field is marked in
provenance so it cannot be confused with observed market data.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from application.services.catalog_enrichment_service import apply_specification_source
from application.services.catalog_federation_service import select_effective_offer

LOCAL_SOURCE_ID = "offline-heuristic-v1"
LOCAL_SOURCE_NAME = "Автономная эвристика"
LOCAL_PRICE_SOURCE_ID = "offline-estimator-v1"

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

# Conservative values are intentionally ordinary rather than aspirational.
# They exist only to make a demo candidate calculable when a feed contains no
# technical specification at all.
_CATEGORY_DEFAULTS: dict[str, dict[str, Any]] = {
    "server": {"ram_gb": 32.0, "cpu_cores": 12.0, "storage_gb": 1024.0, "max_power_watts": 450.0},
    "rack_server": {"ram_gb": 32.0, "cpu_cores": 12.0, "storage_gb": 1024.0, "max_power_watts": 450.0},
    "tower_server": {"ram_gb": 32.0, "cpu_cores": 12.0, "storage_gb": 1024.0, "max_power_watts": 450.0},
    "prebuilt_pc": {"ram_gb": 16.0, "cpu_cores": 8.0, "storage_gb": 512.0, "max_power_watts": 250.0},
    "workstation": {"ram_gb": 16.0, "cpu_cores": 8.0, "storage_gb": 512.0, "max_power_watts": 250.0},
    "desktop": {"ram_gb": 16.0, "cpu_cores": 8.0, "storage_gb": 512.0, "max_power_watts": 250.0},
    "laptop": {"ram_gb": 16.0, "cpu_cores": 8.0, "storage_gb": 512.0, "max_power_watts": 90.0},
    "router": {"lan_ports": 4.0, "lan_speed_mbps": 1000.0, "wifi_total_mbps": 1200.0, "max_power_watts": 18.0},
    "switch": {"lan_ports": 8.0, "lan_speed_mbps": 1000.0, "max_power_watts": 60.0},
    "access_point": {"lan_ports": 1.0, "lan_speed_mbps": 1000.0, "wifi_total_mbps": 1200.0, "max_power_watts": 18.0},
    "network_device": {"lan_ports": 4.0, "lan_speed_mbps": 1000.0, "max_power_watts": 30.0},
}

_BASE_PRICE_RUB = {
    "server": 180_000.0,
    "rack_server": 220_000.0,
    "tower_server": 160_000.0,
    "prebuilt_pc": 65_000.0,
    "workstation": 90_000.0,
    "desktop": 65_000.0,
    "laptop": 75_000.0,
    "router": 9_000.0,
    "switch": 18_000.0,
    "access_point": 12_000.0,
    "network_device": 15_000.0,
}


@dataclass(frozen=True)
class LocalEnrichmentSummary:
    requested: int
    changed: int
    explicit_fields: int
    default_fields: int
    estimated_prices: int
    skipped: int

    def as_dict(self) -> dict[str, int]:
        return {
            "requested": self.requested,
            "changed": self.changed,
            "explicit_fields": self.explicit_fields,
            "default_fields": self.default_fields,
            "estimated_prices": self.estimated_prices,
            "skipped": self.skipped,
        }


def infer_price_candidate(*payloads: Mapping[str, Any]) -> tuple[float | None, str]:
    """Find a plausible current RUB price in arbitrary supplier columns.

    Exact aliases remain preferred, but supplier price sheets often use headers
    such as ``Цена с НДС, руб.``, ``Цена партнёра`` or ``РРЦ``.  Old/previous
    price columns are deliberately penalised.
    """

    best: tuple[int, float, str] | None = None
    for payload in payloads:
        if not isinstance(payload, Mapping):
            continue
        for key, value in payload.items():
            normalized = _token(key)
            if not normalized:
                continue
            score = 0
            if normalized in {
                "price", "pricerub", "cost", "цена", "ценаруб", "стоимость",
                "розничнаяцена", "ценасндс", "ценасндсруб", "ррц", "оптоваяцена",
                "ценапартнера", "ценадляпартнера", "ценаединицы", "ценазашт",
            }:
                score += 100
            if any(marker in normalized for marker in ("цена", "price", "стоимость", "ррц")):
                score += 45
            if any(marker in normalized for marker in ("руб", "rub", "ндс", "retail", "рознич", "партнер", "опт")):
                score += 12
            if any(marker in normalized for marker in ("стара", "old", "previous", "до скид", "закупочн", "себестоим")):
                score -= 55
            number = _number(value)
            if score <= 0 or number is None or number <= 0:
                continue
            candidate = (score, number, str(key))
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    if best is None:
        return None, ""
    return best[1], best[2]


def infer_explicit_metrics(
    raw: Mapping[str, Any],
    *,
    title: str = "",
    category: str = "",
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Extract metrics only when the input contains explicit textual evidence."""

    parts: list[tuple[str, str]] = []
    if title:
        parts.append(("title", str(title)))
    _collect_text_parts(raw, parts, depth=0)
    text = " | ".join(value for _key, value in parts if value)
    lower = text.casefold()
    metrics: dict[str, Any] = {}
    evidence: dict[str, dict[str, Any]] = {}

    def apply(field: str, value: Any, pattern: str, confidence: str = "medium") -> None:
        if value in (None, "") or field in metrics:
            return
        metrics[field] = value
        evidence[field] = {"method": pattern, "confidence": confidence}

    # RAM: require a RAM/DDR/memory cue to avoid treating SSD capacities as RAM.
    ram_patterns = (
        r"(?:озу|оператив\w*\s+памят\w*|ram)\s*[:=]?\s*(\d{1,4}(?:[.,]\d+)?)\s*(?:гб|gb)",
        r"(\d{1,4}(?:[.,]\d+)?)\s*(?:гб|gb)\s*(?:ddr[345]?|ram|озу)",
        r"(?:ddr[345]?)\s*[- ]?(\d{1,4}(?:[.,]\d+)?)\s*(?:гб|gb)",
    )
    for pattern in ram_patterns:
        match = re.search(pattern, lower, re.IGNORECASE)
        if match:
            value = _number(match.group(1))
            if value and 1 <= value <= 4096:
                apply("ram_gb", value, "explicit:ram")
                break

    core_patterns = (
        r"(?:cpu\s*)?(\d{1,3})\s*[- ]?(?:core|cores|ядер|ядра|ядерный)",
        r"(?:ядер|количество\s+ядер|cpu\s+cores?)\s*[:=]?\s*(\d{1,3})",
    )
    for pattern in core_patterns:
        match = re.search(pattern, lower, re.IGNORECASE)
        if match:
            value = _number(match.group(1))
            if value and 1 <= value <= 512:
                apply("cpu_cores", value, "explicit:cpu_cores")
                break

    storage_matches: list[float] = []
    for match in re.finditer(
        r"(\d{1,5}(?:[.,]\d+)?)\s*(tb|тб|gb|гб)\s*(?:ssd|hdd|nvme|накопител|диск)",
        lower,
        re.IGNORECASE,
    ):
        value = _capacity_gb(match.group(1), match.group(2))
        if value:
            storage_matches.append(value)
    for match in re.finditer(
        r"(?:ssd|hdd|nvme|накопител\w*|диск)\s*[:=]?\s*(\d{1,5}(?:[.,]\d+)?)\s*(tb|тб|gb|гб)",
        lower,
        re.IGNORECASE,
    ):
        value = _capacity_gb(match.group(1), match.group(2))
        if value:
            storage_matches.append(value)
    if storage_matches:
        apply("storage_gb", max(storage_matches), "explicit:storage")

    power = re.search(r"(?:мощност\w*|power|tdp|блок\s+питания)\s*[:=]?\s*(\d{1,5}(?:[.,]\d+)?)\s*(?:w|вт)", lower)
    if power is None and category in _CATEGORY_DEFAULTS:
        power = re.search(r"(\d{1,5}(?:[.,]\d+)?)\s*(?:w|вт)\b", lower)
    if power:
        value = _number(power.group(1))
        if value and 1 <= value <= 10000:
            apply("max_power_watts", value, "explicit:power")

    if category in {"router", "switch", "access_point", "network_device"}:
        port_patterns = (
            r"(\d{1,3})\s*[xх×]\s*(?:rj[- ]?45|lan|ethernet)",
            r"(?:lan|rj[- ]?45|ethernet|порт\w*)\s*[:=]?\s*(\d{1,3})\s*(?:порт\w*)?",
            r"(\d{1,3})\s*[- ]?(?:port|порт)\w*\s+(?:switch|коммутатор)",
        )
        for pattern in port_patterns:
            match = re.search(pattern, lower, re.IGNORECASE)
            if match:
                value = _number(match.group(1))
                if value and 1 <= value <= 256:
                    apply("lan_ports", value, "explicit:lan_ports")
                    break

        speed = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:gbit/s|gbps|gbe|гбит/с|гбит)\b", lower)
        if speed:
            value = _number(speed.group(1))
            if value:
                apply("lan_speed_mbps", value * 1000.0, "explicit:lan_speed")
        else:
            speed = re.search(r"(\d{2,6})\s*(?:mbit/s|mbps|мбит/с|мбит)\b", lower)
            if speed:
                value = _number(speed.group(1))
                if value:
                    apply("lan_speed_mbps", value, "explicit:lan_speed")

        wifi = re.search(r"\b(?:ax|ac)(\d{3,5})\b", lower)
        if wifi:
            value = _number(wifi.group(1))
            if value:
                apply("wifi_total_mbps", value, "explicit:wifi_class")
        else:
            wifi = re.search(r"wi[- ]?fi[^|]{0,40}?(\d{2,6})\s*(?:mbit/s|mbps|мбит/с|мбит)", lower)
            if wifi:
                value = _number(wifi.group(1))
                if value:
                    apply("wifi_total_mbps", value, "explicit:wifi_speed")

        if "ipv6" in lower or "ipv 6" in lower:
            apply("ipv6_support", True, "explicit:ipv6", "high")

    return metrics, evidence


def enrich_catalog_item_locally(
    item: Mapping[str, Any],
    *,
    fill_defaults: bool = True,
    estimate_missing_price: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fill only missing values and return transparent enrichment statistics."""

    original = dict(deepcopy(item))
    result = dict(deepcopy(item))
    category = str(result.get("category") or "").strip().lower()
    explicit, evidence = infer_explicit_metrics(
        result,
        title=str(result.get("title") or ""),
        category=category,
    )
    metrics = dict(explicit)
    defaulted: list[str] = []
    existing = _mapping(result.get("attributes"))
    if fill_defaults:
        for field, value in _CATEGORY_DEFAULTS.get(category, {}).items():
            if existing.get(field) in (None, "") and field not in metrics:
                metrics[field] = value
                evidence[field] = {"method": "category_default", "confidence": "low"}
                defaulted.append(field)

    applied_before = set(existing)
    # Local enrichment is deliberately fill-only.  Do not attach a synthetic
    # specification source merely to restate values that the supplier already
    # supplied; on 30k+ catalogs that is both misleading and very expensive.
    missing_metrics = {
        field: value
        for field, value in metrics.items()
        if existing.get(field) in (None, "")
    }
    if missing_metrics:
        specification = {
            "source": LOCAL_SOURCE_ID,
            "source_name": LOCAL_SOURCE_NAME,
            "observed_at": datetime.now(UTC).isoformat(),
            "matched_by": "local_evidence",
            "identity": {},
            "metrics": missing_metrics,
            "mapped_features": {field: evidence.get(field, {}) for field in missing_metrics},
            "confidence": "low" if any(field in defaulted for field in missing_metrics) else "medium",
            "estimated_fields": [field for field in defaulted if field in missing_metrics],
        }
        result = apply_specification_source(result, specification)

    applied_after = _mapping(result.get("attributes"))
    explicit_applied = [
        field for field in explicit
        if field not in applied_before and applied_after.get(field) not in (None, "")
    ]
    default_applied = [
        field for field in defaulted
        if field not in applied_before and applied_after.get(field) not in (None, "")
    ]

    price_estimated = False
    current_price = _positive_number(_mapping(result.get("offer")).get("price"))
    if estimate_missing_price and current_price is None:
        estimate = estimate_demo_price(result)
        if estimate is not None:
            now = datetime.now(UTC).isoformat()
            estimate_offer = {
                "price": estimate,
                "currency": "RUB",
                "availability": "unknown",
                "url": None,
                "region": _mapping(result.get("offer")).get("region") or "",
                "observed_at": now,
                "price_kind": "estimated_price",
                "source": LOCAL_PRICE_SOURCE_ID,
                "source_name": "Автономная оценка цены",
                "confidence": "low",
            }
            offers = [
                dict(deepcopy(value))
                for value in result.get("offers", []) or []
                if isinstance(value, Mapping)
                and str(value.get("source") or "") != LOCAL_PRICE_SOURCE_ID
            ]
            offers.append(estimate_offer)
            result["offers"] = offers
            result["offer"] = select_effective_offer(offers) or estimate_offer
            rub_prices = [
                value
                for candidate in offers
                if str(candidate.get("currency") or "RUB").upper() == "RUB"
                if (value := _positive_number(candidate.get("price"))) is not None
            ]
            summary = _mapping(result.get("price_summary"))
            summary.update(
                {
                    "observation_count": len(rub_prices),
                    "source_count": len({str(value.get("source") or "") for value in offers}),
                    "min_rub": min(rub_prices) if rub_prices else None,
                    "median_rub": statistics.median(rub_prices) if rub_prices else None,
                    "max_rub": max(rub_prices) if rub_prices else None,
                    "effective_source": _mapping(result.get("offer")).get("source"),
                    "freshness": "estimated",
                }
            )
            result["price_summary"] = summary
            provenance = _mapping(result.get("field_provenance"))
            provenance["estimated_price"] = {
                "source": LOCAL_PRICE_SOURCE_ID,
                "method": "category_metric_model_v1",
                "confidence": "low",
                "observed_at": now,
            }
            result["field_provenance"] = provenance
            price_estimated = True

    if not explicit_applied and not default_applied and not price_estimated:
        return original, {
            "changed": False,
            "explicit_fields": 0,
            "default_fields": 0,
            "estimated_price": False,
        }

    review = _mapping(result.get("review"))
    warnings = [
        str(value) for value in review.get("warnings", []) or []
        if str(value) not in {
            "Часть характеристик оценена автономно.",
            "Цена является автономной демонстрационной оценкой.",
        }
    ]
    if default_applied:
        warnings.append("Часть характеристик оценена автономно.")
    if price_estimated:
        warnings.append("Цена является автономной демонстрационной оценкой.")
    review["warnings"] = list(dict.fromkeys(warnings))
    result["review"] = review

    return result, {
        "changed": result != original,
        "explicit_fields": len(explicit_applied),
        "default_fields": len(default_applied),
        "estimated_price": price_estimated,
    }


def estimate_demo_price(item: Mapping[str, Any]) -> float | None:
    """Return a conservative deterministic demo estimate, never a market claim."""

    category = str(item.get("category") or "").strip().lower()
    base = _BASE_PRICE_RUB.get(category)
    if base is None:
        return None
    metrics = _mapping(item.get("attributes"))
    defaults = _CATEGORY_DEFAULTS.get(category, {})

    if category in {"server", "rack_server", "tower_server", "prebuilt_pc", "workstation", "desktop", "laptop"}:
        ram = _positive_number(metrics.get("ram_gb")) or _positive_number(defaults.get("ram_gb")) or 16.0
        cores = _positive_number(metrics.get("cpu_cores")) or _positive_number(defaults.get("cpu_cores")) or 8.0
        storage = _positive_number(metrics.get("storage_gb")) or _positive_number(defaults.get("storage_gb")) or 512.0
        ram_ref = _positive_number(defaults.get("ram_gb")) or 16.0
        core_ref = _positive_number(defaults.get("cpu_cores")) or 8.0
        storage_ref = _positive_number(defaults.get("storage_gb")) or 512.0
        factor = 0.55 + 0.18 * math.sqrt(max(0.25, ram / ram_ref))
        factor += 0.20 * math.sqrt(max(0.25, cores / core_ref))
        factor += 0.07 * math.sqrt(max(0.25, storage / storage_ref))
    else:
        ports = _positive_number(metrics.get("lan_ports")) or _positive_number(defaults.get("lan_ports")) or 1.0
        speed = _positive_number(metrics.get("lan_speed_mbps")) or _positive_number(defaults.get("lan_speed_mbps")) or 1000.0
        wifi = _positive_number(metrics.get("wifi_total_mbps")) or _positive_number(defaults.get("wifi_total_mbps")) or 0.0
        port_ref = _positive_number(defaults.get("lan_ports")) or 4.0
        factor = 0.60 + 0.25 * math.sqrt(max(0.25, ports / port_ref))
        factor += 0.10 * math.sqrt(max(0.25, speed / 1000.0))
        if wifi:
            factor += 0.05 * math.sqrt(max(0.25, wifi / 1200.0))
    return float(max(500.0, round((base * factor) / 100.0) * 100.0))


def enrich_staging_records_locally(
    service: Any,
    *,
    staging_ids: Iterable[str] | None = None,
    fill_defaults: bool = True,
    estimate_missing_price: bool = True,
    manifest_path: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> LocalEnrichmentSummary:
    emit = progress or (lambda _message: None)
    selected = {str(value) for value in (staging_ids or []) if str(value)}
    records = service.list_records()
    target_records = [
        record for record in records
        if not selected or str(record.get("staging_id") or "") in selected
    ]
    requested = len(target_records)
    skipped = sum(1 for record in target_records if str(record.get("status") or "") == "imported")
    process_total = max(0, requested - skipped)
    counters = {
        "requested": requested,
        "changed": 0,
        "explicit_fields": 0,
        "default_fields": 0,
        "estimated_prices": 0,
        "skipped": skipped,
    }
    # Per-row results are useful for a diagnostic manifest, but retaining them
    # for a 30k+ catalog is unnecessary when no manifest was requested.
    results: list[dict[str, Any]] | None = [] if manifest_path else None
    processed = 0

    def transform(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
        nonlocal processed
        processed += 1
        staging_id = str(record.get("staging_id") or "")
        if processed == 1 or processed % 250 == 0 or processed == process_total:
            emit(f"Автообогащение {processed}/{process_total}")
        source_item = _mapping(record.get("source_catalog_item"))
        enriched, stats = enrich_catalog_item_locally(
            source_item,
            fill_defaults=fill_defaults,
            estimate_missing_price=estimate_missing_price,
        )
        counters["explicit_fields"] += int(stats["explicit_fields"])
        counters["default_fields"] += int(stats["default_fields"])
        counters["estimated_prices"] += int(bool(stats["estimated_price"]))
        if stats["changed"]:
            counters["changed"] += 1
        if results is not None:
            results.append({"staging_id": staging_id, **stats})
        return enriched if stats["changed"] else None

    transformer = getattr(service, "transform_source_items", None)
    if callable(transformer):
        transformer(transform, staging_ids=selected or None)
    else:
        # Compatibility path for light-weight service doubles used outside the
        # application.  Production staging uses the streaming-memory transform.
        updates: dict[str, dict[str, Any]] = {}
        for record in target_records:
            if str(record.get("status") or "") == "imported":
                continue
            enriched = transform(record)
            if enriched is not None:
                updates[str(record.get("staging_id") or "")] = dict(enriched)
        if updates:
            service.apply_source_item_updates(updates)

    summary = LocalEnrichmentSummary(**counters)
    if manifest_path:
        path = Path(manifest_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "source": LOCAL_SOURCE_ID,
                    "created_at": datetime.now(UTC).isoformat(),
                    "summary": summary.as_dict(),
                    "results": results or [],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return summary


def _collect_text_parts(value: Any, out: list[tuple[str, str]], *, depth: int) -> None:
    if depth > 2:
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).startswith("_"):
                continue
            if isinstance(item, (Mapping, list, tuple)):
                _collect_text_parts(item, out, depth=depth + 1)
            elif item not in (None, ""):
                out.append((str(key), f"{key}: {item}"))
    elif isinstance(value, (list, tuple)):
        for item in value[:64]:
            _collect_text_parts(item, out, depth=depth + 1)


def _capacity_gb(value: Any, unit: str) -> float | None:
    number = _number(value)
    if number is None or number <= 0:
        return None
    if str(unit).casefold() in {"tb", "тб"}:
        number *= 1024.0
    return number if 1 <= number <= 1_048_576 else None


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    text = str(value).strip().replace("\u00a0", " ")
    # If there are multiple punctuation marks, spaces usually are thousands separators.
    text = re.sub(r"\s+", "", text)
    normalized = re.sub(r"[^0-9,.-]", "", text)
    if normalized.count(",") == 1 and normalized.count(".") == 0:
        normalized = normalized.replace(",", ".")
    elif normalized.count(",") > 0 and normalized.count(".") > 0:
        normalized = normalized.replace(",", "")
    try:
        return float(normalized)
    except ValueError:
        return None


def _positive_number(value: Any) -> float | None:
    number = _number(value)
    return number if number is not None and number > 0 else None


def _token(value: Any) -> str:
    return re.sub(r"[^0-9a-zа-я]+", "", str(value or "").casefold())


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


__all__ = [
    "LOCAL_SOURCE_ID",
    "LocalEnrichmentSummary",
    "enrich_catalog_item_locally",
    "enrich_staging_records_locally",
    "estimate_demo_price",
    "infer_explicit_metrics",
    "infer_price_candidate",
]
