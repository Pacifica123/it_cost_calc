"""Structured specification enrichment for federated catalog items.

P3 deliberately separates commercial offers from technical specifications.
Supplier feeds remain the source of Russian prices and availability, while
Icecat may fill *missing* normalized technical metrics.  Existing supplier or
manual values are never silently replaced; disagreements are kept as conflicts.
"""

from __future__ import annotations

import json
import os
import re
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol

ICECAT_API_URL = "https://live.icecat.biz/api"
ICECAT_SOURCE_ID = "icecat"
ICECAT_SOURCE_NAME = "Open Icecat"
ICECAT_CONTENT = "essentialinfo,featuregroups"

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

# English feature names are used intentionally: the enrichment call defaults to
# EN so mapping remains stable even when the desktop UI is Russian.
_FEATURE_RULES: dict[str, dict[str, int]] = {
    "ram_gb": {
        "internal memory": 100,
        "installed memory": 95,
        "memory capacity": 90,
        "total internal memory": 90,
    },
    "cpu_cores": {
        "processor cores": 100,
        "number of processor cores": 100,
        "total processor cores": 95,
        "cpu cores": 95,
    },
    "storage_gb": {
        "total storage capacity": 100,
        "total ssd capacity": 90,
        "total hdd capacity": 80,
        "storage capacity": 75,
    },
    "max_power_watts": {
        "power consumption max": 100,
        "maximum power consumption": 100,
        "max power consumption": 100,
        "power consumption typical": 80,
        "power consumption": 70,
    },
    "lan_ports": {
        "ethernet lan rj 45 ports": 100,
        "basic switching rj 45 ethernet ports quantity": 100,
        "gigabit ethernet copper ports quantity": 90,
        "rj 45 ports quantity": 85,
        "ethernet ports quantity": 80,
    },
    "lan_speed_mbps": {
        "ethernet lan data rates": 100,
        "ethernet data transfer rates": 100,
        "ethernet lan data rate": 95,
        "data transfer rate ethernet lan": 90,
    },
    "wifi_total_mbps": {
        "maximum wlan data transfer rate": 100,
        "maximum wi fi data rate": 100,
        "wi fi data rate max": 95,
        "wlan data transfer rate max": 95,
    },
    "ipv6_support": {
        "ipv6 support": 100,
        "ipv6": 95,
        "network protocols": 60,
        "supported network protocols": 60,
    },
}

_ICECAT_AUTH_ERROR_CODES = {1, 2, 3, 5, 6, 7, 10}
_ICECAT_UNAVAILABLE_CODES = {4, 8, 9, 12, 404}


class _HttpSession(Protocol):
    def get(self, url: str, **kwargs: Any) -> Any: ...


class IcecatEnrichmentError(RuntimeError):
    """Base class for P3 enrichment failures."""


class IcecatConfigurationError(IcecatEnrichmentError):
    """Credentials or API configuration prevents enrichment."""


class IcecatRequestError(IcecatEnrichmentError):
    """Network/protocol failure while talking to Icecat."""


class IcecatProductUnavailable(IcecatEnrichmentError):
    """Product is absent or unavailable for this Icecat account."""


class IcecatIdentityMismatch(IcecatEnrichmentError):
    """Icecat returned a product different from the requested identity."""


@dataclass(frozen=True)
class IcecatLookup:
    matched_by: str
    requested_identity: dict[str, str]
    specification: dict[str, Any]


@dataclass(frozen=True)
class IcecatEnrichmentSummary:
    requested: int
    eligible: int
    matched: int
    changed: int
    unchanged: int
    skipped: int
    unavailable: int
    errors: int

    def as_dict(self) -> dict[str, int]:
        return {
            "requested": self.requested,
            "eligible": self.eligible,
            "matched": self.matched,
            "changed": self.changed,
            "unchanged": self.unchanged,
            "skipped": self.skipped,
            "unavailable": self.unavailable,
            "errors": self.errors,
        }


class IcecatClient:
    """Minimal Icecat JSON client with strict identity verification."""

    def __init__(
        self,
        *,
        username: str,
        api_token: str = "",
        language: str = "EN",
        endpoint: str = ICECAT_API_URL,
        timeout_seconds: float = 20.0,
        session: _HttpSession | None = None,
    ) -> None:
        self.username = str(username or "").strip()
        self.api_token = str(api_token or "").strip()
        self.language = str(language or "EN").strip().upper() or "EN"
        self.endpoint = str(endpoint or ICECAT_API_URL).strip()
        self.timeout_seconds = float(timeout_seconds)
        self._session = session
        if not self.username:
            raise IcecatConfigurationError("Нужен логин Icecat.")

    def lookup(self, item: Mapping[str, Any]) -> IcecatLookup:
        lookup = icecat_lookup_identity(item)
        if not lookup:
            raise IcecatProductUnavailable("Нет GTIN или пары бренд + MPN.")

        matched_by = str(lookup.pop("matched_by"))
        requested_identity = dict(lookup)
        params: dict[str, str] = {
            "lang": self.language,
            "shopname": self.username,
            "content": ICECAT_CONTENT,
        }
        if matched_by == "gtin":
            params["GTIN"] = requested_identity["gtin"]
        else:
            params["Brand"] = requested_identity["brand"]
            params["ProductCode"] = requested_identity["mpn"]
        headers = {"Accept": "application/json"}
        if self.api_token:
            headers["api-token"] = self.api_token

        session = self._session or _requests_session()
        try:
            response = session.get(
                self.endpoint,
                params=params,
                headers=headers,
                timeout=self.timeout_seconds,
            )
        except Exception as exc:  # requests is intentionally lazy/optional for smoke import
            raise IcecatRequestError(f"Icecat недоступен: {exc}") from exc

        status = int(getattr(response, "status_code", 200) or 200)
        if status in {401, 403}:
            raise IcecatConfigurationError(f"Icecat отклонил доступ: HTTP {status}.")
        if status == 404:
            raise IcecatProductUnavailable("Товар не найден в Icecat.")
        if status >= 400:
            raise IcecatRequestError(f"Icecat вернул HTTP {status}.")

        try:
            payload = response.json()
        except Exception as exc:
            raise IcecatRequestError("Icecat вернул некорректный JSON.") from exc
        if not isinstance(payload, Mapping):
            raise IcecatRequestError("Icecat вернул неожиданный формат ответа.")
        _raise_for_icecat_status(payload)

        specification = parse_icecat_specification(
            payload,
            requested_identity=requested_identity,
            matched_by=matched_by,
            language=self.language,
        )
        return IcecatLookup(
            matched_by=matched_by,
            requested_identity=requested_identity,
            specification=specification,
        )


def icecat_lookup_identity(item: Mapping[str, Any]) -> dict[str, str]:
    """Return the safest supported Icecat lookup key for a catalog item."""

    identity = _mapping(item.get("identity"))
    gtin = _normalize_gtin(identity.get("gtin"))
    if gtin:
        return {"matched_by": "gtin", "gtin": gtin}
    brand = _text(identity.get("brand"))
    mpn = _text(identity.get("mpn"))
    if brand and mpn:
        return {"matched_by": "brand_mpn", "brand": brand, "mpn": mpn}
    return {}


def parse_icecat_specification(
    payload: Mapping[str, Any],
    *,
    requested_identity: Mapping[str, Any],
    matched_by: str,
    language: str = "EN",
    observed_at: str | None = None,
) -> dict[str, Any]:
    """Validate one Icecat product response and map useful normalized metrics."""

    data = _mapping(payload.get("data")) or _mapping(payload)
    info = (
        _mapping(data.get("GeneralInfo"))
        or _mapping(data.get("EssentialInfo"))
        or data
    )
    returned_identity = _icecat_identity(info)
    _verify_identity(requested_identity, returned_identity, matched_by=matched_by)

    metrics, mapped_features = map_icecat_metrics(data)
    timestamp = observed_at or datetime.now(UTC).isoformat()
    category = _nested_text(info.get("Category"), "Name", "Value") or _value_text(
        info.get("Category")
    )
    title = (
        _text(info.get("Title"))
        or _text(info.get("ProductName"))
        or _nested_text(info.get("ProductNameInfo"), "ProductIntName")
    )
    icecat_id = (
        info.get("IcecatId")
        or info.get("IcecatID")
        or info.get("Icecat_id")
        or data.get("IcecatId")
    )
    return {
        "source": ICECAT_SOURCE_ID,
        "source_name": ICECAT_SOURCE_NAME,
        "observed_at": timestamp,
        "matched_by": matched_by,
        "language": str(language or "EN").upper(),
        "icecat_id": icecat_id,
        "title": title or None,
        "category": category or None,
        "identity": returned_identity,
        "metrics": metrics,
        "mapped_features": mapped_features,
    }


def map_icecat_metrics(data: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Map selected Icecat feature names/units onto the diploma's metric contract."""

    candidates: dict[str, list[tuple[int, Any, dict[str, Any]]]] = {
        field: [] for field in _METRIC_FIELDS
    }
    groups = data.get("FeaturesGroups") or data.get("FeatureGroups") or []
    if isinstance(groups, Mapping):
        groups = list(groups.values())
    if not isinstance(groups, list):
        groups = []

    for group in groups:
        if not isinstance(group, Mapping):
            continue
        features = group.get("Features") or []
        if not isinstance(features, list):
            continue
        for feature in features:
            if not isinstance(feature, Mapping):
                continue
            name = _feature_name(feature)
            normalized_name = _token(name)
            if not normalized_name:
                continue
            raw_value = _feature_value(feature)
            unit = _feature_unit(feature)
            for field, aliases in _FEATURE_RULES.items():
                priority = aliases.get(normalized_name)
                if priority is None:
                    continue
                value = _convert_metric(field, raw_value, unit, normalized_name)
                if value is None:
                    continue
                evidence = {
                    "feature_id": _mapping(feature.get("Feature")).get("ID")
                    or feature.get("CategoryFeatureId")
                    or feature.get("ID"),
                    "name": name,
                    "raw_value": raw_value,
                    "presentation_value": feature.get("PresentationValue"),
                    "unit": unit or None,
                    "priority": priority,
                }
                candidates[field].append((priority, value, evidence))

    metrics: dict[str, Any] = {}
    provenance: dict[str, dict[str, Any]] = {}
    for field, values in candidates.items():
        if not values:
            continue
        # Prefer an exact/strong semantic match.  If equally strong, use the
        # largest numeric value for rates/capacities and deterministic text for booleans.
        values.sort(key=lambda value: (value[0], _numeric_rank(value[1])), reverse=True)
        _priority, metric_value, evidence = values[0]
        metrics[field] = metric_value
        provenance[field] = evidence
    return metrics, provenance


def replace_specification_source(
    item: Mapping[str, Any],
    specification: Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Replace one enrichment source, preserving supplier/manual fields."""

    source_id = _text(specification.get("source") or ICECAT_SOURCE_ID)
    base = _remove_specification_source(item, source_id)
    enriched = apply_specification_source(base, specification)
    return enriched, enriched != dict(item)


def apply_specification_source(
    item: Mapping[str, Any],
    specification: Mapping[str, Any],
) -> dict[str, Any]:
    """Fill missing identity/metrics and record conflicts without overwriting."""

    result = dict(deepcopy(item))
    source = dict(deepcopy(specification))
    source_id = _text(source.get("source") or ICECAT_SOURCE_ID)
    source_name = _text(source.get("source_name") or source_id)
    observed_at = _text(source.get("observed_at")) or datetime.now(UTC).isoformat()
    source["source"] = source_id
    source["source_name"] = source_name
    source["observed_at"] = observed_at

    provenance = _mapping(result.get("field_provenance"))
    spec_provenance = _mapping(provenance.get("specifications"))
    identity_provenance = _mapping(provenance.get("specification_identity"))

    identity = _mapping(result.get("identity"))
    source_identity = _mapping(source.get("identity"))
    conflicts: dict[str, dict[str, Any]] = {}
    applied_identity: list[str] = []
    for field in ("brand", "mpn", "gtin"):
        incoming = _identity_value(field, source_identity.get(field))
        if not incoming:
            continue
        current = _identity_value(field, identity.get(field))
        if not current:
            identity[field] = incoming
            identity_provenance[field] = _provenance_entry(source, kind="identity")
            applied_identity.append(field)
        elif not _identity_equal(field, current, incoming):
            conflicts[f"identity.{field}"] = {"catalog": current, source_id: incoming}
    result["identity"] = identity

    attributes = _mapping(result.get("attributes"))
    source_metrics = _mapping(source.get("metrics"))
    mapped_features = _mapping(source.get("mapped_features"))
    applied_metrics: list[str] = []
    for field in _METRIC_FIELDS:
        if field not in source_metrics or source_metrics[field] in (None, ""):
            continue
        incoming = source_metrics[field]
        current = attributes.get(field)
        if current in (None, ""):
            attributes[field] = incoming
            evidence = _provenance_entry(source, kind="metric")
            evidence["feature"] = deepcopy(_mapping(mapped_features.get(field)))
            spec_provenance[field] = evidence
            applied_metrics.append(field)
        elif not _metric_equal(current, incoming):
            conflicts[f"attributes.{field}"] = {"catalog": current, source_id: incoming}
    result["attributes"] = attributes

    sources = [
        deepcopy(value)
        for value in result.get("specification_sources", []) or []
        if isinstance(value, Mapping) and _text(value.get("source")) != source_id
    ]
    source["applied_metrics"] = applied_metrics
    source["applied_identity"] = applied_identity
    source["conflicts"] = conflicts
    sources.append(source)
    result["specification_sources"] = sources

    provenance["specifications"] = spec_provenance
    provenance["specification_identity"] = identity_provenance
    result["field_provenance"] = provenance
    result["specification_summary"] = _specification_summary(sources)

    review = _mapping(result.get("review"))
    warnings = [
        str(value)
        for value in review.get("warnings", []) or []
        if str(value) != "Источник характеристик расходится с каталогом."
    ]
    if conflicts:
        warnings.append("Источник характеристик расходится с каталогом.")
    review["warnings"] = list(dict.fromkeys(warnings))
    result["review"] = review
    return result


def carry_specification_sources(
    item: Mapping[str, Any],
    previous_item: Mapping[str, Any],
) -> dict[str, Any]:
    """Reapply enrichment after supplier federation was rebuilt by a refresh."""

    result = dict(deepcopy(item))
    for source in previous_item.get("specification_sources", []) or []:
        if isinstance(source, Mapping):
            result = apply_specification_source(result, source)
    return result


def enrich_staging_records(
    service: Any,
    client: IcecatClient,
    *,
    staging_ids: Iterable[str] | None = None,
    manifest_path: str | Path | None = None,
    request_delay_seconds: float = 0.0,
    progress: Callable[[str], None] | None = None,
) -> IcecatEnrichmentSummary:
    """Enrich eligible staging records and persist updates through the staging service."""

    emit = progress or (lambda _message: None)
    selected = {str(value) for value in (staging_ids or []) if str(value)}
    records = service.list_records()
    targets = [
        record
        for record in records
        if (not selected or str(record.get("staging_id")) in selected)
    ]
    updates: dict[str, dict[str, Any]] = {}
    results: list[dict[str, Any]] = []
    counters = {
        "requested": len(targets),
        "eligible": 0,
        "matched": 0,
        "changed": 0,
        "unchanged": 0,
        "skipped": 0,
        "unavailable": 0,
        "errors": 0,
    }

    for index, record in enumerate(targets, 1):
        staging_id = str(record.get("staging_id") or "")
        status = str(record.get("status") or "")
        source_item = _mapping(record.get("source_catalog_item"))
        lookup_key = icecat_lookup_identity(source_item)
        public_identity = {key: value for key, value in lookup_key.items() if key != "matched_by"}

        if status == "imported":
            counters["skipped"] += 1
            results.append({"staging_id": staging_id, "status": "skipped_imported"})
            continue
        if not lookup_key:
            counters["skipped"] += 1
            results.append({"staging_id": staging_id, "status": "skipped_identity"})
            continue

        counters["eligible"] += 1
        emit(f"Icecat {index}/{len(targets)}: {source_item.get('title') or staging_id}")
        try:
            lookup = client.lookup(source_item)
            counters["matched"] += 1
            enriched, changed = replace_specification_source(source_item, lookup.specification)
            if changed:
                updates[staging_id] = enriched
                counters["changed"] += 1
                state = "changed"
            else:
                counters["unchanged"] += 1
                state = "unchanged"
            results.append(
                {
                    "staging_id": staging_id,
                    "status": state,
                    "matched_by": lookup.matched_by,
                    "requested_identity": public_identity,
                    "icecat_id": lookup.specification.get("icecat_id"),
                    "metrics": sorted(_mapping(lookup.specification.get("metrics"))),
                }
            )
        except IcecatProductUnavailable as exc:
            counters["unavailable"] += 1
            results.append(
                {
                    "staging_id": staging_id,
                    "status": "unavailable",
                    "requested_identity": public_identity,
                    "message": str(exc),
                }
            )
        except IcecatIdentityMismatch as exc:
            counters["errors"] += 1
            results.append(
                {
                    "staging_id": staging_id,
                    "status": "identity_mismatch",
                    "requested_identity": public_identity,
                    "message": str(exc),
                }
            )
        except IcecatConfigurationError:
            # Credentials/configuration is global: continuing would only hammer the API.
            _write_enrichment_manifest(manifest_path, counters, results, fatal="configuration")
            raise
        except IcecatEnrichmentError as exc:
            counters["errors"] += 1
            results.append(
                {
                    "staging_id": staging_id,
                    "status": "error",
                    "requested_identity": public_identity,
                    "message": str(exc),
                }
            )
        if request_delay_seconds > 0 and index < len(targets):
            time.sleep(request_delay_seconds)

    if updates:
        service.apply_source_item_updates(updates)
    _write_enrichment_manifest(manifest_path, counters, results)
    return IcecatEnrichmentSummary(**counters)


def _remove_specification_source(item: Mapping[str, Any], source_id: str) -> dict[str, Any]:
    result = dict(deepcopy(item))
    sources = [
        deepcopy(value)
        for value in result.get("specification_sources", []) or []
        if isinstance(value, Mapping) and _text(value.get("source")) != source_id
    ]
    removed = [
        value
        for value in result.get("specification_sources", []) or []
        if isinstance(value, Mapping) and _text(value.get("source")) == source_id
    ]
    if not removed:
        return result

    provenance = _mapping(result.get("field_provenance"))
    spec_provenance = _mapping(provenance.get("specifications"))
    identity_provenance = _mapping(provenance.get("specification_identity"))
    attributes = _mapping(result.get("attributes"))
    identity = _mapping(result.get("identity"))

    for field in list(spec_provenance):
        field_source = _text(_mapping(spec_provenance.get(field)).get("source"))
        if field_source == source_id:
            attributes.pop(field, None)
            spec_provenance.pop(field, None)
    for field in list(identity_provenance):
        field_source = _text(_mapping(identity_provenance.get(field)).get("source"))
        if field_source == source_id:
            identity.pop(field, None)
            identity_provenance.pop(field, None)

    provenance["specifications"] = spec_provenance
    provenance["specification_identity"] = identity_provenance
    result["field_provenance"] = provenance
    result["attributes"] = attributes
    result["identity"] = identity
    result["specification_sources"] = sources
    result["specification_summary"] = _specification_summary(sources)

    review = _mapping(result.get("review"))
    review["warnings"] = [
        str(value)
        for value in review.get("warnings", []) or []
        if str(value) != "Источник характеристик расходится с каталогом."
    ]
    result["review"] = review
    return result


def _specification_summary(sources: list[dict[str, Any]]) -> dict[str, Any]:
    conflicts: dict[str, Any] = {}
    filled_fields: list[str] = []
    source_names: list[str] = []
    observed: list[str] = []
    for source in sources:
        source_names.append(_text(source.get("source_name") or source.get("source")))
        filled_fields.extend(str(value) for value in source.get("applied_metrics", []) or [])
        observed_at = _text(source.get("observed_at"))
        if observed_at:
            observed.append(observed_at)
        for field, value in _mapping(source.get("conflicts")).items():
            conflicts[field] = deepcopy(value)
    return {
        "source_count": len(sources),
        "sources": list(dict.fromkeys(value for value in source_names if value)),
        "filled_metrics": sorted(set(filled_fields)),
        "conflicts": conflicts,
        "last_observed_at": max(observed) if observed else None,
    }


def _write_enrichment_manifest(
    path: str | Path | None,
    counters: Mapping[str, int],
    results: list[dict[str, Any]],
    *,
    fatal: str | None = None,
) -> None:
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "source": ICECAT_SOURCE_ID,
        "generated_at": datetime.now(UTC).isoformat(),
        "summary": dict(counters),
        "fatal": fatal,
        "results": results,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _raise_for_icecat_status(payload: Mapping[str, Any]) -> None:
    code_value = (
        payload.get("statusCode")
        if payload.get("statusCode") is not None
        else payload.get("StatusCode")
    )
    if code_value in (None, "", 0, "0", 200, "200"):
        message = _text(payload.get("msg"))
        if not message or message.upper() == "OK":
            return
    try:
        code = int(code_value)
    except (TypeError, ValueError):
        code = 0
    message = _text(payload.get("message") or payload.get("Message") or payload.get("msg"))
    if code in _ICECAT_UNAVAILABLE_CODES:
        raise IcecatProductUnavailable(message or f"Icecat: товар недоступен ({code}).")
    if code in _ICECAT_AUTH_ERROR_CODES:
        raise IcecatConfigurationError(message or f"Icecat: ошибка доступа ({code}).")
    if code:
        raise IcecatRequestError(message or f"Icecat вернул ошибку {code}.")


def _icecat_identity(info: Mapping[str, Any]) -> dict[str, Any]:
    brand = _value_text(info.get("Brand"))
    mpn = _text(info.get("BrandPartCode") or info.get("ProductCode"))
    gtins = _gtin_values(info.get("GTIN")) + _gtin_values(info.get("GTINs"))
    result: dict[str, Any] = {}
    if brand:
        result["brand"] = brand
    if mpn:
        result["mpn"] = mpn
    unique_gtins = list(dict.fromkeys(value for value in gtins if value))
    if unique_gtins:
        result["gtin"] = unique_gtins[0]
        result["gtins"] = unique_gtins
    return result


def _verify_identity(
    requested: Mapping[str, Any],
    returned: Mapping[str, Any],
    *,
    matched_by: str,
) -> None:
    if matched_by == "gtin":
        expected = _normalize_gtin(requested.get("gtin"))
        values = {
            _normalize_gtin(value)
            for value in ([returned.get("gtin")] + list(returned.get("gtins") or []))
            if _normalize_gtin(value)
        }
        if not expected or expected not in values:
            raise IcecatIdentityMismatch("Icecat вернул другой GTIN; enrichment отклонён.")
        return

    expected_brand = _token(requested.get("brand"))
    expected_mpn = _token(requested.get("mpn"))
    actual_brand = _token(returned.get("brand"))
    actual_mpn = _token(returned.get("mpn"))
    if not expected_brand or not expected_mpn or expected_brand != actual_brand or expected_mpn != actual_mpn:
        raise IcecatIdentityMismatch("Icecat вернул другой бренд/MPN; enrichment отклонён.")


def _feature_name(feature: Mapping[str, Any]) -> str:
    return _nested_text(feature.get("Feature"), "Name", "Value") or _text(feature.get("Name"))


def _feature_value(feature: Mapping[str, Any]) -> Any:
    for key in ("RawValue", "Value", "LocalValue", "PresentationValue"):
        value = feature.get(key)
        if value not in (None, ""):
            return value
    return None


def _feature_unit(feature: Mapping[str, Any]) -> str:
    feature_meta = _mapping(feature.get("Feature"))
    measure = _mapping(feature_meta.get("Measure"))
    signs = _mapping(measure.get("Signs"))
    return _text(signs.get("_") or measure.get("Sign") or feature_meta.get("Sign"))


def _convert_metric(field: str, value: Any, unit: str, feature_name: str) -> Any:
    if field == "ipv6_support":
        if "protocol" in feature_name and "ipv6" in _text(value).lower():
            return True
        return _boolean(value)

    numbers = _numbers(value)
    if not numbers:
        return None
    number = max(numbers)
    normalized_unit = _token(unit)

    if field in {"ram_gb", "storage_gb"}:
        if normalized_unit in {"tb", "terabyte", "terabytes"}:
            number *= 1024
        elif normalized_unit in {"mb", "megabyte", "megabytes"}:
            number /= 1024
        elif normalized_unit in {"kb", "kilobyte", "kilobytes"}:
            number /= 1024 * 1024
        elif normalized_unit in {"b", "byte", "bytes"}:
            number /= 1024 * 1024 * 1024
        # Unknown/no unit is accepted only because the feature semantic itself is exact.
        return _clean_number(number)
    if field == "max_power_watts":
        if normalized_unit in {"kw", "kilowatt", "kilowatts"}:
            number *= 1000
        elif normalized_unit in {"mw", "milliwatt", "milliwatts"}:
            number /= 1000
        return _clean_number(number)
    if field in {"lan_speed_mbps", "wifi_total_mbps"}:
        unit_text = normalized_unit
        if unit_text in {"gbit s", "gbps", "gbit sec", "gigabit s"}:
            number *= 1000
        elif unit_text in {"kbit s", "kbps", "kbit sec"}:
            number /= 1000
        return _clean_number(number)
    if field in {"cpu_cores", "lan_ports"}:
        return int(round(number)) if number >= 0 else None
    return _clean_number(number)


def _numbers(value: Any) -> list[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, (list, tuple, set)):
        result: list[float] = []
        for part in value:
            result.extend(_numbers(part))
        return result
    text = _text(value).replace("\u00a0", " ").replace(",", " ")
    result = []
    for match in re.findall(r"[-+]?\d+(?:\.\d+)?", text):
        try:
            result.append(float(match))
        except ValueError:
            continue
    return result


def _boolean(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = _text(value).strip().lower()
    if text in {"1", "true", "yes", "y", "supported", "available", "да", "есть"}:
        return True
    if text in {"0", "false", "no", "n", "unsupported", "not supported", "нет"}:
        return False
    return None


def _numeric_rank(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _clean_number(value: float) -> int | float:
    rounded = round(float(value), 4)
    return int(rounded) if rounded.is_integer() else rounded


def _metric_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right or _boolean(left) == _boolean(right)
    try:
        return abs(float(left) - float(right)) < 1e-9
    except (TypeError, ValueError):
        return _text(left) == _text(right)


def _identity_equal(field: str, left: Any, right: Any) -> bool:
    if field == "gtin":
        return _normalize_gtin(left) == _normalize_gtin(right)
    return _token(left) == _token(right)


def _identity_value(field: str, value: Any) -> str:
    return _normalize_gtin(value) if field == "gtin" else _text(value)


def _provenance_entry(source: Mapping[str, Any], *, kind: str) -> dict[str, Any]:
    return {
        "source": _text(source.get("source")),
        "source_name": _text(source.get("source_name")),
        "observed_at": _text(source.get("observed_at")) or None,
        "icecat_id": source.get("icecat_id"),
        "matched_by": _text(source.get("matched_by")) or None,
        "kind": kind,
    }


def _gtin_values(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        direct = value.get("GTIN") or value.get("Value") or value.get("_")
        return _gtin_values(direct) if direct not in (None, "") else []
    if isinstance(value, (list, tuple, set)):
        result: list[str] = []
        for part in value:
            result.extend(_gtin_values(part))
        return result
    normalized = _normalize_gtin(value)
    return [normalized] if normalized else []


def _normalize_gtin(value: Any) -> str:
    digits = re.sub(r"\D+", "", _text(value))
    return digits if 8 <= len(digits) <= 14 else ""


def _nested_text(value: Any, *path: str) -> str:
    current = value
    for key in path:
        if not isinstance(current, Mapping):
            return ""
        current = current.get(key)
    return _text(current)


def _value_text(value: Any) -> str:
    if isinstance(value, Mapping):
        return _text(value.get("Value") or value.get("Name") or value.get("_"))
    return _text(value)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _token(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", _text(value).lower()))


def _requests_session() -> Any:
    try:
        import requests
    except ModuleNotFoundError as exc:  # pragma: no cover - project dependency in normal runtime
        raise IcecatConfigurationError("Для Icecat требуется зависимость requests.") from exc
    return requests.Session()


def api_token_from_environment() -> str:
    return os.environ.get("ICECAT_API_TOKEN", "").strip()


__all__ = [
    "ICECAT_API_URL",
    "ICECAT_SOURCE_ID",
    "IcecatClient",
    "IcecatConfigurationError",
    "IcecatEnrichmentError",
    "IcecatEnrichmentSummary",
    "IcecatIdentityMismatch",
    "IcecatProductUnavailable",
    "IcecatRequestError",
    "apply_specification_source",
    "carry_specification_sources",
    "enrich_staging_records",
    "icecat_lookup_identity",
    "map_icecat_metrics",
    "parse_icecat_specification",
    "replace_specification_source",
]
