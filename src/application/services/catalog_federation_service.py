"""Deterministic federation of catalog items from independent data sources.

The staging layer keeps one source observation per supplier/feed and builds a
single reviewable product around shared identity.  Identity matching is strict:
GTIN, then brand+MPN, then brand+model.  Titles alone never merge products.
"""

from __future__ import annotations

import hashlib
import re
import statistics
from copy import deepcopy
from datetime import UTC, datetime
from typing import Any, Iterable, Mapping

_IDENTITY_PRIORITY = ("gtin", "brand_mpn", "brand_model")
_PRICE_KIND_PRIORITY = {
    "commercial_quote": 60,
    "supplier_price": 50,
    "retail_offer": 40,
    "contract_price": 35,
    "procurement_benchmark": 30,
    "historical_price": 20,
    "estimated_price": 10,
}
_AVAILABLE_VALUES = {
    "1",
    "true",
    "yes",
    "available",
    "in_stock",
    "instock",
    "есть",
    "в наличии",
    "доступно",
}
_UNAVAILABLE_VALUES = {
    "0",
    "false",
    "no",
    "unavailable",
    "out_of_stock",
    "outofstock",
    "нет",
    "нет в наличии",
}


def federate_catalog_items(items: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Merge source observations into stable multi-source catalog items."""

    observations = [_leaf_observation(item) for item in items if isinstance(item, Mapping)]
    if not observations:
        return []

    parents = list(range(len(observations)))
    key_owner: dict[str, int] = {}

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left == root_right:
            return
        if root_left > root_right:
            root_left, root_right = root_right, root_left
        parents[root_right] = root_left

    for index, item in enumerate(observations):
        for _kind, key in identity_candidates(item):
            previous = key_owner.get(key)
            if previous is None:
                key_owner[key] = index
            else:
                union(index, previous)

    groups: dict[int, list[dict[str, Any]]] = {}
    for index, item in enumerate(observations):
        groups.setdefault(find(index), []).append(item)

    return [_merge_group(group) for group in groups.values()]


def identity_candidates(item: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    """Return ordered strict identity keys for one normalized catalog item."""

    identity = _mapping(item.get("identity"))
    brand = _token(identity.get("brand"))
    gtin = _gtin(identity.get("gtin"))
    mpn = _token(identity.get("mpn"))
    model = _token(identity.get("model"))
    result: list[tuple[str, str]] = []
    if gtin:
        result.append(("gtin", f"gtin:{gtin}"))
    if brand and mpn:
        result.append(("brand_mpn", f"brand_mpn:{brand}:{mpn}"))
    if brand and model:
        result.append(("brand_model", f"brand_model:{brand}:{model}"))
    return tuple(result)


def source_observations(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Recover leaf source rows stored inside a federated item."""

    raw = item.get("source_observations")
    if isinstance(raw, list):
        result = [
            _leaf_observation(value)
            for value in raw
            if isinstance(value, Mapping)
        ]
        if result:
            return result
    return [_leaf_observation(item)]


def observation_keys(item: Mapping[str, Any]) -> set[str]:
    """Keys used only to preserve staging/manual-review continuity on refresh."""

    result: set[str] = set()
    federation = _mapping(item.get("federation"))
    for value in federation.get("observation_keys", []) or []:
        text = str(value or "").strip()
        if text:
            result.add(text)
    if result:
        return result
    for observation in source_observations(item):
        result.add(_observation_key(observation))
    return result


def select_effective_offer(offers: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Choose one backward-compatible effective offer.

    Policy: usable RUB price -> available stock -> newest observation ->
    higher-trust price kind -> lower price as a deterministic final tie-breaker.
    """

    candidates = [
        dict(deepcopy(offer))
        for offer in offers
        if _positive_number(offer.get("price")) is not None
        and str(offer.get("currency") or "RUB").strip().upper() == "RUB"
    ]
    if not candidates:
        return {}

    def rank(offer: Mapping[str, Any]) -> tuple[int, float, int, float, str]:
        price = _positive_number(offer.get("price")) or 0.0
        return (
            _availability_rank(offer.get("availability")),
            _timestamp(offer.get("observed_at")),
            _PRICE_KIND_PRIORITY.get(str(offer.get("price_kind") or "").strip(), 0),
            -price,
            str(offer.get("source") or ""),
        )

    selected = max(candidates, key=rank)
    selected["freshness"] = offer_freshness(selected)
    return selected


def offer_freshness(offer: Mapping[str, Any], *, now: datetime | None = None) -> str:
    value = _parse_datetime(offer.get("observed_at"))
    if value is None:
        return "unknown"
    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    age_days = max(0.0, (current.astimezone(UTC) - value).total_seconds() / 86400.0)
    if age_days <= 30:
        return "fresh"
    if age_days <= 90:
        return "aging"
    return "stale"


def _merge_group(group: list[dict[str, Any]]) -> dict[str, Any]:
    offers = _collect_offers(group)
    effective = select_effective_offer(offers)
    representative = _representative(group, effective)

    identity, identity_conflicts = _merge_identity(group, representative)
    categories = _unique_text(item.get("category") for item in group)
    sources = _unique_text(item.get("source") for item in group)
    matched_by = _shared_identity_kinds(group)
    canonical_kind, canonical_key = _canonical_identity(group)

    merged = dict(deepcopy(representative))
    merged["item_id"] = (
        str(representative.get("item_id") or "")
        if len(sources) == 1
        else _federated_item_id(canonical_key)
    )
    merged["source"] = sources[0] if len(sources) == 1 else "federated"
    if len(sources) > 1:
        merged["source_product_id"] = None
    merged["identity"] = identity
    merged["attributes"] = _merge_attributes(group, representative)
    merged["offers"] = offers
    merged["offer"] = effective or _mapping(representative.get("offer"))
    merged["price_summary"] = _price_summary(offers, effective)
    merged["source_observations"] = [_leaf_observation(item) for item in group]
    merged["federation"] = {
        "identity_key": canonical_key,
        "identity_kind": canonical_kind,
        "matched_by": matched_by,
        "sources": sources,
        "source_count": len(sources),
        "observation_count": len(group),
        "offer_count": len(offers),
        "observation_keys": sorted(_observation_key(item) for item in group),
        "identity_conflicts": identity_conflicts,
        "category_conflicts": categories if len(categories) > 1 else [],
    }

    provenance = _mapping(merged.get("field_provenance"))
    provenance["federation"] = {
        "sources": sources,
        "matched_by": matched_by,
        "effective_offer_source": effective.get("source") if effective else None,
        "effective_offer_observed_at": effective.get("observed_at") if effective else None,
    }
    merged["field_provenance"] = provenance

    review = _mapping(merged.get("review"))
    warnings = list(review.get("warnings") or [])
    if identity_conflicts:
        warnings.append("Источники расходятся по идентификаторам товара.")
    if len(categories) > 1:
        warnings.append("Источники расходятся по категории товара.")
    review["warnings"] = list(dict.fromkeys(str(value) for value in warnings if value))
    merged["review"] = review
    return merged


def _collect_offers(group: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    offers: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for item in group:
        source = str(item.get("source") or "").strip()
        source_product_id = str(item.get("source_product_id") or "").strip()
        source_name = _source_name(item)
        candidates = item.get("offers")
        raw_offers = candidates if isinstance(candidates, list) and candidates else [item.get("offer")]
        for raw in raw_offers:
            if not isinstance(raw, Mapping):
                continue
            offer = dict(deepcopy(raw))
            offer.setdefault("source", source)
            offer.setdefault("source_name", source_name)
            offer.setdefault("source_product_id", source_product_id or None)
            offer.setdefault("price_kind", "retail_offer")
            offer["currency"] = str(offer.get("currency") or "RUB").upper()
            key = (
                offer.get("source"),
                offer.get("source_product_id"),
                offer.get("url"),
                offer.get("price"),
                offer.get("currency"),
                offer.get("observed_at"),
                offer.get("region"),
            )
            if key in seen:
                continue
            seen.add(key)
            offer["offer_id"] = _offer_id(offer)
            offer["freshness"] = offer_freshness(offer)
            offers.append(offer)
    return sorted(
        offers,
        key=lambda offer: (
            str(offer.get("source") or ""),
            str(offer.get("source_product_id") or ""),
            str(offer.get("observed_at") or ""),
            str(offer.get("offer_id") or ""),
        ),
    )


def _representative(
    group: list[dict[str, Any]],
    effective: Mapping[str, Any],
) -> dict[str, Any]:
    source = str(effective.get("source") or "")
    product_id = str(effective.get("source_product_id") or "")
    for item in group:
        if str(item.get("source") or "") != source:
            continue
        if product_id and str(item.get("source_product_id") or "") != product_id:
            continue
        return item
    return max(
        group,
        key=lambda item: (
            _timestamp(_mapping(item.get("offer")).get("observed_at")),
            bool(item.get("identity")),
            bool(item.get("attributes")),
            str(item.get("source") or ""),
        ),
    )


def _merge_identity(
    group: list[dict[str, Any]],
    representative: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    ordered = [representative, *[item for item in group if item is not representative]]
    result: dict[str, Any] = {}
    conflicts: dict[str, list[str]] = {}
    for field in ("brand", "model", "mpn", "gtin"):
        values = _unique_text(_mapping(item.get("identity")).get(field) for item in ordered)
        if values:
            result[field] = values[0]
        normalized = {_identity_value(field, value) for value in values if value}
        normalized.discard("")
        if len(normalized) > 1:
            conflicts[field] = values
    return result, conflicts


def _merge_attributes(
    group: list[dict[str, Any]],
    representative: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(deepcopy(_mapping(representative.get("attributes"))))
    for item in group:
        for key, value in _mapping(item.get("attributes")).items():
            if key not in result or result[key] in (None, "", [], {}):
                if value not in (None, "", [], {}):
                    result[key] = deepcopy(value)
    return result


def _shared_identity_kinds(group: list[Mapping[str, Any]]) -> list[str]:
    counts: dict[str, dict[str, int]] = {kind: {} for kind in _IDENTITY_PRIORITY}
    for item in group:
        for kind, key in identity_candidates(item):
            counts[kind][key] = counts[kind].get(key, 0) + 1
    return [
        kind
        for kind in _IDENTITY_PRIORITY
        if any(count >= 2 for count in counts[kind].values())
    ]


def _canonical_identity(group: list[Mapping[str, Any]]) -> tuple[str, str]:
    candidates: dict[str, list[str]] = {kind: [] for kind in _IDENTITY_PRIORITY}
    for item in group:
        for kind, key in identity_candidates(item):
            candidates[kind].append(key)
    for kind in _IDENTITY_PRIORITY:
        if candidates[kind]:
            return kind, sorted(set(candidates[kind]))[0]
    fallback = sorted(_observation_key(item) for item in group)[0]
    return "source_item", f"source_item:{fallback}"


def _price_summary(
    offers: list[Mapping[str, Any]],
    effective: Mapping[str, Any],
) -> dict[str, Any]:
    rub = [
        float(value)
        for offer in offers
        if str(offer.get("currency") or "RUB").upper() == "RUB"
        if (value := _positive_number(offer.get("price"))) is not None
    ]
    timestamps = [
        value
        for offer in offers
        if (value := _parse_datetime(offer.get("observed_at"))) is not None
    ]
    return {
        "observation_count": len(rub),
        "source_count": len({str(offer.get("source") or "") for offer in offers if offer.get("source")}),
        "min_rub": min(rub) if rub else None,
        "median_rub": statistics.median(rub) if rub else None,
        "max_rub": max(rub) if rub else None,
        "oldest_observed_at": min(timestamps).isoformat() if timestamps else None,
        "newest_observed_at": max(timestamps).isoformat() if timestamps else None,
        "effective_offer_id": effective.get("offer_id") if effective else None,
        "effective_source": effective.get("source") if effective else None,
        "freshness": effective.get("freshness") if effective else "unknown",
    }


def _source_name(item: Mapping[str, Any]) -> str:
    feed = _mapping(_mapping(item.get("field_provenance")).get("feed"))
    return str(feed.get("source_name") or item.get("source") or "").strip()


def _leaf_observation(item: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(deepcopy(item))
    for key in ("offers", "price_summary", "federation", "source_observations"):
        result.pop(key, None)
    return result


def _observation_key(item: Mapping[str, Any]) -> str:
    source = str(item.get("source") or "unknown").strip()
    identity = _mapping(item.get("identity"))
    value = (
        item.get("source_product_id")
        or item.get("item_id")
        or identity.get("gtin")
        or identity.get("mpn")
        or _mapping(item.get("offer")).get("url")
        or item.get("title")
        or "unknown"
    )
    return f"{source}:{str(value).strip()}"


def _federated_item_id(identity_key: str) -> str:
    digest = hashlib.sha1(identity_key.encode("utf-8")).hexdigest()[:16]
    return f"catalog-fed-{digest}"


def _offer_id(offer: Mapping[str, Any]) -> str:
    payload = "|".join(
        str(offer.get(key) or "")
        for key in ("source", "source_product_id", "url", "observed_at", "price", "currency", "region")
    )
    return "offer-" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _availability_rank(value: Any) -> int:
    text = str(value or "unknown").strip().casefold()
    if text in _AVAILABLE_VALUES:
        return 2
    if text in _UNAVAILABLE_VALUES:
        return 0
    return 1


def _timestamp(value: Any) -> float:
    parsed = _parse_datetime(value)
    return parsed.timestamp() if parsed is not None else 0.0


def _parse_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _positive_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number <= 0:
        return None
    return number


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _unique_text(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        normalized = text.casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(text)
    return result


def _identity_value(field: str, value: Any) -> str:
    return _gtin(value) if field == "gtin" else _token(value)


def _gtin(value: Any) -> str:
    digits = re.sub(r"\D+", "", str(value or ""))
    return digits if len(digits) >= 8 else ""


def _token(value: Any) -> str:
    return re.sub(r"[^0-9a-zа-я]+", "", str(value or "").casefold())


__all__ = [
    "federate_catalog_items",
    "identity_candidates",
    "observation_keys",
    "offer_freshness",
    "select_effective_offer",
    "source_observations",
]
