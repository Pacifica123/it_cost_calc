from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_NUMBER_RE = re.compile(r"(?P<number>\d+(?:[.,]\d+)?)")
_CAPACITY_RE = re.compile(
    r"(?P<number>\d+(?:[.,]\d+)?)\s*(?P<unit>тб|tb|гб|gb)",
    re.IGNORECASE,
)
_SPEED_RE = re.compile(
    r"(?P<number>\d+(?:[.,]\d+)?)\s*(?P<unit>гбит/с|gbps|gbit/s|мбит/с|mbps|mbit/s)",
    re.IGNORECASE,
)


@dataclass(slots=True)
class DnsTechnicalParseResult:
    parsed_metrics: dict[str, Any] = field(default_factory=dict)
    parse_warnings: list[str] = field(default_factory=list)
    confidence: float = 0.0


def _normalized(value: Any) -> str:
    return " ".join(str(value or "").replace("\xa0", " ").strip().lower().split())


def _number(value: Any) -> float | None:
    match = _NUMBER_RE.search(_normalized(value))
    if not match:
        return None
    return float(match.group("number").replace(",", "."))


def _capacity_gb(value: Any) -> int | None:
    match = _CAPACITY_RE.search(_normalized(value))
    if not match:
        return None
    amount = float(match.group("number").replace(",", "."))
    if match.group("unit").lower() in {"тб", "tb"}:
        amount *= 1024
    return round(amount)


def _speed_mbps(value: Any) -> int | None:
    match = _SPEED_RE.search(_normalized(value))
    if not match:
        return None
    amount = float(match.group("number").replace(",", "."))
    if match.group("unit").lower() in {"гбит/с", "gbps", "gbit/s"}:
        amount *= 1000
    return round(amount)


def _boolean(value: Any) -> bool | None:
    text = _normalized(value)
    if text in {"да", "есть", "true", "yes", "поддерживается", "+"}:
        return True
    if text in {"нет", "false", "no", "не поддерживается", "-"}:
        return False
    return None


def parse_dns_technical_specs(specs: dict[str, Any]) -> DnsTechnicalParseResult:
    """Извлечь runtime-метрики из пар ключ/значение карточки товара."""

    result = DnsTechnicalParseResult()
    storage_parts: list[int] = []

    for raw_key, raw_value in specs.items():
        key = _normalized(raw_key)
        value = _normalized(raw_value)
        if not key or not value:
            continue

        if any(marker in key for marker in ("оперативн", "объем ram", "ram capacity")):
            capacity = _capacity_gb(raw_value)
            if capacity is not None:
                result.parsed_metrics.setdefault("ram_gb", capacity)

        if any(marker in key for marker in ("количество ядер", "число ядер", "cpu cores")):
            cores = _number(raw_value)
            if cores is not None:
                result.parsed_metrics.setdefault("cpu_cores", round(cores))

        storage_key = any(
            marker in key
            for marker in ("объем ssd", "объем hdd", "объем накопител", "storage capacity")
        )
        if storage_key:
            capacity = _capacity_gb(raw_value)
            if capacity is not None:
                storage_parts.append(capacity)

        explicit_consumption = (
            "потребляем" in key and "мощ" in key
        ) or "power consumption" in key
        if explicit_consumption:
            watts = _number(raw_value)
            if watts is not None:
                result.parsed_metrics.setdefault("max_power_watts", round(watts))

        if ("lan" in key or "ethernet" in key) and "порт" in key:
            ports = _number(raw_value)
            if ports is not None:
                result.parsed_metrics.setdefault("lan_ports", round(ports))

        if any(marker in key for marker in ("скорость lan", "ethernet speed", "скорость ethernet")):
            speed = _speed_mbps(raw_value)
            if speed is not None:
                result.parsed_metrics.setdefault("lan_speed_mbps", speed)

        if ("wi-fi" in key or "wifi" in key) and "скорост" in key:
            speed = _speed_mbps(raw_value)
            if speed is not None:
                result.parsed_metrics.setdefault("wifi_total_mbps", speed)

        if "ipv6" in key:
            supported = _boolean(raw_value)
            if supported is not None:
                result.parsed_metrics.setdefault("ipv6_support", supported)

    if storage_parts:
        result.parsed_metrics["storage_gb"] = sum(storage_parts)

    result.confidence = (
        round(min(0.98, 0.45 + len(result.parsed_metrics) * 0.08), 2)
        if result.parsed_metrics
        else 0.0
    )
    return result


def merge_technical_metrics_with_attributes(
    attributes: dict[str, Any],
    parse_result: DnsTechnicalParseResult,
) -> dict[str, Any]:
    merged = dict(attributes)
    prior_metrics = dict(merged.get("parsed_metrics") or {})
    prior_warnings = list(merged.get("parse_warnings") or [])
    parsed_metrics = {**prior_metrics, **parse_result.parsed_metrics}

    for key, value in parse_result.parsed_metrics.items():
        if key in merged and merged[key] not in (None, ""):
            prior_warnings.append(f"manual field '{key}' has priority over parsed value")
            continue
        merged[key] = value

    merged["parsed_metrics"] = parsed_metrics
    merged["parse_warnings"] = prior_warnings + parse_result.parse_warnings
    merged["confidence"] = max(float(merged.get("confidence") or 0.0), parse_result.confidence)
    merged["parse_source"] = "dns-product-specs"
    return merged
