from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_WIFI_STANDARD_RE = re.compile(r"802\.11\s*([a-z]+)", re.IGNORECASE)
_LAN_PORTS_RE = re.compile(r"(?P<count>\d+)\s*(?:x\s*)?(?:LAN|лан)\b", re.IGNORECASE)
_WIFI_SPEED_RE = re.compile(r"Wi[-\s]?Fi\s*(?P<speed>\d{2,5})\s*(?:Мбит/с|Mbps|Mbit/s)", re.IGNORECASE)
_ANY_SPEED_RE = re.compile(r"(?P<speed>\d{2,5})\s*(?:Мбит/с|Mbps|Mbit/s)", re.IGNORECASE)

_WIFI_GENERATION_BY_STANDARD = {
    "n": 4,
    "ac": 5,
    "ax": 6,
    "be": 7,
}


@dataclass(slots=True)
class DnsNetworkParseResult:
    """Best-effort результат извлечения сетевых характеристик из названия DNS-Shop."""

    parsed_metrics: dict[str, Any] = field(default_factory=dict)
    parse_warnings: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_attributes(self) -> dict[str, Any]:
        return {
            "parsed_metrics": dict(self.parsed_metrics),
            "parse_warnings": list(self.parse_warnings),
            "confidence": self.confidence,
            "parse_source": "dns-title-best-effort",
        }


def parse_dns_network_title(title: str) -> DnsNetworkParseResult:
    """Извлечь сетевые метрики из названия товара DNS-Shop.

    Парсер намеренно ограничен: он не пытается понимать все форматы карточек,
    не выбрасывает исключения при незнакомом названии и помечает спорные скорости
    предупреждениями. Ручные поля из specs обрабатываются на этапе merge в builder.
    """

    normalized = str(title or "").strip()
    result = DnsNetworkParseResult()
    if not normalized:
        result.parse_warnings.append("empty title; network metrics were not parsed")
        return result

    lower = normalized.lower()
    network_like = any(marker in lower for marker in ("wi-fi", "wifi", "роутер", "router", "ipv6", "lan"))
    if not network_like:
        result.parse_warnings.append("title does not look like a network device")
        return result

    lan_match = _LAN_PORTS_RE.search(normalized)
    if lan_match:
        result.parsed_metrics["lan_ports"] = int(lan_match.group("count"))
    else:
        result.parse_warnings.append("LAN ports were not found in title")

    wifi_standards = sorted({match.group(1).lower() for match in _WIFI_STANDARD_RE.finditer(normalized)})
    if wifi_standards:
        result.parsed_metrics["wifi_standards"] = wifi_standards
        generations = [_WIFI_GENERATION_BY_STANDARD.get(standard) for standard in wifi_standards]
        known_generations = [generation for generation in generations if generation is not None]
        if known_generations:
            result.parsed_metrics["wifi_generation_rank"] = max(known_generations)
    else:
        result.parse_warnings.append("Wi-Fi standards were not found in title")

    wifi_speed_match = _WIFI_SPEED_RE.search(normalized)
    if wifi_speed_match:
        result.parsed_metrics["wifi_total_mbps"] = int(wifi_speed_match.group("speed"))
    else:
        result.parse_warnings.append("explicit Wi-Fi total speed was not found in title")

    speeds = [int(match.group("speed")) for match in _ANY_SPEED_RE.finditer(normalized)]
    wifi_speed = result.parsed_metrics.get("wifi_total_mbps")
    candidate_lan_speeds = [speed for speed in speeds if speed != wifi_speed]
    if candidate_lan_speeds:
        result.parsed_metrics["lan_speed_mbps"] = max(candidate_lan_speeds)
        if len(set(candidate_lan_speeds)) > 1:
            result.parse_warnings.append(
                "multiple non-Wi-Fi speeds were found; lan_speed_mbps uses the maximum value"
            )
    else:
        result.parse_warnings.append("LAN speed was not found in title")

    if "ipv6" in lower:
        result.parsed_metrics["ipv6_support"] = True
    else:
        result.parse_warnings.append("IPv6 support was not found in title")

    parsed_count = len(result.parsed_metrics)
    warning_penalty = min(len(result.parse_warnings) * 0.08, 0.32)
    result.confidence = round(max(0.15, min(0.95, 0.20 + parsed_count * 0.14 - warning_penalty)), 2)
    return result


def merge_parsed_metrics_with_specs(
    specs: dict[str, Any],
    parse_result: DnsNetworkParseResult,
) -> dict[str, Any]:
    """Вернуть attributes с приоритетом ручных полей над распознанными.

    Прямые поля из specs не перезаписываются результатом парсера, но сам блок
    parsed_metrics сохраняется для прозрачности и ручной проверки.
    """

    merged = dict(specs)
    parsed_metrics = dict(parse_result.parsed_metrics)
    parse_warnings = list(parse_result.parse_warnings)

    for key, value in parsed_metrics.items():
        if key in merged and merged[key] not in (None, ""):
            parse_warnings.append(f"manual field '{key}' has priority over parsed value")
            continue
        merged[key] = value

    merged.update(
        {
            "parsed_metrics": parsed_metrics,
            "parse_warnings": parse_warnings,
            "confidence": parse_result.confidence,
            "parse_source": "dns-title-best-effort",
        }
    )
    return merged
