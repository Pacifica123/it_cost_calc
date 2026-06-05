from tools.catalog_parser.catalog_builder import normalize_dns_snapshot
from tools.catalog_parser.dns_network_metrics import (
    merge_parsed_metrics_with_specs,
    parse_dns_network_title,
)


ROUTER_TITLE = (
    "Wi-Fi роутер TP-Link Archer Air R5 "
    "[1 LAN, 1000 Мбит/с, 100 Мбит/с, 4 (802.11n), "
    "5 (802.11ac), 6 (802.11ax), Wi-Fi 2976 Мбит/с, IPv6]"
)


def test_parse_dns_network_title_extracts_router_metrics() -> None:
    result = parse_dns_network_title(ROUTER_TITLE)

    assert result.parsed_metrics["lan_ports"] == 1
    assert result.parsed_metrics["lan_speed_mbps"] == 1000
    assert result.parsed_metrics["wifi_total_mbps"] == 2976
    assert result.parsed_metrics["wifi_standards"] == ["ac", "ax", "n"]
    assert result.parsed_metrics["wifi_generation_rank"] == 6
    assert result.parsed_metrics["ipv6_support"] is True
    assert result.confidence > 0.6
    assert any("multiple non-Wi-Fi speeds" in warning for warning in result.parse_warnings)


def test_parse_dns_network_title_unknown_format_returns_warning_not_exception() -> None:
    result = parse_dns_network_title("Неизвестное устройство без сетевых характеристик")

    assert result.parsed_metrics == {}
    assert result.parse_warnings
    assert result.confidence == 0.0


def test_merge_parsed_metrics_keeps_manual_specs_priority() -> None:
    result = parse_dns_network_title(ROUTER_TITLE)
    attributes = merge_parsed_metrics_with_specs({"lan_speed_mbps": 2500}, result)

    assert attributes["lan_speed_mbps"] == 2500
    assert attributes["parsed_metrics"]["lan_speed_mbps"] == 1000
    assert any("manual field 'lan_speed_mbps'" in warning for warning in attributes["parse_warnings"])


def test_normalize_dns_snapshot_adds_parsed_metrics_to_router_attributes() -> None:
    items = normalize_dns_snapshot(
        {
            "routers": [
                {
                    "title": ROUTER_TITLE,
                    "url": "https://example.test/router",
                    "price_int": 7999,
                    "specs": {},
                }
            ]
        },
        snapshot_name="routers.json",
    )

    attrs = items[0].attributes
    assert attrs["lan_ports"] == 1
    assert attrs["wifi_total_mbps"] == 2976
    assert attrs["ipv6_support"] is True
    assert attrs["parse_source"] == "dns-title-best-effort"
