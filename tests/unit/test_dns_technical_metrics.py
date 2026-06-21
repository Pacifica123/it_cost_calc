from tools.catalog_parser.dns_technical_metrics import parse_dns_technical_specs


def test_parse_dns_technical_specs_normalizes_units() -> None:
    result = parse_dns_technical_specs(
        {
            "Объем оперативной памяти": "32 ГБ",
            "Количество ядер процессора": "12",
            "Объем SSD": "1 ТБ",
            "Объем HDD": "500 ГБ",
            "Максимальная потребляемая мощность": "275 Вт",
            "Количество LAN портов": "4",
            "Скорость LAN": "2.5 Гбит/с",
            "Максимальная скорость Wi-Fi": "2402 Мбит/с",
            "Поддержка IPv6": "Да",
        }
    )
    assert result.parsed_metrics == {
        "ram_gb": 32,
        "cpu_cores": 12,
        "max_power_watts": 275,
        "lan_ports": 4,
        "lan_speed_mbps": 2500,
        "wifi_total_mbps": 2402,
        "ipv6_support": True,
        "storage_gb": 1524,
    }
    assert result.confidence > 0.9


def test_nominal_psu_power_is_not_treated_as_device_consumption() -> None:
    result = parse_dns_technical_specs({"Мощность блока питания": "750 Вт"})
    assert "max_power_watts" not in result.parsed_metrics
