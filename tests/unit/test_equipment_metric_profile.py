from application.services.analysis_scope_profile_service import AnalysisScopeProfileService
from application.services.genetic_optimization_service import GeneticOptimizationService
from application.services.runtime_entity_normalization_service import normalize_runtime_row
from domain.decision.ahp.aggregation import aggregate_configuration
from infrastructure.repositories.in_memory_entity_repository import InMemoryEntityRepository


def test_runtime_normalizer_adds_explicit_technical_metric_aliases_and_warnings():
    row = normalize_runtime_row(
        {
            "name": "Router",
            "quantity": 1,
            "max_power": 18,
            "lan_port_count": 4,
            "ethernet_speed_mbps": 1000,
            "wifi_speed_mbps": 1200,
            "ipv6": "да",
        },
        category="network",
    )

    assert row["max_power_watts"] == 18
    assert row["lan_ports"] == 4
    assert row["lan_speed_mbps"] == 1000
    assert row["wifi_total_mbps"] == 1200
    assert row["ipv6_support"] is True
    assert "metric_warnings" not in row


def test_missing_technical_metrics_are_visible_warnings_not_silent_context():
    row = normalize_runtime_row(
        {"name": "Workstation", "quantity": 1, "price": 100, "client_seats": 1},
        category="client",
    )

    assert row["component_type"] == "workstation"
    assert any("ram_gb" in warning for warning in row["metric_warnings"])
    assert any("cpu_cores" in warning for warning in row["metric_warnings"])
    assert any("storage_gb" in warning for warning in row["metric_warnings"])


def test_technical_scope_profile_exposes_network_metrics_for_ga_ahp_pareto():
    profile = AnalysisScopeProfileService().get_profile("technical")

    assert "lan_ports" in profile.criterion_ids()
    assert "lan_speed_mbps" in profile.criterion_ids()
    assert "wifi_total_mbps" in profile.criterion_ids()
    assert "ipv6_support_count" in profile.criterion_ids()


def test_ga_summary_carries_concrete_equipment_metrics_and_metric_warnings():
    repository = InMemoryEntityRepository(
        {
            "network": [
                {
                    "name": "Router",
                    "quantity": 1,
                    "price": 10,
                    "max_power_watts": 18,
                    "lan_ports": 4,
                    "lan_speed_mbps": 1000,
                    "wifi_total_mbps": 1200,
                    "ipv6_support": True,
                }
            ],
            "client": [{"name": "Thin client", "quantity": 1, "price": 20, "client_seats": 1}],
        }
    )
    service = GeneticOptimizationService(repository)

    summary = service.run(
        analysis_scope="technical",
        categories=("network", "client"),
        required_categories=("network",),
        min_selected_items=1,
        ga_params={"pop_size": 4, "generations": 2, "elite": 1, "seed": 3},
    )

    assert summary["totals"]["lan_ports"] >= 4
    assert summary["totals"]["lan_speed_mbps"] >= 1000
    assert isinstance(summary["totals"]["metric_warnings"], list)


def test_ahp_aggregate_preserves_network_metrics():
    aggregate = aggregate_configuration(
        {
            "id": "router_cfg",
            "devices": [
                {
                    "role": "network",
                    "quantity": 2,
                    "lan_ports": 4,
                    "lan_speed_mbps": 1000,
                    "wifi_total_mbps": 1200,
                    "ipv6_support": True,
                    "max_power_watts": 18,
                }
            ],
        }
    )

    assert aggregate["lan_ports"] == 8
    assert aggregate["lan_speed_mbps"] == 1000
    assert aggregate["wifi_total_mbps"] == 2400
    assert aggregate["ipv6_support_count"] == 2


def test_runtime_normalizer_ignores_structured_energy_block_as_power_number():
    row = normalize_runtime_row(
        {
            "name": "Software service",
            "quantity": 1,
            "price": 100,
            "scope": "technical",
            "component_type": "network_device",
            "energy": {"applicable": False},
            "lan_ports": 4,
            "lan_speed_mbps": 1000,
            "wifi_total_mbps": 1200,
            "ipv6_support": True,
        },
        category="network",
    )

    assert "max_power_watts" not in row
    assert row["lan_ports"] == 4
    assert any("max_power_watts" in warning for warning in row["metric_warnings"])
