import numpy as np

from application.services.candidate_configuration_service import CandidateConfigurationService
from domain.decision.ahp.aggregation import aggregate_configuration
from domain.decision.ahp.pipeline import run_ahp_pipeline
from ui.tabs.ahp_analysis.matrix_utils import extract_default_expert_matrices, default_pairwise_value


def test_ahp_aggregate_exposes_profile_metrics_from_devices_and_candidate_payload():
    cfg = {
        "id": "cfg",
        "devices": [
            {
                "role": "client",
                "quantity": 2,
                "ram_gb": 16,
                "cpu_cores": 8,
                "storage_gb": 512,
                "max_power": 120,
                "cost": 1000,
            },
            {
                "role": "licenses",
                "quantity": 10,
                "functionality_score": 4.5,
                "support_score": 4.0,
                "cost": 500,
            },
        ],
        "totals": {"capital_cost": 1500, "software_license_quantity": 10},
    }

    aggregate = aggregate_configuration(cfg)

    assert aggregate["capital_cost"] == 1500
    assert aggregate["total_ram_gb"] == 32
    assert aggregate["total_cpu_cores"] == 16
    assert aggregate["total_storage_gb"] == 1024
    assert aggregate["total_power_watts"] == 120
    assert aggregate["functionality_score"] == 4.5
    assert aggregate["support_score"] == 4.0


def test_candidate_pool_to_ahp_preserves_totals_and_metrics_for_profile_criteria():
    service = CandidateConfigurationService()
    [config] = service.to_ahp_configurations(
        [
            {
                "id": "ga_1",
                "name": "GA candidate",
                "scope": "technical",
                "components": [{"source_category": "client", "quantity": 1, "price": 10}],
                "totals": {"capital_cost": 10, "total_ram_gb": 16},
                "metrics": {"total_cpu_cores": 8},
                "metadata": {"source": "test"},
            }
        ]
    )

    assert config["totals"]["total_ram_gb"] == 16
    assert config["metrics"]["total_cpu_cores"] == 8
    assert config["metadata"]["source"] == "test"


def test_profile_unknown_default_expert_matrices_are_neutral_equal_weight():
    matrices = extract_default_expert_matrices(["total_ram_gb", "total_cpu_cores"])

    assert len(matrices) == 2
    assert np.allclose(matrices[0], np.ones((2, 2)))
    assert default_pairwise_value("total_ram_gb", "total_cpu_cores") == 1.0


def test_ahp_pipeline_uses_min_direction_as_lower_is_better_when_supplied():
    configs = [
        {
            "id": "cheap",
            "devices": [{"role": "client", "cost": 100, "ram_gb": 16}],
            "totals": {"capital_cost": 100, "total_ram_gb": 16},
        },
        {
            "id": "expensive",
            "devices": [{"role": "client", "cost": 500, "ram_gb": 16}],
            "totals": {"capital_cost": 500, "total_ram_gb": 16},
        },
    ]

    report = run_ahp_pipeline(
        configurations=configs,
        soft_criteria=["capital_cost", "total_ram_gb"],
        experts_criteria_matrices=[np.ones((2, 2)), np.ones((2, 2))],
        constraints={"max_budget": 1000},
        criterion_directions={"capital_cost": "min", "total_ram_gb": "max"},
    )

    assert report["criteria"]["directions"]["capital_cost"] == "min"
    assert report["final"]["winner_id"] == "cheap"
    assert report["alternatives"]["directed_scores"][0][0] > report["alternatives"]["directed_scores"][0][1]
