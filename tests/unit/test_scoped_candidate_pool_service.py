from application.services.candidate_configuration_service import CandidateConfigurationService
from application.services.scoped_candidate_pool_service import ScopedCandidatePoolService
from shared.constants import ANALYSIS_SCOPE_TECHNICAL


def _candidate(candidate_id, *, cost, ram, power, seats):
    return {
        "id": candidate_id,
        "name": candidate_id,
        "scope": ANALYSIS_SCOPE_TECHNICAL,
        "components": [
            {
                "name": f"{candidate_id} workstation",
                "source_category": "client",
                "client_seats": seats,
                "cost": cost,
                "max_power": power,
                "ram_gb": ram,
            }
        ],
        "totals": {
            "capital_cost": cost,
            "client_seats": seats,
            "ram_gb": ram,
        },
        "metrics": {
            "raw_scores_by_criterion": {
                "total_ram_gb": ram,
                "total_power_watts": power,
                "capital_cost": cost,
            }
        },
        "source": "ga",
        "metadata": {"rank": 1},
    }


def test_scoped_candidate_pool_builds_ahp_configurations_with_people_from_client_seats():
    service = ScopedCandidatePoolService()
    service.replace(
        ANALYSIS_SCOPE_TECHNICAL,
        [_candidate("GA-1", cost=100.0, ram=16.0, power=80.0, seats=3)],
        source_label="top-1 GA-кандидатов",
        source_method="ga",
    )

    configs = service.to_ahp_configurations(ANALYSIS_SCOPE_TECHNICAL)

    assert configs[0]["id"] == "GA-1"
    assert configs[0]["meta"]["people"] == 3
    assert configs[0]["devices"][0]["energy"] == 80.0


def test_scoped_candidate_pool_builds_pareto_case_and_inverts_min_criteria():
    service = ScopedCandidatePoolService()
    service.replace(
        ANALYSIS_SCOPE_TECHNICAL,
        [
            _candidate("cheap", cost=100.0, ram=16.0, power=80.0, seats=3),
            _candidate("expensive", cost=200.0, ram=32.0, power=150.0, seats=3),
        ],
        source_label="top-2 GA-кандидатов",
        source_method="ga",
    )

    case_data = service.to_criteria_case(ANALYSIS_SCOPE_TECHNICAL)

    assert [item["id"] for item in case_data["alternatives"]] == ["cheap", "expensive"]
    assert case_data["scores"]["expensive"]["total_ram_gb"] > case_data["scores"]["cheap"]["total_ram_gb"]
    assert "capital_cost" not in case_data["scores"]["cheap"]
    assert case_data["scores"]["cheap"]["total_power_watts"] > case_data["scores"]["expensive"]["total_power_watts"]
    assert case_data["metadata"]["score_scale"] == "1..5 utility, min criteria inverted"


def test_candidate_configuration_service_derives_software_people_from_license_quantity():
    service = CandidateConfigurationService()
    config = service.to_ahp_configuration(
        {
            "id": "sw",
            "name": "software candidate",
            "scope": "software",
            "components": [],
            "totals": {"software_license_quantity": 25},
            "metrics": {},
            "source": "ga",
            "metadata": {},
        }
    )

    assert config["meta"]["people"] == 25
